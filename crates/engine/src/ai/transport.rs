use super::agent::visible_stream_text;
use super::{ChatMessage, Config};
use std::io::BufRead;

/// Which part of a provider reply to show while it is still arriving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    /// The model returns the answer itself, as in Translate.
    Plain,
    /// The model returns a JSON object. Only the `answer` string is shown.
    Answer,
}
use base64::Engine as _;
use craft_ai::{AiError, GenOpts, Provider, ProviderInfo};
use serde_json::{Value, json};
use zeroize::Zeroizing;

/// PdfCraft adapter for the shared provider trait. Redirects must never resend document text.
/// See docs/ai-assistant.md for the shared OpenAiCompat transport API gap.
pub struct OpenAiProvider {
    config: Config,
    key: Zeroizing<String>,
    endpoint: url::Url,
}

impl OpenAiProvider {
    pub fn new(config: Config, key: String) -> Result<Self, String> {
        let key = Zeroizing::new(key);
        config.validate_connection()?;
        if !config.model.trim().is_empty() {
            config.require_model()?;
        }
        if key.len() > 8192 || key.bytes().any(|b| !(0x21..=0x7e).contains(&b)) {
            return Err("API key must contain at most 8192 printable ASCII characters, without spaces or line breaks.".into());
        }
        let endpoint = config.endpoint()?;
        if endpoint.host_str() == Some("api.openai.com") && key.is_empty() {
            return Err("Enter an API key for OpenAI. A ChatGPT subscription is not an API key.".into());
        }
        Ok(Self { config, key, endpoint })
    }

    fn agent() -> ureq::Agent {
        ureq::Agent::config_builder()
            // Even a redirect to the same host could disclose text outside the configured API.
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_global(Some(std::time::Duration::from_secs(120)))
            .timeout_connect(Some(std::time::Duration::from_secs(10)))
            .build()
            .into()
    }

    /// Generation has no total time limit. A reasoning model can stream for longer than the
    /// connection test. Connecting, and waiting for the response headers, still time out.
    fn completion_agent() -> ureq::Agent {
        ureq::Agent::config_builder()
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_connect(Some(std::time::Duration::from_secs(10)))
            .timeout_recv_response(Some(std::time::Duration::from_secs(120)))
            .build()
            .into()
    }

    fn url(&self, path: &str) -> Result<String, AiError> {
        self.endpoint.join(path).map(|u| u.to_string()).map_err(|_| AiError::NotConfigured)
    }

    fn response(&self, mut response: ureq::http::Response<ureq::Body>, listing: bool) -> Result<Value, AiError> {
        let code = response.status().as_u16();
        // Provider bodies and transport diagnostics may echo the credential or document.
        // Report bounded, actionable status messages instead of forwarding those strings.
        if !(200..300).contains(&code) {
            return Err(AiError::Rejected(format!("HTTP {code}: {}.", Self::status_help(code, listing))));
        }
        let bytes =
            response.body_mut().with_config().limit(u64::MAX).read_to_vec().map_err(|_| AiError::BadResponse("response unreadable".into()))?;
        serde_json::from_slice(&bytes).map_err(|_| AiError::BadResponse("expected a JSON response".into()))
    }

    fn status_help(code: u16, listing: bool) -> &'static str {
        match (code, listing) {
            (300..=399, _) => "redirect refused; configure the final API base URL",
            (401 | 403, _) => "authentication failed; check the API key from platform.openai.com (a ChatGPT subscription is not an API key)",
            (404, true) => {
                "model listing was not found at this URL. Keep https://api.openai.com/v1, enter a model ID such as gpt-4o-mini (not ChatGPT), and try Send"
            }
            (404, false) => "that model was not found; for OpenAI use gpt-4o-mini or gpt-4o, not ChatGPT",
            (429, _) => "rate limit or quota exceeded; check the provider account and try later",
            (400 | 422, _) => "request rejected; check the model",
            (500..=599, _) => "provider unavailable; try again later",
            _ => "request rejected by the provider",
        }
    }

    fn answer(&self, value: Value) -> Result<String, AiError> {
        let choice = value.pointer("/choices/0").ok_or_else(|| AiError::BadResponse("no completion returned".into()))?;
        let reason = choice.get("finish_reason").and_then(Value::as_str);
        match reason {
            Some("stop") | Some("length") | None => {}
            _ => return Err(AiError::BadResponse("provider did not finish a text answer (refusal or unsupported tool call)".into())),
        }
        let answer = choice.pointer("/message/content").and_then(Value::as_str).unwrap_or("");
        if answer.trim().is_empty() {
            return Err(AiError::BadResponse(if reason == Some("length") {
                "answer exceeded the output limit; choose fewer pages or a shorter question".into()
            } else {
                "provider returned no text; check the model or reduce the context".into()
            }));
        }
        // A stop at the output limit still has the text already written. Keep it.
        // A malicious or misconfigured endpoint can reflect Authorization into its answer.
        Ok(self.redact(answer))
    }

    fn redact(&self, text: &str) -> String {
        if self.key.is_empty() { text.to_string() } else { text.replace(self.key.as_str(), "[API key removed]") }
    }
}

struct StreamAccum<'a> {
    raw_body: &'a mut String,
    content: &'a mut String,
    sse: &'a mut Option<bool>,
    stopped_for_length: &'a mut bool,
    shown: &'a mut String,
}

fn network_error(_: ureq::Error) -> AiError {
    AiError::Http("connection failed or timed out; check the URL, network, and TLS certificate".into())
}

impl OpenAiProvider {
    fn completion_body(&self, messages: Vec<Value>, stream: bool) -> Result<Value, AiError> {
        if let Err(e) = self.config.require_model() {
            return Err(AiError::Rejected(e));
        }
        // No max_tokens or max_completion_tokens. The provider stops at its own context window.
        // Temperature is omitted because reasoning models reject sampler parameters.
        Ok(json!({"model": self.config.model.trim(), "messages": messages, "stream": stream}))
    }

    fn send_completion(&self, body: Value) -> Result<ureq::http::Response<ureq::Body>, AiError> {
        let mut req = Self::completion_agent().post(self.url("chat/completions")?);
        if !self.key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.key.as_str()));
        }
        req.send_json(body).map_err(network_error)
    }

    fn post_completion(&mut self, messages: Vec<Value>) -> Result<String, AiError> {
        let body = self.completion_body(messages, false)?;
        let response = self.send_completion(body)?;
        self.completion_text(response)
    }

    /// A chat body that is a completion is returned as its text. Anything else that arrived,
    /// including a body that is not JSON, is the answer itself.
    fn completion_text(&self, mut response: ureq::http::Response<ureq::Body>) -> Result<String, AiError> {
        let code = response.status().as_u16();
        if !(200..300).contains(&code) {
            return Err(AiError::Rejected(format!("HTTP {code}: {}.", Self::status_help(code, false))));
        }
        let bytes =
            response.body_mut().with_config().limit(u64::MAX).read_to_vec().map_err(|_| AiError::BadResponse("response unreadable".into()))?;
        let text = String::from_utf8_lossy(&bytes);
        self.text_or_completion(text.trim())
    }

    fn text_or_completion(&self, text: &str) -> Result<String, AiError> {
        if let Ok(value) = serde_json::from_str::<Value>(text)
            && value.pointer("/choices/0").is_some()
        {
            return self.answer(value);
        }
        let shown = self.redact(text);
        if shown.trim().is_empty() {
            return Err(AiError::BadResponse("provider returned no text; check the model or reduce the context".into()));
        }
        Ok(shown)
    }

    fn chat_payload(&self, messages: &[ChatMessage]) -> Result<Vec<Value>, AiError> {
        let mut total = 0usize;
        for message in messages {
            if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
                return Err(AiError::Rejected("conversation messages must use system, user, or assistant roles".into()));
            }
            total = total.saturating_add(message.text.len());
            for image in &message.images {
                total = total.saturating_add(image.bytes().len().saturating_mul(2));
            }
        }
        let image_budget = super::MAX_CHAT_IMAGES.saturating_mul(super::MAX_CHAT_IMAGE_BYTES.saturating_mul(2));
        if messages.is_empty() || total > super::MAX_TEXT_BYTES + super::MAX_HISTORY_BYTES + 2048 + image_budget {
            return Err(AiError::Rejected("request exceeds the text limit".into()));
        }
        messages.iter().map(message_json).collect()
    }

    /// Multi-turn `/chat/completions` body. Document text belongs in the first user message.
    /// Pictures use OpenAI vision `image_url` data URLs only when a turn has them, so text-only
    /// servers keep a string `content`. This waits for the full answer.
    pub fn complete_chat(&mut self, messages: &[ChatMessage]) -> Result<String, AiError> {
        let payload = self.chat_payload(messages)?;
        self.post_completion(payload)
    }

    /// When streaming is enabled, `on_text` receives the visible answer as it arrives and the
    /// returned string is still the full reply. When streaming is off, `on_text` is not called.
    pub fn complete_chat_reporting<F>(&mut self, messages: &[ChatMessage], kind: StreamKind, on_text: &mut F) -> Result<String, AiError>
    where
        F: FnMut(&str),
    {
        let payload = self.chat_payload(messages)?;
        if !self.config.stream_responses {
            return self.post_completion(payload);
        }
        let body = self.completion_body(payload, true)?;
        let response = self.send_completion(body)?;
        self.read_stream(response, kind, on_text)
    }

    fn read_stream<F>(&self, mut response: ureq::http::Response<ureq::Body>, kind: StreamKind, on_text: &mut F) -> Result<String, AiError>
    where
        F: FnMut(&str),
    {
        let code = response.status().as_u16();
        if !(200..300).contains(&code) {
            return Err(AiError::Rejected(format!("HTTP {code}: {}.", Self::status_help(code, false))));
        }
        on_text("");
        // Invalid bytes are shown as text. A size limit here used to discard a long reply.
        let reader = response.body_mut().with_config().limit(u64::MAX).lossy_utf8(true).reader();
        let mut reader = std::io::BufReader::new(reader);
        let mut line = String::new();
        let mut raw_body = String::new();
        let mut content = String::new();
        let mut shown = String::new();
        let mut sse = None;
        let mut stopped_for_length = false;
        let mut failure = None;
        loop {
            line.clear();
            let mut accum = StreamAccum {
                raw_body: &mut raw_body,
                content: &mut content,
                sse: &mut sse,
                stopped_for_length: &mut stopped_for_length,
                shown: &mut shown,
            };
            // A dropped connection keeps every piece already shown.
            if reader.read_line(&mut line).is_err() {
                if !line.is_empty() {
                    let _ = self.accept_stream_line(&line, &mut accum, kind, on_text);
                }
                break;
            }
            if line.is_empty() {
                break;
            }
            if let Err(err) = self.accept_stream_line(&line, &mut accum, kind, on_text) {
                failure = Some(err);
                break;
            }
            if line.trim() == "data: [DONE]" {
                break;
            }
        }
        if sse != Some(true) {
            return self.deliver_body(raw_body.trim(), kind, on_text);
        }
        if !content.trim().is_empty() {
            return Ok(self.redact(&content));
        }
        if let Some(err) = failure {
            return Err(err);
        }
        if stopped_for_length {
            return Err(AiError::BadResponse("answer exceeded the output limit; choose fewer pages or a shorter question".into()));
        }
        Err(AiError::BadResponse("provider returned no text; check the model or reduce the context".into()))
    }

    /// One line of a streamed body. A server that ignores streaming is buffered until the end.
    /// A `data:` line that is not JSON is appended as text.
    fn accept_stream_line<F>(&self, line: &str, accum: &mut StreamAccum<'_>, kind: StreamKind, on_text: &mut F) -> Result<(), AiError>
    where
        F: FnMut(&str),
    {
        if *accum.sse != Some(true) {
            accum.raw_body.push_str(line);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        if accum.sse.is_none() {
            *accum.sse = Some(trimmed.starts_with("data:"));
        }
        if *accum.sse != Some(true) {
            return Ok(());
        }
        let Some(data) = trimmed.strip_prefix("data:") else { return Ok(()) };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(());
        }
        if serde_json::from_str::<Value>(data).is_err() {
            Self::append_plain(accum.content, data);
            self.show_content(accum.content, kind, accum.shown, on_text);
            return Ok(());
        }
        match self.push_stream_chunk(data, accum.content, accum.stopped_for_length, kind, accum.shown, on_text) {
            Ok(()) => Ok(()),
            // A refusal after text has already arrived keeps that text.
            Err(_) if !accum.content.trim().is_empty() => Ok(()),
            Err(err) => Err(err),
        }
    }

    fn append_plain(content: &mut String, data: &str) {
        if !content.is_empty() && !content.ends_with('\n') {
            content.push('\n');
        }
        content.push_str(data);
    }

    fn deliver_body<F>(&self, raw: &str, kind: StreamKind, on_text: &mut F) -> Result<String, AiError>
    where
        F: FnMut(&str),
    {
        let answer = self.text_or_completion(raw)?;
        let mut shown = String::new();
        self.show_content(&answer, kind, &mut shown, on_text);
        Ok(answer)
    }

    fn show_content<F>(&self, content: &str, kind: StreamKind, shown: &mut String, on_text: &mut F)
    where
        F: FnMut(&str),
    {
        let next = self.redact(&match kind {
            StreamKind::Plain => content.to_string(),
            StreamKind::Answer => visible_stream_text(content),
        });
        if !next.is_empty() && next != *shown {
            *shown = next;
            on_text(shown);
        }
    }

    fn push_stream_chunk<F>(
        &self,
        data: &str,
        content: &mut String,
        stopped_for_length: &mut bool,
        kind: StreamKind,
        shown: &mut String,
        on_text: &mut F,
    ) -> Result<(), AiError>
    where
        F: FnMut(&str),
    {
        let value: Value = serde_json::from_str(data).map_err(|_| AiError::BadResponse("expected a JSON response".into()))?;
        if value.get("error").is_some() {
            return Err(AiError::BadResponse("provider did not finish a text answer (refusal or unsupported tool call)".into()));
        }
        match value.pointer("/choices/0/finish_reason").and_then(Value::as_str) {
            Some("stop") | None => {}
            Some("length") => *stopped_for_length = true,
            _ => return Err(AiError::BadResponse("provider did not finish a text answer (refusal or unsupported tool call)".into())),
        }
        if let Some(piece) = value.pointer("/choices/0/delta/content").and_then(Value::as_str) {
            content.push_str(piece);
        } else if let Some(full) = value.pointer("/choices/0/message/content").and_then(Value::as_str)
            && (content.is_empty() || full.starts_with(content.as_str()))
        {
            content.clear();
            content.push_str(full);
        }
        let next = self.redact(&match kind {
            StreamKind::Plain => content.clone(),
            StreamKind::Answer => visible_stream_text(content),
        });
        if !next.is_empty() && next != *shown {
            *shown = next;
            on_text(shown);
        }
        Ok(())
    }
}

fn message_json(message: &ChatMessage) -> Result<Value, AiError> {
    if message.images.is_empty() {
        return Ok(json!({"role": message.role, "content": message.text}));
    }
    let mut parts = vec![json!({"type": "text", "text": message.text})];
    for image in &message.images {
        if image.bytes().len() > super::MAX_CHAT_IMAGE_BYTES {
            return Err(AiError::Rejected("an attached image exceeds the size limit".into()));
        }
        let url = format!("data:{};base64,{}", image.mime(), base64::engine::general_purpose::STANDARD.encode(image.bytes()));
        parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
    }
    Ok(json!({"role": message.role, "content": parts}))
}

impl Provider for OpenAiProvider {
    fn name(&self) -> String {
        "OpenAI-compatible".into()
    }

    fn complete(&mut self, prompt: &str, _opts: &GenOpts) -> Result<String, AiError> {
        if prompt.len() > super::MAX_TEXT_BYTES + super::MAX_PROMPT_BYTES + 2048 {
            return Err(AiError::Rejected("request exceeds the text limit".into()));
        }
        self.post_completion(vec![json!({"role": "user", "content": prompt})])
    }

    fn test(&mut self) -> Result<ProviderInfo, AiError> {
        let mut req = Self::agent().get(self.url("models")?);
        if !self.key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.key.as_str()));
        }
        let value = self.response(req.call().map_err(network_error)?, true)?;
        let entries = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| AiError::BadResponse("expected a models data array; enter the model manually if listing is unsupported".into()))?;
        let models = entries
            .iter()
            .take(256)
            .filter_map(|v| v.get("id").and_then(Value::as_str))
            .filter(|s| s.len() <= 256 && !s.chars().any(char::is_control) && (self.key.is_empty() || !s.contains(self.key.as_str())))
            .map(str::to_owned)
            .collect();
        Ok(ProviderInfo { label: "Connected. Model listing succeeded; generation has not been tested.".into(), models })
    }

    fn is_local(&self) -> bool {
        // A loopback server may itself forward to a cloud service. Do not promise locality.
        false
    }
}
