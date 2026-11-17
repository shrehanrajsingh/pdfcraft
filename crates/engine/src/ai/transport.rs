use super::{ChatMessage, Config};
use base64::Engine as _;
use craft_ai::{AiError, GenOpts, Provider, ProviderInfo};
use serde_json::{Value, json};
use zeroize::Zeroizing;

const MAX_RESPONSE: u64 = 1024 * 1024;

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
        let bytes = response
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE)
            .read_to_vec()
            .map_err(|_| AiError::BadResponse("response unreadable or exceeds the 1 MiB limit".into()))?;
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
            (400 | 422, _) => "request rejected; check the model and try the legacy token-limit option if your server requires it",
            (500..=599, _) => "provider unavailable; try again later",
            _ => "request rejected by the provider",
        }
    }

    fn answer(&self, value: Value) -> Result<String, AiError> {
        let choice = value.pointer("/choices/0").ok_or_else(|| AiError::BadResponse("no completion returned".into()))?;
        match choice.get("finish_reason").and_then(Value::as_str) {
            Some("stop") | None => {}
            Some("length") => return Err(AiError::BadResponse("answer exceeded the output limit; choose fewer pages or a shorter question".into())),
            _ => return Err(AiError::BadResponse("provider did not finish a text answer (refusal or unsupported tool call)".into())),
        }
        let answer = choice
            .pointer("/message/content")
            .and_then(Value::as_str)
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| AiError::BadResponse("provider returned no text; check the model or reduce the context".into()))?;
        // A malicious or misconfigured endpoint can reflect Authorization into its answer.
        Ok(if self.key.is_empty() { answer.into() } else { answer.replace(self.key.as_str(), "[API key removed]") })
    }
}

fn network_error(_: ureq::Error) -> AiError {
    AiError::Http("connection failed or timed out; check the URL, network, and TLS certificate".into())
}

impl OpenAiProvider {
    fn post_completion(&mut self, messages: Vec<Value>, opts: &GenOpts) -> Result<String, AiError> {
        if let Err(e) = self.config.require_model() {
            return Err(AiError::Rejected(e));
        }
        let mut body = json!({"model": self.config.model.trim(), "messages": messages, "stream": false});
        // Omitting temperature supports reasoning models which reject sampler parameters.
        let limit = if self.config.legacy_max_tokens { "max_tokens" } else { "max_completion_tokens" };
        body[limit] = json!(opts.max_tokens.clamp(1, 8192));
        let mut req = Self::agent().post(self.url("chat/completions")?);
        if !self.key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.key.as_str()));
        }
        let response = req.send_json(body).map_err(network_error)?;
        self.answer(self.response(response, false)?)
    }

    /// Multi-turn `/chat/completions` body. Document text belongs in the first user message.
    /// Pictures use OpenAI vision `image_url` data URLs only when a turn has them, so text-only
    /// servers keep a string `content`.
    pub fn complete_chat(&mut self, messages: &[ChatMessage]) -> Result<String, AiError> {
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
        let payload = messages.iter().map(message_json).collect::<Result<Vec<_>, _>>()?;
        self.post_completion(payload, &GenOpts::default())
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

    fn complete(&mut self, prompt: &str, opts: &GenOpts) -> Result<String, AiError> {
        if prompt.len() > super::MAX_TEXT_BYTES + super::MAX_PROMPT_BYTES + 2048 {
            return Err(AiError::Rejected("request exceeds the text limit".into()));
        }
        self.post_completion(vec![json!({"role": "user", "content": prompt})], opts)
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
