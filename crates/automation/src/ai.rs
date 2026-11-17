use super::{
    Automation, Value,
    error::{Result, failed},
    json::Args,
};
use pdfcraft_engine::ai;
#[cfg(not(target_arch = "wasm32"))]
use pdfcraft_engine::ai::{Config, Role, Task, Turn};
use serde_json::json;

impl Automation {
    fn ai_pages(&self, a: &Args) -> Result<Vec<usize>> {
        if a.get("pages").and_then(Value::as_array).is_some_and(|p| p.len() > ai::MAX_PAGES) {
            return Err(failed("AI requests accept at most 100 pages."));
        }
        if a.get("pages").is_some() { self.pages(a, "pages") } else { Ok(vec![]) }
    }

    pub(crate) fn ai_preview(&self, a: &Args) -> Result<Value> {
        let doc = self.doc(a)?;
        let pages = self.ai_pages(a)?;
        let text = doc.ai_source(&pages).map_err(failed)?.prepare().map_err(failed)?;
        Ok(json!({"pages":text.pages,"bytes":text.text.len(),"text":text.text}))
    }

    pub(crate) fn ai_request(&self, name: &str, a: &Args) -> Result<Value> {
        // Consent is per call. A model-controlled tool argument cannot choose a destination
        // for the launcher-held API key: URL, model and key all come from launch configuration.
        if !a.opt_bool("allow_send")?.unwrap_or(false) {
            return Err(failed("Set allow_send=true only after approving this provider request. Use ai_preview to inspect document text first."));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let config = Config {
                enabled: std::env::var("PDFCRAFT_AI_ENABLED").is_ok_and(|s| s == "1"),
                base_url: std::env::var("PDFCRAFT_AI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into()),
                model: std::env::var("PDFCRAFT_AI_MODEL").unwrap_or_default(),
                legacy_max_tokens: std::env::var("PDFCRAFT_AI_LEGACY_TOKENS").is_ok_and(|s| s == "1"),
            };
            let key = std::env::var("PDFCRAFT_AI_API_KEY").unwrap_or_default();
            self.ai_request_with(name, a, config, key)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = name;
            Err(failed("AI requests require a native CLI build."))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ai_request_with(&self, name: &str, a: &Args, config: Config, key: String) -> Result<Value> {
        let mut provider = ai::OpenAiProvider::new(config, key).map_err(failed)?;
        if name == "ai_provider_test" {
            return Ok(json!({"connection": ai::test_provider(&mut provider).map_err(failed)?}));
        }
        let task = match name {
            "ai_summary" => Task::Summary,
            "ai_ask" => Task::Ask,
            "ai_translate" => Task::Translate,
            _ => return Err(failed("Unknown AI task.")),
        };
        let input = a.opt_str("input")?.unwrap_or("");
        task.instruction(input).map_err(failed)?;
        let pages = self.ai_pages(a)?;
        let prepared = self.doc(a)?.ai_source(&pages).map_err(failed)?.prepare().map_err(failed)?;
        let answer = prepared.complete(&mut provider, task, input).map_err(failed)?;
        Ok(json!({"answer":answer,"pages":prepared.pages,"source_bytes":prepared.text.len(),"citations_verified":false}))
    }

    pub(crate) fn ai_chat(&self, a: &Args) -> Result<Value> {
        if !a.opt_bool("allow_send")?.unwrap_or(false) {
            return Err(failed("Set allow_send=true only after approving this provider request. Use ai_preview to inspect document text first."));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let config = Config {
                enabled: std::env::var("PDFCRAFT_AI_ENABLED").is_ok_and(|s| s == "1"),
                base_url: std::env::var("PDFCRAFT_AI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into()),
                model: std::env::var("PDFCRAFT_AI_MODEL").unwrap_or_default(),
                legacy_max_tokens: std::env::var("PDFCRAFT_AI_LEGACY_TOKENS").is_ok_and(|s| s == "1"),
            };
            let key = std::env::var("PDFCRAFT_AI_API_KEY").unwrap_or_default();
            self.ai_chat_with(a, config, key)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(failed("AI requests require a native CLI build."))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn ai_chat_with(&self, a: &Args, config: Config, key: String) -> Result<Value> {
        let task = Task::from_name(a.opt_str("task")?.unwrap_or("ask")).map_err(failed)?;
        let input = a.opt_str("input")?.unwrap_or("");
        let turns = self.chat_turns(a)?;
        let mut provider = ai::OpenAiProvider::new(config, key).map_err(failed)?;
        let pages = self.ai_pages(a)?;
        let prepared = self.doc(a)?.ai_source(&pages).map_err(failed)?.prepare().map_err(failed)?;
        let snapshot = self.chat_images(a.get("images"))?;
        let answer = prepared.complete_conversation(&mut provider, task, input, &turns, &snapshot).map_err(failed)?;
        Ok(json!({"answer":answer,"pages":prepared.pages,"source_bytes":prepared.text.len(),"citations_verified":false,"images":snapshot.len()}))
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn chat_turns(&self, a: &Args) -> Result<Vec<Turn>> {
        let entries = a.get("messages").and_then(Value::as_array).ok_or_else(|| failed("messages must be an array of {role, content} objects."))?;
        if entries.is_empty() {
            return Err(failed("messages must end with the new user turn."));
        }
        let mut turns = Vec::new();
        for entry in entries {
            let role = entry.get("role").and_then(Value::as_str).ok_or_else(|| failed("message role must be user or assistant."))?;
            let content = entry.get("content").and_then(Value::as_str).ok_or_else(|| failed("message content must be a string."))?;
            let images = self.chat_images(entry.get("images"))?;
            turns.push(Turn { role: Role::from_name(role).map_err(failed)?, text: content.to_owned(), images });
        }
        Ok(turns)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn chat_images(&self, value: Option<&Value>) -> Result<Vec<ai::ChatImage>> {
        let Some(entries) = value.and_then(Value::as_array) else {
            return Ok(Vec::new());
        };
        if entries.len() > ai::MAX_CHAT_IMAGES_PER_TURN {
            return Err(failed(format!("Attach at most {} images in one message.", ai::MAX_CHAT_IMAGES_PER_TURN)));
        }
        let mut images = Vec::new();
        for entry in entries {
            let path = entry.as_str().ok_or_else(|| failed("images must be file paths."))?;
            let resolved = self.resolve(path, false)?;
            let bytes = crate::read_input(&resolved)?;
            images.push(ai::ChatImage::from_encoded(bytes).map_err(failed)?);
        }
        Ok(images)
    }
}
