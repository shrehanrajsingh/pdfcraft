use super::{
    Automation, Value,
    error::{Result, failed},
    json::Args,
};
use pdfcraft_engine::ai;
#[cfg(not(target_arch = "wasm32"))]
use pdfcraft_engine::ai::{Config, Task};
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
}
