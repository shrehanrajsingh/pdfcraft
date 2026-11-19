use super::{
    Automation, Value,
    error::{Result, failed},
    json::Args,
};
use pdfcraft_engine::ai;
use pdfcraft_engine::ai::{AgentAnswer, EvidencePack, EvidenceRequest, Focus, Task};
#[cfg(not(target_arch = "wasm32"))]
use pdfcraft_engine::ai::{Config, ImageChoice, Role, Turn};
use serde_json::json;

#[cfg(not(target_arch = "wasm32"))]
fn image_block(send_images: bool, accepts_images: bool) -> Option<&'static str> {
    if !send_images {
        Some("Page images are off. The text of those pages is included instead.")
    } else if !accepts_images {
        Some("Page images were not sent because this model is not marked as accepting images. The text of those pages is included instead.")
    } else {
        None
    }
}

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

    pub(crate) fn ai_index(&self, a: &Args) -> Result<Value> {
        let doc = self.doc(a)?;
        let pages = self.ai_pages(a)?;
        let _ = doc.ai_source(&pages).map_err(failed)?;
        let outline = doc.ai_outline().map_err(failed)?;
        let list = if pages.is_empty() { (0..doc.info.pages.len()).collect::<Vec<_>>() } else { pages };
        let mut indexed = Vec::new();
        for page in list {
            let index = doc.ai_page(page).map_err(failed)?;
            indexed.push(json!({
                "page": page + 1,
                "truncated": index.truncated,
                "spans": index.spans.iter().map(span_json).collect::<Vec<_>>(),
            }));
        }
        Ok(json!({
            "outline": outline.iter().map(|item| json!({"title": item.title, "page": item.page.map(|page| page + 1), "depth": item.depth})).collect::<Vec<_>>(),
            "pages": indexed,
        }))
    }

    pub(crate) fn ai_search(&self, a: &Args) -> Result<Value> {
        let doc = self.doc(a)?;
        let query = a.str("query")?;
        let pages = self.ai_pages(a)?;
        let hits = doc.ai_search(query, &pages).map_err(failed)?;
        Ok(json!({"hits": hits.iter().map(span_json).collect::<Vec<_>>()}))
    }

    pub(crate) fn ai_request(&self, name: &str, a: &Args) -> Result<Value> {
        // Consent is per call. A model-controlled tool argument cannot choose a destination
        // for the launcher-held API key: URL, model and key all come from launch configuration.
        if !a.opt_bool("allow_send")?.unwrap_or(false) {
            return Err(failed("Set allow_send=true only after approving this provider request. Use ai_preview to inspect document text first."));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (config, key) = configured();
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
        let send_images = config.send_page_images;
        let accepts_images = config.model_accepts_images;
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
        if task == Task::Translate {
            let pages = self.ai_pages(a)?;
            let prepared = self.doc(a)?.ai_source(&pages).map_err(failed)?.prepare().map_err(failed)?;
            let answer = prepared.complete(&mut provider, task, input).map_err(failed)?;
            return Ok(json!({"answer":answer,"pages":prepared.pages,"source_bytes":prepared.text.len(),"citations_verified":false}));
        }
        let question = if task == Task::Summary && input.is_empty() { "Summarize".to_string() } else { input.to_string() };
        let (source, outline, request) = self.evidence_request(a, task, &question)?;
        let choice = ai::image_choice(send_images, accepts_images, &request.question, request.focus.region.is_some());
        let (pack, images, note) = ai::prepare_pack(&source, &outline, &request, &choice).map_err(failed)?;
        let mut notebook = ai::PageNotebook::default();
        let answer = ai::complete_evidence(
            &mut provider,
            &source,
            &outline,
            pack.clone(),
            &mut notebook,
            task,
            input,
            &[],
            &images,
            image_block(send_images, accepts_images),
            {
                let source = source.clone();
                let outline = outline.clone();
                let base = request.clone();
                move |query| {
                    let mut again = base.clone();
                    again.question = query.to_string();
                    Ok(ai::prepare_pack(&source, &outline, &again, &ImageChoice::Off)?.0)
                }
            },
            |_| {},
            |_| {},
        )
        .map_err(failed)?;
        Ok(agent_json(&answer, &pack, note, images.len()))
    }

    pub(crate) fn ai_chat(&self, a: &Args) -> Result<Value> {
        if !a.opt_bool("allow_send")?.unwrap_or(false) {
            return Err(failed("Set allow_send=true only after approving this provider request. Use ai_preview to inspect document text first."));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (config, key) = configured();
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
        let send_images = config.send_page_images;
        let accepts_images = config.model_accepts_images;
        let mut provider = ai::OpenAiProvider::new(config, key).map_err(failed)?;
        let snapshot = self.chat_images(a.get("images"))?;
        if task == Task::Translate {
            let pages = self.ai_pages(a)?;
            let prepared = self.doc(a)?.ai_source(&pages).map_err(failed)?.prepare().map_err(failed)?;
            let answer = prepared.complete_conversation(&mut provider, task, input, &turns, &snapshot).map_err(failed)?;
            return Ok(
                json!({"answer":answer,"pages":prepared.pages,"source_bytes":prepared.text.len(),"citations_verified":false,"images":snapshot.len()}),
            );
        }
        let question = turns
            .last()
            .map(|turn| turn.text.clone())
            .filter(|text| !text.trim().is_empty())
            .unwrap_or_else(|| if input.is_empty() { "Summarize".to_string() } else { input.to_string() });
        let (source, outline, mut request) = self.evidence_request(a, task, &question)?;
        if !input.is_empty() && !request.question.contains(input) {
            request.question = format!("{input}\n{}", request.question);
        }
        let choice = ai::image_choice(send_images, accepts_images, &request.question, request.focus.region.is_some());
        let (pack, mut images, note) = ai::prepare_pack(&source, &outline, &request, &choice).map_err(failed)?;
        for image in &snapshot {
            if images.len() >= ai::MAX_CHAT_IMAGES_PER_TURN {
                break;
            }
            images.push(image.clone());
        }
        let mut notebook = ai::PageNotebook::default();
        let answer = ai::complete_evidence(
            &mut provider,
            &source,
            &outline,
            pack.clone(),
            &mut notebook,
            task,
            input,
            &turns,
            &images,
            image_block(send_images, accepts_images),
            {
                let source = source.clone();
                let outline = outline.clone();
                let base = request.clone();
                move |query| {
                    let mut again = base.clone();
                    again.question = query.to_string();
                    Ok(ai::prepare_pack(&source, &outline, &again, &ImageChoice::Off)?.0)
                }
            },
            |_| {},
            |_| {},
        )
        .map_err(failed)?;
        let mut value = agent_json(&answer, &pack, note, images.len());
        if let Some(object) = value.as_object_mut() {
            object.insert("images".into(), serde_json::json!(images.len()));
        }
        Ok(value)
    }

    fn evidence_request(
        &self,
        a: &Args,
        task: Task,
        question: &str,
    ) -> Result<(pdfcraft_engine::export::ExportSource, Vec<ai::OutlineEntry>, EvidenceRequest)> {
        let doc = self.doc(a)?;
        let pages = self.ai_pages(a)?;
        let _ = doc.ai_source(&pages).map_err(failed)?;
        let outline = doc.ai_outline().map_err(failed)?;
        let mut source = doc.export_source();
        source.config.system_fonts = false;
        let current = match a.get("page") {
            Some(value) => {
                let page = value.as_u64().ok_or_else(|| failed("page must be a 1-based page number."))?;
                let page = usize::try_from(page)
                    .map_err(|_| failed("page must be a 1-based page number."))?
                    .checked_sub(1)
                    .ok_or_else(|| failed("page must be a 1-based page number."))?;
                if page >= doc.info.pages.len() {
                    return Err(failed("page is outside the document."));
                }
                Some(page)
            }
            None => None,
        };
        let focus = focus_from(a.get("selection"))?;
        Ok((source, outline, EvidenceRequest { task, question: question.to_string(), pages, current_page: current, focus }))
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
            turns.push(Turn { role: Role::from_name(role).map_err(failed)?, text: content.to_owned(), images, citations: Vec::new() });
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

#[cfg(not(target_arch = "wasm32"))]
fn configured() -> (Config, String) {
    let config = Config {
        enabled: std::env::var("PDFCRAFT_AI_ENABLED").is_ok_and(|value| value == "1"),
        base_url: std::env::var("PDFCRAFT_AI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".into()),
        model: std::env::var("PDFCRAFT_AI_MODEL").unwrap_or_default(),
        legacy_max_tokens: std::env::var("PDFCRAFT_AI_LEGACY_TOKENS").is_ok_and(|value| value == "1"),
        send_page_images: std::env::var("PDFCRAFT_AI_PAGE_IMAGES").is_ok_and(|value| value == "1"),
        model_accepts_images: std::env::var("PDFCRAFT_AI_MODEL_IMAGES").is_ok_and(|value| value == "1"),
        stream_responses: std::env::var("PDFCRAFT_AI_STREAM").is_ok_and(|value| value == "1"),
    };
    (config, std::env::var("PDFCRAFT_AI_API_KEY").unwrap_or_default())
}

fn span_json(span: &ai::Span) -> Value {
    json!({
        "id": span.id,
        "page": span.page + 1,
        "kind": span.kind.as_str(),
        "text": span.text,
        "rect": span.rect,
        "rects": span.rects,
    })
}

fn agent_json(answer: &AgentAnswer, pack: &EvidencePack, note: Option<String>, images: usize) -> Value {
    let mut notes = answer.notes.clone();
    if let Some(note) = note {
        notes.push(note);
    }
    json!({
        "answer": answer.answer,
        "pages": pack.pages,
        "source_bytes": pack.text.len(),
        "citations_verified": true,
        "citations": answer.citations.iter().map(|cite| json!({
            "id": cite.id,
            "page": cite.page + 1,
            "span_id": cite.span_id,
            "quote": cite.quote,
            "rects": cite.rects,
            "page_level": cite.page_level,
        })).collect::<Vec<_>>(),
        "notes": notes,
        "images": images,
    })
}

fn focus_from(value: Option<&Value>) -> Result<Focus> {
    let Some(selection) = value else { return Ok(Focus::default()) };
    if !selection.is_object() {
        return Err(failed("selection must be an object."));
    }
    let text = selection.get("text").and_then(Value::as_str).unwrap_or("").to_string();
    if text.len() > ai::MAX_PROMPT_BYTES {
        return Err(failed(format!("selection text must fit in {} UTF-8 bytes.", ai::MAX_PROMPT_BYTES)));
    }
    let page = match selection.get("page") {
        Some(value) => Some(
            usize::try_from(value.as_u64().ok_or_else(|| failed("selection page must be a 1-based page number."))?)
                .map_err(|_| failed("selection page must be a 1-based page number."))?
                .checked_sub(1)
                .ok_or_else(|| failed("selection page must be a 1-based page number."))?,
        ),
        None => None,
    };
    let glyph = |key: &str| -> Result<Option<usize>> {
        match selection.get(key) {
            Some(value) => Ok(Some(
                usize::try_from(value.as_u64().ok_or_else(|| failed(format!("selection {key} must be a glyph index.")))?)
                    .map_err(|_| failed(format!("selection {key} must be a glyph index.")))?,
            )),
            None => Ok(None),
        }
    };
    let start = glyph("start")?;
    let end = glyph("end")?;
    let ranges = match (page, start, end) {
        (Some(page), Some(start), Some(end)) if end > start => vec![(page, start, end)],
        (Some(_), Some(_), Some(_)) => return Err(failed("selection end must be after start.")),
        _ => Vec::new(),
    };
    let rect = match selection.get("rect") {
        Some(value) => {
            let nums = value.as_array().ok_or_else(|| failed("selection rect must be [x0, y0, x1, y1]."))?;
            if nums.len() != 4 {
                return Err(failed("selection rect must be [x0, y0, x1, y1]."));
            }
            let mut out = [0.0_f32; 4];
            for (i, num) in nums.iter().enumerate() {
                let n = num.as_f64().ok_or_else(|| failed("selection rect must be [x0, y0, x1, y1]."))?;
                if !n.is_finite() {
                    return Err(failed("selection rect must be finite."));
                }
                out[i] = n as f32;
            }
            Some(out)
        }
        None => None,
    };
    Ok(Focus { ranges, region_page: page.filter(|_| rect.is_some()), region: rect, text })
}
