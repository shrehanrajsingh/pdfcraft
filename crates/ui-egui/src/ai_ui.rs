//! Settings and chat are exclusive inspector views. Background work is one job per window.
use crate::theme::{self, Tokens};
use crate::{Dialog, PdfCraftApp, RightPanel, icons, widgets};
use egui::{Align, CornerRadius, Layout, RichText, Stroke};
use pdfcraft_engine::{
    DocId,
    ai::{
        self, AgentAnswer, ChatImage, Config, EvidencePack, EvidenceRequest, Focus, ImageChoice, PageNotebook, Prepared, Role, Task, Turn,
        ViewerAction,
    },
};
use std::sync::{Arc, Mutex, PoisonError};
use zeroize::Zeroizing;

struct PendingImage {
    encoded: ChatImage,
    preview: Arc<egui::ColorImage>,
}

enum ResultData {
    Preview(Prepared),
    Evidence { pack: EvidencePack, images: Vec<ChatImage>, note: Option<String> },
    Answer(String),
    Agent(AgentAnswer),
    Connection(String),
}

struct Job {
    revision: u64,
    pending: Option<(Turn, String)>,
    receiver: std::sync::mpsc::Receiver<Result<ResultData, String>>,
    activity: Arc<Mutex<String>>,
    /// Visible answer text while a streamed reply is still arriving.
    partial: Arc<Mutex<String>>,
}

/// A selection the user asked about. It stays until they clear it or open another document.
struct Attachment {
    text: String,
    page: usize,
    ranges: Vec<(usize, usize, usize)>,
    region: bool,
    rect: Option<[f32; 4]>,
}

#[derive(Default)]
pub(crate) struct Assistant {
    pub(crate) config: Config,
    key: Zeroizing<String>,
    task: Task,
    input: String,
    starter: String,
    current_page: bool,
    doc: Option<(DocId, u64)>,
    preview: Option<Prepared>,
    /// Viewed page index of a refresh that already ran. Stops a failed refresh from restarting every frame.
    preview_attempt: Option<usize>,
    consent: bool,
    messages: Vec<Turn>,
    status: String,
    revision: u64,
    job: Option<Job>,
    /// Inspector gear: provider fields. Off by default; forced on while AI is disabled.
    pub(crate) settings_open: bool,
    pending_images: Vec<PendingImage>,
    /// Pictures sent with the first request, attached to the hidden document message.
    starter_images: Vec<ChatImage>,
    /// Retrieved passages for Ask and Summarize. Translate keeps using `preview` alone.
    evidence: Option<EvidencePack>,
    /// Pages a tool or a named range has already read. Kept for later messages in this chat.
    notebook: PageNotebook,
    /// Page or region rasters that the preview said would be sent.
    page_images: Vec<ChatImage>,
    image_note: String,
    attachment: Option<Attachment>,
    follow_ups: Vec<String>,
    /// Answer text arriving from the provider. Empty once the turn is committed.
    streaming: String,
    input_focused: bool,
    markdown: widgets::markdown::Cache,
}

impl Assistant {
    pub(crate) fn state(&self) -> serde_json::Value {
        // The control channel never receives the key, prompt, extracted text, or answers.
        serde_json::json!({"enabled": self.config.enabled, "busy": self.job.is_some(),
            "preview_ready": self.preview.is_some(),
            "answer_ready": self.messages.iter().any(|t| t.role == Role::Assistant),
            "thread": !self.messages.is_empty(),
            "consent": self.consent, "status": self.status})
    }

    pub(crate) fn showing_settings(&self) -> bool {
        self.settings_open || !self.config.enabled
    }

    pub(crate) fn has_thread(&self) -> bool {
        !self.messages.is_empty()
    }

    /// The first Summarize, Ask, or Translate send has no committed turn yet. The chat still has
    /// to be on screen, or the reply only appears after the provider finishes.
    fn showing_chat(&self) -> bool {
        self.has_thread() || self.job.as_ref().is_some_and(|job| job.pending.is_some()) || !self.streaming.is_empty()
    }

    fn invalidate(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.preview = None;
        self.preview_attempt = None;
        self.consent = false;
        self.messages.clear();
        self.starter.clear();
        self.status.clear();
        self.starter_images.clear();
        self.evidence = None;
        self.notebook = PageNotebook::default();
        self.page_images.clear();
        self.image_note.clear();
        self.follow_ups.clear();
        self.streaming.clear();
        // Keep pending_images and the selection attachment: Preview and question edits must not
        // drop a paste or a passage the user still intends to send. reset_thread clears both.
    }

    pub(crate) fn reset_thread(&mut self) {
        self.invalidate();
        self.pending_images.clear();
        self.attachment = None;
        self.input.clear();
    }

    fn attach_color_image(&mut self, image: &egui::ColorImage) -> Result<(), String> {
        let [w, h] = image.size;
        let mut rgba = Vec::new();
        for px in &image.pixels {
            rgba.extend_from_slice(&px.to_srgba_unmultiplied());
        }
        let encoded = ChatImage::from_rgba(w, h, &rgba)?;
        self.push_pending(encoded, color_thumb(image))
    }

    fn attach_encoded(&mut self, bytes: Vec<u8>) -> Result<(), String> {
        let encoded = ChatImage::from_encoded(bytes)?;
        let preview = color_image_from_chat(&encoded)?;
        self.push_pending(encoded, preview)
    }

    fn push_pending(&mut self, encoded: ChatImage, preview: egui::ColorImage) -> Result<(), String> {
        if self.pending_images.len() >= ai::MAX_CHAT_IMAGES_PER_TURN {
            return Err(format!("Attach at most {} images in one message.", ai::MAX_CHAT_IMAGES_PER_TURN));
        }
        let already =
            self.starter_images.len().saturating_add(self.messages.iter().map(|t| t.images.len()).sum()).saturating_add(self.pending_images.len());
        if already >= ai::MAX_CHAT_IMAGES {
            return Err(format!("This conversation already has {} images. Start a new conversation.", ai::MAX_CHAT_IMAGES));
        }
        self.pending_images.push(PendingImage { encoded, preview: Arc::new(preview) });
        Ok(())
    }

    fn api_turns(&self) -> Vec<Turn> {
        match self.messages.first() {
            Some(turn) if turn.role == Role::User => self.messages.get(1..).unwrap_or(&[]).to_vec(),
            _ => self.messages.clone(),
        }
    }

    fn launch(
        &mut self,
        ctx: Option<egui::Context>,
        pending: Option<(Turn, String)>,
        activity: Arc<Mutex<String>>,
        partial: Arc<Mutex<String>>,
        work: impl FnOnce() -> Result<ResultData, String> + Send + 'static,
    ) {
        if self.job.is_some() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver) = std::sync::mpsc::channel();
            let job = Job { revision: self.revision, pending, receiver, activity, partial };
            match std::thread::Builder::new().name("pdfcraft-ai".into()).spawn(move || {
                let result = pdfcraft_engine::guard(work).unwrap_or_else(|_| Err("AI operation stopped unexpectedly.".into()));
                let _ = sender.send(result);
                if let Some(ctx) = ctx {
                    ctx.request_repaint();
                }
            }) {
                Ok(_) => {
                    self.job = Some(job);
                    self.status = "Working…".into();
                }
                Err(_) => self.status = "Could not start the AI worker.".into(),
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (ctx, pending, activity, partial, work);
            self.status = "AI requests require the desktop app in this version.".into();
        }
    }
}

impl PdfCraftApp {
    pub(crate) fn open_ai(&mut self, command: &str) {
        let task = match command {
            "ai.ask" => Task::Ask,
            "ai.translate" => Task::Translate,
            _ => Task::Summary,
        };
        if command != "ai.settings" && task != self.ai.task {
            self.ai.invalidate();
            self.ai.task = task;
            self.ai.input.clear();
            self.ai.current_page = task == Task::Translate;
        }
        if self.dialog == Some(Dialog::Preferences) {
            self.dialog = None;
        }
        // Settings is a separate view: Summarize / Ask / Translate must not leave the gear open.
        self.ai.settings_open = command == "ai.settings";
        self.choose_right_panel(Some(RightPanel::Ai));
    }

    pub(crate) fn poll_ai(&mut self) {
        let doc = self.active_ids().and_then(|(_, id)| self.session.get(id).map(|d| (id, d.edit_generation())));
        if self.ai.doc != doc {
            self.ai.attachment = None;
            self.ai.invalidate();
            self.ai.doc = doc;
        }
        if self.ai.job.is_none() {
            self.maybe_refresh_thread_preview();
            return;
        }
        let (result, pending, revision) = {
            let Some(job) = &self.ai.job else { return };
            if let Ok(text) = job.partial.lock() {
                self.ai.streaming = text.clone();
            }
            if self.ai.streaming.is_empty() {
                if let Ok(text) = job.activity.lock()
                    && !text.is_empty()
                {
                    self.ai.status = text.clone();
                }
            } else {
                // The answer is on screen. Leave the search line behind it.
                self.ai.status.clear();
            }
            let result = match job.receiver.try_recv() {
                Ok(result) => result,
                Err(std::sync::mpsc::TryRecvError::Empty) => return,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("AI worker stopped unexpectedly.".into()),
            };
            (result, job.pending.clone(), job.revision)
        };
        let streamed = std::mem::take(&mut self.ai.streaming);
        if revision == self.ai.revision {
            self.ai.status.clear();
            match result {
                Ok(ResultData::Preview(preview)) => self.ai.preview = Some(preview),
                Ok(ResultData::Evidence { pack, images, note }) => {
                    let mut text = pack.text.clone();
                    if let Some(note) = &note {
                        text.push_str("\n\n");
                        text.push_str(note);
                    }
                    if !images.is_empty() {
                        text.push_str(&format!("\n\n{} page image(s) will be sent with this request.", images.len()));
                    }
                    let pages = pack.pages.clone();
                    self.ai.evidence = Some(pack);
                    self.ai.page_images = images;
                    self.ai.image_note = note.unwrap_or_default();
                    self.ai.preview = Some(Prepared { text, pages });
                }
                Ok(ResultData::Answer(answer)) => {
                    if let Some((user, starter)) = pending {
                        if self.ai.messages.is_empty() {
                            self.ai.starter = starter;
                        }
                        self.ai.messages.push(user);
                        self.ai.messages.push(Turn { role: Role::Assistant, text: answer, images: Vec::new(), citations: Vec::new() });
                        self.ai.input.clear();
                    }
                }
                Ok(ResultData::Agent(answer)) => {
                    if let Some((user, starter)) = pending {
                        if self.ai.messages.is_empty() {
                            self.ai.starter = starter;
                        }
                        let citations = answer.citations.clone();
                        self.ai.follow_ups = answer.follow_ups.clone();
                        self.ai.notebook = answer.notebook.clone();
                        self.ai.messages.push(user);
                        self.ai.messages.push(Turn { role: Role::Assistant, text: answer.answer.clone(), images: Vec::new(), citations });
                        self.ai.input.clear();
                        if !answer.notes.is_empty() {
                            self.ai.status = answer.notes.join(" ");
                        }
                        self.apply_viewer_actions(&answer);
                    }
                }
                Ok(ResultData::Connection(status)) => self.ai.status = status,
                Err(status) => {
                    // The reply was on screen. Keep that text when a later piece cannot be read.
                    if !streamed.trim().is_empty()
                        && let Some((user, starter)) = pending
                    {
                        if self.ai.messages.is_empty() {
                            self.ai.starter = starter;
                        }
                        self.ai.messages.push(user);
                        self.ai.messages.push(Turn { role: Role::Assistant, text: streamed, images: Vec::new(), citations: Vec::new() });
                        self.ai.input.clear();
                    } else {
                        self.ai.status = status;
                    }
                }
            }
        }
        self.ai.job = None;
        self.maybe_refresh_thread_preview();
    }

    pub(crate) fn start_ai(&mut self, send: bool) {
        self.poll_ai();
        // A page-change preview must not hold the only worker slot. Sending continues in this chat.
        let preview_only = self.ai.job.as_ref().is_some_and(|job| job.pending.is_none());
        if send && preview_only && self.ai.has_thread() {
            self.ai.job = None;
            self.ai.preview_attempt = None;
        }
        if self.ai.job.is_some() {
            return;
        }
        if let Err(e) = self.prepare_ai_job(send) {
            self.ai.status = e;
        }
    }

    fn prepare_ai_job(&mut self, send: bool) -> Result<(), String> {
        if !send {
            return self.prepare_preview();
        }
        if self.ai.has_thread() {
            return self.prepare_follow_up();
        }
        self.prepare_first_send()
    }

    fn prepare_preview(&mut self) -> Result<(), String> {
        if !self.ai.config.enabled {
            return Err("AI assistant is off. Enable it and configure a provider first.".into());
        }
        self.ai.task.instruction(&self.ai.input)?;
        if self.ai.task == Task::Translate {
            let source = self.ai_source()?;
            self.ai.invalidate();
            self.ai.launch(self.ctx.clone(), None, Arc::new(Mutex::new(String::new())), Arc::new(Mutex::new(String::new())), move || {
                source.prepare().map(ResultData::Preview)
            });
            return Ok(());
        }
        let (source, outline, request) = self.evidence_inputs()?;
        let choice = self.page_image_choice(&request);
        self.ai.invalidate();
        let activity = Arc::new(Mutex::new("Searching the document".into()));
        self.ai.launch(self.ctx.clone(), None, activity, Arc::new(Mutex::new(String::new())), move || {
            let (pack, images, note) = ai::prepare_pack(&source, &outline, &request, &choice)?;
            Ok(ResultData::Evidence { pack, images, note })
        });
        Ok(())
    }

    fn prepare_first_send(&mut self) -> Result<(), String> {
        self.ai.config.validate()?;
        self.ai.task.instruction(&self.ai.input)?;
        let view = self.active_ids().ok_or("Open a document first.")?.0;
        if !self.ai.consent {
            return Err("Review the text and allow this request before sending.".into());
        }
        if !self.scoped_preview_current(view)? {
            return self.prepare_preview();
        }
        let starter = self.ai.input.clone();
        let label = starter_label(self.ai.task, self.ai.current_page, &self.ai.input);
        if self.ai.task == Task::Translate { self.launch_completion(&[], starter, label) } else { self.launch_agent(&[], starter, label, false) }
    }

    fn prepare_follow_up(&mut self) -> Result<(), String> {
        self.ai.config.validate()?;
        let follow = self.ai.input.trim();
        if follow.is_empty() && self.ai.pending_images.is_empty() {
            return Err("Enter a follow-up question or attach an image.".into());
        }
        if follow.len() > ai::MAX_PROMPT_BYTES {
            return Err(format!("Question or language must fit in {} UTF-8 bytes.", ai::MAX_PROMPT_BYTES));
        }
        let images: Vec<ChatImage> = self.ai.pending_images.iter().map(|p| p.encoded.clone()).collect();
        let mut turns = self.ai.api_turns();
        turns.push(Turn { role: Role::User, text: follow.to_owned(), images, citations: Vec::new() });
        let label = follow.to_owned();
        let starter = self.ai.starter.clone();
        if self.ai.task == Task::Translate { self.launch_completion(&turns, starter, label) } else { self.launch_agent(&turns, starter, label, true) }
    }

    fn ai_source(&self) -> Result<ai::Source, String> {
        let (view, id) = self.active_ids().ok_or("Open a document first.")?;
        let doc = self.session.get(id).ok_or("Document is no longer open.")?;
        let pages = if self.ai.current_page { vec![self.views[view].current] } else { vec![] };
        doc.ai_source(&pages)
    }

    fn viewed_page(&self) -> Option<usize> {
        let (view, _) = self.active_ids()?;
        self.views.get(view).map(|view| view.current)
    }

    /// `Err` when nothing has been previewed. `Ok(false)` when "This page" no longer matches the open page.
    fn scoped_preview_current(&self, view: usize) -> Result<bool, String> {
        let preview = self.ai.preview.as_ref().ok_or("Preview the document text before sending.")?;
        let page = self.views.get(view).map(|view| view.current).ok_or("Open a document first.")?;
        Ok(!self.ai.current_page || preview.pages == vec![page + 1])
    }

    fn thread_page_is_stale(&self) -> bool {
        if !self.ai.has_thread() || !self.ai.current_page {
            return false;
        }
        let Some(page) = self.viewed_page() else { return false };
        !self.ai.preview.as_ref().is_some_and(|preview| preview.pages == vec![page + 1])
    }

    /// Re-read the open page without clearing the conversation, then show that text in the chat.
    fn maybe_refresh_thread_preview(&mut self) {
        if self.ai.job.is_some() || !self.ai.config.enabled || !self.thread_page_is_stale() {
            return;
        }
        let Some(page) = self.viewed_page() else { return };
        if self.ai.preview_attempt == Some(page) {
            return;
        }
        self.ai.preview_attempt = Some(page);
        if let Err(error) = self.refresh_thread_preview() {
            self.ai.status = error;
        }
    }

    fn refresh_thread_preview(&mut self) -> Result<(), String> {
        if self.ai.task == Task::Translate {
            let source = self.ai_source()?;
            self.ai.launch(self.ctx.clone(), None, Arc::new(Mutex::new(String::new())), Arc::new(Mutex::new(String::new())), move || {
                source.prepare().map(ResultData::Preview)
            });
            return Ok(());
        }
        let (source, outline, request) = self.evidence_inputs()?;
        let choice = self.page_image_choice(&request);
        let activity = Arc::new(Mutex::new("Searching the document".into()));
        self.ai.launch(self.ctx.clone(), None, activity, Arc::new(Mutex::new(String::new())), move || {
            let (pack, images, note) = ai::prepare_pack(&source, &outline, &request, &choice)?;
            Ok(ResultData::Evidence { pack, images, note })
        });
        Ok(())
    }

    fn launch_completion(&mut self, turns: &[Turn], starter: String, user_text: String) -> Result<(), String> {
        let preview = self.ai.preview.as_ref().ok_or("Preview the document text before sending.")?;
        let snapshot_images: Vec<ChatImage> =
            if turns.is_empty() { self.ai.pending_images.iter().map(|p| p.encoded.clone()).collect() } else { self.ai.starter_images.clone() };
        let messages = preview.conversation_messages(self.ai.task, &starter, turns, &snapshot_images)?;
        let user_images = if turns.is_empty() { snapshot_images.clone() } else { turns.last().map(|t| t.images.clone()).unwrap_or_default() };
        let user = Turn { role: Role::User, text: user_text, images: user_images, citations: Vec::new() };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut provider = ai::OpenAiProvider::new(self.ai.config.clone(), self.ai.key.to_string())?;
            self.ai.pending_images.clear();
            if turns.is_empty() {
                self.ai.starter_images = snapshot_images;
            }
            let partial = Arc::new(Mutex::new(String::new()));
            let partial_job = Arc::clone(&partial);
            let ctx = self.ctx.clone();
            self.ai.launch(self.ctx.clone(), Some((user, starter)), Arc::new(Mutex::new(String::new())), partial_job, move || {
                ai::complete_messages_reporting(&mut provider, &messages, ai::StreamKind::Plain, move |text| report_ctx(&partial, &ctx, text))
                    .map(ResultData::Answer)
            });
            Ok(())
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (messages, user);
            Err("AI requests require the desktop app in this version.".into())
        }
    }

    /// Ask and Summarize. `refresh` gathers again for a follow-up; the first send uses the previewed pack.
    fn launch_agent(&mut self, turns: &[Turn], starter: String, user_text: String, refresh: bool) -> Result<(), String> {
        if self.ai.preview.is_none() {
            return Err("Preview the document text before sending.".into());
        }
        if !refresh && self.ai.evidence.is_none() {
            return Err("Preview the document text before sending.".into());
        }
        let (source, outline, mut request) = self.evidence_inputs()?;
        let user_images: Vec<ChatImage> = if turns.is_empty() {
            self.ai.pending_images.iter().map(|image| image.encoded.clone()).collect()
        } else {
            turns.last().map(|turn| turn.images.clone()).unwrap_or_default()
        };
        if refresh && !user_text.trim().is_empty() {
            request.question =
                if request.focus.text.is_empty() { user_text.trim().to_string() } else { format!("{}\n{}", user_text.trim(), request.focus.text) };
        }
        let choice = if refresh { self.page_image_choice(&request) } else { ImageChoice::Off };
        let stored_pack = if refresh { None } else { self.ai.evidence.clone() };
        let stored_images = if refresh { Vec::new() } else { self.ai.page_images.clone() };
        let mut notebook = self.ai.notebook.clone();
        let image_block = if !self.ai.config.send_page_images {
            Some("Page images are off. The text of those pages is included instead.")
        } else if !self.ai.config.model_accepts_images {
            Some("Page images were not sent because this model is not marked as accepting images. The text of those pages is included instead.")
        } else {
            None
        };
        let task = self.ai.task;
        let user = Turn { role: Role::User, text: user_text, images: user_images.clone(), citations: Vec::new() };
        let extra_images = if turns.is_empty() { user_images } else { Vec::new() };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut provider = ai::OpenAiProvider::new(self.ai.config.clone(), self.ai.key.to_string())?;
            self.ai.pending_images.clear();
            let turns = turns.to_vec();
            let activity = Arc::new(Mutex::new("Searching the document".into()));
            let report_to = Arc::clone(&activity);
            let partial = Arc::new(Mutex::new(String::new()));
            let partial_job = Arc::clone(&partial);
            let ctx = self.ctx.clone();
            self.ai.launch(self.ctx.clone(), Some((user, starter.clone())), activity, partial_job, move || {
                let (pack, mut images) = if let Some(pack) = stored_pack {
                    (pack, stored_images)
                } else {
                    report_ctx(&report_to, &ctx, "Searching the document");
                    let (pack, images, note) = ai::prepare_pack(&source, &outline, &request, &choice)?;
                    if let Some(note) = &note {
                        report_ctx(&report_to, &ctx, note);
                    }
                    (pack, images)
                };
                for image in extra_images {
                    if images.len() >= ai::MAX_CHAT_IMAGES_PER_TURN {
                        break;
                    }
                    images.push(image);
                }
                let search_slot = Arc::clone(&report_to);
                let search_ctx = ctx.clone();
                let more_source = source.clone();
                let more_outline = outline.clone();
                let base = request.clone();
                let answer = ai::complete_evidence(
                    &mut provider,
                    &source,
                    &outline,
                    pack,
                    &mut notebook,
                    task,
                    &starter,
                    &turns,
                    &images,
                    image_block,
                    move |query| {
                        let mut again = base.clone();
                        again.question = query.to_string();
                        report_ctx(&search_slot, &search_ctx, &format!("Searching for {query}"));
                        let (pack, _, _) = ai::prepare_pack(&more_source, &more_outline, &again, &ImageChoice::Off)?;
                        Ok(pack)
                    },
                    {
                        let slot = Arc::clone(&report_to);
                        let ctx = ctx.clone();
                        move |text| report_ctx(&slot, &ctx, text)
                    },
                    {
                        let slot = Arc::clone(&partial);
                        let ctx = ctx.clone();
                        move |text| report_ctx(&slot, &ctx, text)
                    },
                )?;
                Ok(ResultData::Agent(answer))
            });
            Ok(())
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (source, outline, request, choice, stored_pack, stored_images, notebook, image_block, task, user, extra_images, turns, starter);
            Err("AI requests require the desktop app in this version.".into())
        }
    }

    fn evidence_inputs(&self) -> Result<(pdfcraft_engine::export::ExportSource, Vec<ai::OutlineEntry>, EvidenceRequest), String> {
        let (view, id) = self.active_ids().ok_or("Open a document first.")?;
        let doc = self.session.get(id).ok_or("Document is no longer open.")?;
        let current = self.views.get(view).map(|view| view.current).ok_or("Open a document first.")?;
        let pages = if self.ai.current_page { vec![current] } else { Vec::new() };
        // Same copy and redaction checks as the full snapshot.
        let _ = doc.ai_source(&pages)?;
        let outline = doc.ai_outline()?;
        let mut source = doc.export_source();
        source.config.system_fonts = false;
        let focus = self.ai.attachment.as_ref().map(attachment_focus).unwrap_or_default();
        let question = match self.ai.task {
            Task::Summary if focus.text.is_empty() => "Summarize".to_string(),
            Task::Summary => format!("Summarize\n{}", focus.text),
            _ => self.ai.input.trim().to_string(),
        };
        Ok((source, outline, EvidenceRequest { task: self.ai.task, question, pages, current_page: Some(current), focus }))
    }

    fn page_image_choice(&self, request: &EvidenceRequest) -> ImageChoice {
        ai::image_choice(self.ai.config.send_page_images, self.ai.config.model_accepts_images, &request.question, request.focus.region.is_some())
    }

    pub(crate) fn attach_selection_ask(&mut self, ask: crate::canvas::SelectionAsk) {
        let task = if ask.task_name == "summarize" { Task::Summary } else { Task::Ask };
        if !(self.ai.has_thread() && self.ai.task == task) {
            self.ai.invalidate();
            self.ai.task = task;
            self.ai.input.clear();
            self.ai.current_page = true;
        }
        match ask.task_name {
            "explain" => self.ai.input = "Explain this selection.".into(),
            "ask" if ask.region && self.ai.input.trim().is_empty() => self.ai.input = "What is in this region?".into(),
            _ => {}
        }
        self.ai.attachment = Some(Attachment { text: ask.text, page: ask.page, ranges: ask.ranges, region: ask.region, rect: ask.rect });
        self.ai.settings_open = false;
        if self.dialog == Some(Dialog::Preferences) {
            self.dialog = None;
        }
        self.choose_right_panel(Some(RightPanel::Ai));
    }

    pub(crate) fn show_citation(&mut self, citation: &ai::Citation) {
        let Some((index, id)) = self.active_ids() else { return };
        let info = match self.session.get(id) {
            Some(doc) => doc.info.clone(),
            None => return,
        };
        let user = info.pages.get(citation.page).zip(citation.rects.first().copied()).map(|(page, rect)| {
            let a = page.view_to_user(rect[0], rect[1]);
            let b = page.view_to_user(rect[2], rect[3]);
            [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])]
        });
        let view = &mut self.views[index];
        if let Some(rect) = user {
            view.reveal_rect(citation.page, rect, &info);
        } else {
            view.go_to_page(citation.page);
        }
        view.citation_flash = Some((citation.page, citation.rects.clone(), 0.0));
    }

    fn apply_viewer_actions(&mut self, answer: &AgentAnswer) {
        if let Some(span_id) = answer.actions.iter().find_map(|action| match action {
            ViewerAction::Highlight { span_id } => Some(span_id.as_str()),
            ViewerAction::Navigate { .. } => None,
        }) && let Some(cite) = answer.citations.iter().find(|cite| cite.span_id == span_id).cloned()
        {
            self.show_citation(&cite);
            return;
        }
        if let Some(page) = answer.actions.iter().find_map(|action| match action {
            ViewerAction::Navigate { page } => Some(*page),
            ViewerAction::Highlight { .. } => None,
        }) && let Some((index, _)) = self.active_ids()
        {
            self.views[index].go_to_page(page);
        }
    }

    pub(crate) fn test_ai(&mut self) {
        self.poll_ai();
        if self.ai.job.is_some() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        match ai::OpenAiProvider::new(self.ai.config.clone(), self.ai.key.to_string()) {
            Ok(mut provider) => {
                self.ai.launch(self.ctx.clone(), None, Arc::new(Mutex::new(String::new())), Arc::new(Mutex::new(String::new())), move || {
                    ai::test_provider(&mut provider).map(ResultData::Connection)
                })
            }
            Err(e) => self.ai.status = e,
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.ai.status = "AI requests require the desktop app in this version.".into();
        }
    }

    pub(crate) fn set_ai_option(&mut self, key: &str, value: &str) -> Result<(), String> {
        if self.ai.job.is_some() {
            return Err("Wait for the current AI request to finish.".into());
        }
        if value.len() > ai::MAX_PROMPT_BYTES {
            return Err("AI setting is too long.".into());
        }
        if key == "ai-consent" {
            self.ai.consent = value.parse().map_err(|_| "ai-consent must be true or false")?;
            return Ok(());
        }
        if key == "ai-input" {
            if self.ai.messages.is_empty() {
                self.ai.invalidate();
            }
            self.ai.input = value.into();
            return Ok(());
        }
        if key == "ai-stream" {
            self.ai.config.stream_responses = value.parse().map_err(|_| "ai-stream must be true or false")?;
            return Ok(());
        }
        self.ai.invalidate();
        match key {
            "ai-enabled" => self.ai.config.enabled = value.parse().map_err(|_| "ai-enabled must be true or false")?,
            "ai-url" => self.ai.config.base_url = value.into(),
            "ai-model" => self.ai.config.model = value.into(),
            "ai-current-page" => self.ai.current_page = value.parse().map_err(|_| "ai-current-page must be true or false")?,
            "ai-legacy-tokens" => self.ai.config.legacy_max_tokens = value.parse().map_err(|_| "ai-legacy-tokens must be true or false")?,
            _ => return Err("Unknown AI setting. Enter API keys only in the masked desktop field.".into()),
        }
        if !self.ai.config.enabled {
            self.ai.key = Zeroizing::new(String::new());
            self.ai.pending_images.clear();
        }
        Ok(())
    }
}

fn attachment_focus(attachment: &Attachment) -> Focus {
    Focus {
        ranges: attachment.ranges.clone(),
        region_page: attachment.region.then_some(attachment.page),
        region: attachment.rect,
        text: attachment.text.clone(),
    }
}

fn report_ctx(slot: &Mutex<String>, ctx: &Option<egui::Context>, text: &str) {
    *slot.lock().unwrap_or_else(PoisonError::into_inner) = text.to_string();
    if let Some(ctx) = ctx {
        ctx.request_repaint();
    }
}

fn faint(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(text).font(theme::regular(11.0)).color(t.text_faint));
}

fn starter_label(task: Task, current_page: bool, input: &str) -> String {
    match task {
        Task::Summary if current_page => tl!("Summarize this page").into(),
        Task::Summary => tl!("Summarize this document").into(),
        Task::Ask => input.trim().to_owned(),
        Task::Translate if current_page => crate::i18n::fmt(tl!("Translate this page into {lang}"), &[("lang", input.trim())]),
        Task::Translate => crate::i18n::fmt(tl!("Translate this document into {lang}"), &[("lang", input.trim())]),
    }
}

/// Enable, endpoint, model, and the session-only API key. Shared by Preferences and the assistant panel.
pub(crate) fn provider_settings(ui: &mut egui::Ui, app: &mut PdfCraftApp) -> bool {
    let busy = app.ai.job.is_some();
    let mut test = false;
    ui.add_enabled_ui(!busy, |ui| {
        ui.spacing_mut().item_spacing.y = 8.0;
        let a = &mut app.ai;
        let mut changed = ui.add(widgets::checkbox(&mut a.config.enabled, tl!("Enable AI assistant"))).changed();
        let url_label = ui.label(tl!("API base URL"));
        changed |= ui.add(widgets::line(&mut a.config.base_url).char_limit(2048).desired_width(f32::INFINITY)).labelled_by(url_label.id).changed();
        let model_label = ui.label(tl!("Model ID"));
        let model = ui
            .add(widgets::line(&mut a.config.model).hint_text("gpt-4o-mini").char_limit(256).desired_width(f32::INFINITY))
            .labelled_by(model_label.id)
            .on_hover_text(tl!("For OpenAI this is gpt-4o-mini or gpt-4o, not ChatGPT."));
        changed |= model.changed();
        let label = ui.label(tl!("API key (optional for local servers)"));
        changed |= ui.add(widgets::line(&mut *a.key).password(true).char_limit(8192).desired_width(f32::INFINITY)).labelled_by(label.id).changed();
        faint(ui, tl!("Paste the key here. It stays in memory for this session and is never saved in preferences."));
        if ui.button(tl!("Clear API key")).clicked() {
            a.key = Zeroizing::new(String::new());
            changed = true;
        }
        let _ = ui.add(widgets::checkbox(&mut a.config.stream_responses, tl!("Stream the answer as it is written")));
        faint(ui, tl!("Turn this off to wait for the complete answer before showing it."));
        changed |= ui.add(widgets::checkbox(&mut a.config.model_accepts_images, tl!("This model accepts images"))).changed();
        if !a.config.model_accepts_images && a.config.send_page_images {
            a.config.send_page_images = false;
            changed = true;
        }
        ui.add_enabled_ui(a.config.model_accepts_images, |ui| {
            changed |= ui.add(widgets::checkbox(&mut a.config.send_page_images, tl!("Send PDF page images"))).changed();
        });
        if a.config.model_accepts_images {
            faint(ui, tl!("Page images are sent only for a visual question or a selected region, and the provider may charge for them."));
        } else {
            faint(ui, tl!("Page images stay off because this model is not marked as accepting images."));
        }
        let enabled = a.config.enabled;
        ui.add_enabled_ui(enabled, |ui| {
            test = widgets::pill_button(ui, tl!("Test connection"), false).on_hover_text(tl!("Test connection (no document text)")).clicked();
        });
        if changed {
            a.invalidate();
            if !a.config.enabled {
                a.key = Zeroizing::new(String::new());
                a.pending_images.clear();
            }
        }
    });
    if !app.ai.status.is_empty() {
        faint(ui, &app.ai.status);
    }
    test
}

pub(crate) fn body(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    let size = ui.available_size();
    if size.x.is_finite() && size.y.is_finite() {
        ui.set_min_size(size);
    }
    ui.set_clip_rect(ui.max_rect());
    if !app.ai.config.enabled {
        app.ai.settings_open = true;
    }
    app.ai.input_focused = false;
    if app.ai.showing_settings() {
        settings_view(ui, app);
        return;
    }
    if app.ai.showing_chat() {
        thread_view(ui, app);
    } else {
        starter_view(ui, app);
    }
    absorb_chat_images(ui, app);
}

fn settings_view(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    let mut test = false;
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        test = provider_settings(ui, app);
    });
    if test {
        app.test_ai();
    }
}

/// Claim the inspector body so the page cannot paint through empty space, then split a dock.
fn pane_regions(ui: &mut egui::Ui, dock_h: f32) -> (egui::Rect, egui::Rect) {
    let full = ui.available_rect_before_wrap();
    let _ = ui.allocate_rect(full, egui::Sense::hover());
    ui.set_clip_rect(full);
    let dock_h = if full.height().is_finite() { dock_h.clamp(40.0, (full.height() * 0.5).max(40.0)) } else { dock_h };
    let dock = egui::Rect::from_min_max(egui::pos2(full.left(), full.bottom() - dock_h), full.max);
    let main = egui::Rect::from_min_max(full.min, egui::pos2(full.right(), dock.top()));
    (main, dock)
}

fn pane_scroll(ui: &mut egui::Ui, rect: egui::Rect, salt: &str, stick: bool, add: impl FnOnce(&mut egui::Ui)) {
    ui.scope_builder(egui::UiBuilder::new().max_rect(rect).layout(Layout::top_down(Align::Min)), |ui| {
        egui::ScrollArea::vertical().id_salt(salt).auto_shrink([false, false]).stick_to_bottom(stick).show(ui, add);
    });
}

fn starter_view(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    let mut preview = false;
    let mut send = false;
    let mut attach = false;
    let (main, dock) = pane_regions(ui, 48.0);
    pane_scroll(ui, main, "ai-start", false, |ui| {
        starter_form(ui, app);
        pending_row(ui, app);
        preview_block(ui, app);
        if !app.ai.status.is_empty() {
            faint(ui, &app.ai.status);
        }
        #[cfg(target_arch = "wasm32")]
        faint(ui, tl!("AI requests require the desktop app in this version."));
    });
    ui.scope_builder(egui::UiBuilder::new().max_rect(dock).layout(Layout::left_to_right(Align::Center)), |ui| {
        attach = attach_button(ui);
        ui.add_space(6.0);
        action_button(ui, app, &mut preview, &mut send);
    });
    if attach {
        pick_chat_image(app);
    }
    if preview {
        app.start_ai(false);
    }
    if send {
        app.start_ai(true);
    }
}

fn thread_view(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    let mut send = false;
    thread_chip(ui, app);
    ui.add_space(6.0);
    let composer_h = if app.ai.pending_images.is_empty() { 128.0 } else { 188.0 };
    // The selection sits above the follow-up box. Its height is whatever the
    // wrapped quote used last frame, so the dock grows with the text instead of
    // pinning the card to the top of the thread.
    let attach_id = ui.scope_id().with("ai-attachment-height");
    let attach_h = if app.ai.attachment.is_some() { ui.data(|d| d.get_temp::<f32>(attach_id)).unwrap_or(112.0) } else { 0.0 };
    let (main, dock) = pane_regions(ui, composer_h + attach_h);
    let follow_stream = !app.ai.streaming.is_empty();
    pane_scroll(ui, main, "ai-thread", follow_stream, |ui| {
        // The preview sits above the turns so sticking to the bottom follows the
        // reply that is still being written, including the first Summarize.
        thread_preview(ui, app);
        transcript(ui, app);
        if app.ai.streaming.is_empty() && !app.ai.status.is_empty() {
            ui.add_space(6.0);
            faint(ui, &app.ai.status);
        }
    });
    let composer_h = composer_h.min(dock.height());
    let composer_rect = egui::Rect::from_min_max(egui::pos2(dock.left(), dock.bottom() - composer_h), dock.max);
    let attach_rect = egui::Rect::from_min_max(dock.min, egui::pos2(dock.right(), composer_rect.top()));
    if app.ai.attachment.is_some() && attach_rect.height() > 0.0 {
        ui.scope_builder(egui::UiBuilder::new().max_rect(attach_rect).layout(Layout::top_down(Align::Min)), |ui| {
            ui.set_clip_rect(attach_rect);
            let before = ui.cursor().min.y;
            attachment_card(ui, app);
            let used = ui.cursor().min.y - before;
            if used.is_finite() && used > 0.0 {
                ui.data_mut(|d| d.insert_temp(attach_id, used));
            }
        });
    }
    ui.scope_builder(egui::UiBuilder::new().max_rect(composer_rect).layout(Layout::top_down(Align::Min)), |ui| {
        composer(ui, app, &mut send);
    });
    if send {
        app.start_ai(true);
    }
}

fn starter_form(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    let busy = app.ai.job.is_some();
    ui.add_enabled_ui(!busy, |ui| {
        ui.spacing_mut().item_spacing.y = 10.0;
        let t = Tokens::get(ui.ctx());
        let mut changed = false;
        changed |= crate::dialogs::preferences::segmented_pill_choice(
            ui,
            &t,
            &mut app.ai.task,
            &[(Task::Summary, tl!("Summarize")), (Task::Ask, tl!("Ask")), (Task::Translate, tl!("Translate"))],
        );
        if app.ai.task == Task::Ask {
            let label = ui.label(tl!("Question"));
            let response = ui
                .add(widgets::block(&mut app.ai.input).id_salt("ai-input").desired_rows(3).char_limit(2000).desired_width(f32::INFINITY))
                .labelled_by(label.id);
            app.ai.input_focused |= response.has_focus();
            changed |= response.changed();
        } else if app.ai.task == Task::Translate {
            let label = ui.label(tl!("Translate into"));
            let response = ui
                .add(widgets::block(&mut app.ai.input).id_salt("ai-input").desired_rows(3).char_limit(2000).desired_width(f32::INFINITY))
                .labelled_by(label.id);
            app.ai.input_focused |= response.has_focus();
            changed |= response.changed();
        }
        changed |= crate::dialogs::preferences::segmented_pill_choice(
            ui,
            &t,
            &mut app.ai.current_page,
            &[(true, tl!("This page")), (false, tl!("Document"))],
        );
        if changed {
            app.ai.invalidate();
        }
    });
    attachment_card(ui, app);
}

fn preview_block(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    let busy = app.ai.job.is_some();
    ui.add_enabled_ui(!busy, |ui| {
        let t = Tokens::get(ui.ctx());
        if let Some(p) = &app.ai.preview {
            let page_s = crate::i18n::trn(crate::i18n::current(), p.pages.len() as u64, "{n} page", "{n} pages");
            faint(ui, &format!("{} · {}", page_s, crate::panels::human_size(p.text.len())));
            ui.collapsing(tl!("Text to send"), |ui| {
                egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                    ui.add(egui::Label::new(&p.text).wrap());
                });
            });
            let dest = format!("{} · {}", app.ai.config.base_url, app.ai.config.model);
            ui.add(egui::Label::new(RichText::new(dest).font(theme::regular(11.0)).color(t.text_faint)).truncate());
            if !app.ai.image_note.is_empty() {
                faint(ui, &app.ai.image_note);
            }
            let page_images = !app.ai.page_images.is_empty();
            let attached = !app.ai.pending_images.is_empty();
            let consent = match (page_images, attached) {
                (true, true) => tl!("Allow sending this text, page images, and attached images for this request"),
                (true, false) => tl!("Allow sending this text and page images for this request"),
                (false, true) => tl!("Allow sending this text and attached images for this request"),
                (false, false) => tl!("Allow sending this text for this request"),
            };
            ui.add(widgets::checkbox(&mut app.ai.consent, consent));
            faint(
                ui,
                if page_images || attached {
                    tl!("Sending shares this text, your question, and the listed images with the chosen provider, which may charge for usage. A local server may forward requests elsewhere.")
                } else {
                    tl!("Sending shares this text and your question with the chosen provider, which may charge for usage. A local server may forward requests elsewhere.")
                },
            );
        }
    });
}

fn action_button(ui: &mut egui::Ui, app: &mut PdfCraftApp, preview: &mut bool, send: &mut bool) {
    let busy = app.ai.job.is_some();
    if busy {
        ui.horizontal(|ui| {
            ui.spinner();
            let status = if app.ai.status.is_empty() { tl!("Working…").to_string() } else { app.ai.status.clone() };
            faint(ui, &status);
        });
        return;
    }
    if app.ai.preview.is_none() {
        ui.add_enabled_ui(app.ai.config.enabled, |ui| {
            *preview = widgets::pill_button_fill(ui, tl!("Preview"), true).clicked();
        });
        return;
    }
    ui.add_enabled_ui(app.ai.consent, |ui| {
        *send = widgets::pill_button_fill(ui, tl!("Send"), true).clicked();
    });
}

fn thread_preview(ui: &mut egui::Ui, app: &PdfCraftApp) {
    let Some(preview) = &app.ai.preview else { return };
    if app.thread_page_is_stale() {
        return;
    }
    ui.add_space(8.0);
    let page_s = crate::i18n::trn(crate::i18n::current(), preview.pages.len() as u64, "{n} page", "{n} pages");
    faint(ui, &format!("{} · {}", page_s, crate::panels::human_size(preview.text.len())));
    egui::CollapsingHeader::new(tl!("Text to send")).default_open(true).show(ui, |ui| {
        egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
            ui.add(egui::Label::new(&preview.text).wrap());
        });
    });
    if !app.ai.image_note.is_empty() {
        faint(ui, &app.ai.image_note);
    }
}

fn thread_chip(ui: &mut egui::Ui, app: &PdfCraftApp) {
    let t = Tokens::get(ui.ctx());
    let task = match app.ai.task {
        Task::Summary => tl!("Summarize"),
        Task::Ask => tl!("Ask"),
        Task::Translate => tl!("Translate"),
    };
    let scope = if app.ai.current_page { tl!("This page") } else { tl!("Document") };
    let size = app.ai.preview.as_ref().map(|p| crate::panels::human_size(p.text.len())).unwrap_or_default();
    let text = if size.is_empty() { format!("{task} · {scope}") } else { format!("{task} · {scope} · {size}") };
    ui.add(egui::Label::new(RichText::new(text).font(theme::medium(12.0)).color(t.text_muted)).truncate());
}

fn transcript(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    ui.spacing_mut().item_spacing.y = 10.0;
    let mut first_assistant = true;
    let mut clicked: Option<(String, Vec<ai::Citation>)> = None;
    let mut i = 0;
    while i < app.ai.messages.len() {
        let role = app.ai.messages.get(i).map(|turn| turn.role);
        match role {
            Some(Role::User) => {
                if let Some(turn) = app.ai.messages.get(i) {
                    user_bubble(ui, turn);
                }
            }
            Some(Role::Assistant) => {
                let text = app.ai.messages.get(i).map(|turn| turn.text.clone()).unwrap_or_default();
                let citations = app.ai.messages.get(i).map(|turn| turn.citations.clone()).unwrap_or_default();
                if let Some(id) = assistant_bubble(ui, &mut app.ai.markdown, &text, &citations, first_assistant) {
                    clicked = Some((id, citations));
                }
                first_assistant = false;
            }
            None => break,
        }
        i = i.saturating_add(1);
    }
    let pending_user = app.ai.job.as_ref().and_then(|job| job.pending.as_ref().map(|(turn, _)| turn.clone()));
    let streaming = app.ai.streaming.clone();
    if let Some(turn) = pending_user.as_ref() {
        user_bubble(ui, turn);
    }
    if !streaming.is_empty() {
        // Citations are checked when the reply finishes, so chips in the partial text are not live yet.
        let _ = assistant_bubble(ui, &mut app.ai.markdown, &streaming, &[], first_assistant);
    }
    if let Some((id, citations)) = clicked
        && let Some(cite) = citations.into_iter().find(|cite| cite.id == id)
    {
        app.show_citation(&cite);
    }
}

fn user_bubble(ui: &mut egui::Ui, turn: &Turn) {
    let t = Tokens::get(ui.ctx());
    egui::Frame::NONE.fill(t.accent_soft).corner_radius(CornerRadius::same(8)).inner_margin(egui::Margin::same(10)).show(ui, |ui| {
        ui.set_width(ui.available_width());
        show_turn_images(ui, &turn.images);
        if !turn.text.is_empty() {
            ui.label(RichText::new(&turn.text).font(theme::regular(12.0)).color(t.text));
        }
    });
}

fn assistant_bubble(
    ui: &mut egui::Ui,
    cache: &mut widgets::markdown::Cache,
    text: &str,
    citations: &[ai::Citation],
    disclaimer: bool,
) -> Option<String> {
    let t = Tokens::get(ui.ctx());
    let mut copy = false;
    let mut clicked = None;
    let cites: Vec<widgets::markdown::CiteRef<'_>> = citations
        .iter()
        .map(|cite| widgets::markdown::CiteRef {
            id: &cite.id,
            page: cite.page.saturating_add(1),
            quote: cite.quote.as_str(),
            page_level: cite.page_level,
        })
        .collect();
    egui::Frame::NONE
        .fill(t.field)
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(RichText::new(tl!("Answer")).font(theme::semibold(12.5)));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if icons::button(ui, "copy", 22.0, false, tl!("Copy answer")).clicked() {
                        copy = true;
                    }
                });
            });
            if disclaimer {
                if citations.is_empty() {
                    faint(ui, tl!("AI-generated answer — verify against the document. Page citations are not independently verified."));
                } else {
                    faint(ui, tl!("AI-generated answer — verify against the document. A citation chip opens the checked passage."));
                }
                ui.add_space(6.0);
            }
            clicked = if citations.is_empty() {
                widgets::markdown::show(ui, cache, text);
                None
            } else {
                widgets::markdown::show_cited(ui, cache, text, &cites)
            };
        });
    if copy {
        ui.ctx().copy_text(text.to_owned());
    }
    clicked
}

fn attachment_card(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    let Some((quote, page, region)) = app.ai.attachment.as_ref().map(|attachment| {
        let mut quote: String = attachment.text.chars().take(160).collect();
        if attachment.text.chars().count() > 160 {
            quote.push('…');
        }
        (quote, attachment.page.saturating_add(1), attachment.region)
    }) else {
        return;
    };
    ui.add_space(8.0);
    // The quote is one long line. A child with a fixed width wraps it inside the
    // inspector instead of widening the panel past its edge.
    let width = ui.available_width();
    let mut clear = false;
    ui.scope(|ui| {
        if width.is_finite() {
            ui.set_width(width);
        }
        let page_s = page.to_string();
        let label = if region {
            crate::i18n::fmt(tl!("Selected region on page {page}"), &[("page", &page_s)])
        } else {
            crate::i18n::fmt(tl!("Selected text on page {page}"), &[("page", &page_s)])
        };
        let t = Tokens::get(ui.ctx());
        ui.add(egui::Label::new(RichText::new(label).font(theme::regular(11.0)).color(t.text_faint)).wrap());
        if !quote.is_empty() {
            ui.add(egui::Label::new(RichText::new(quote).font(theme::regular(12.0))).wrap());
        }
        clear = ui.button(tl!("Clear selection")).clicked();
    });
    if clear {
        app.ai.attachment = None;
        if !app.ai.has_thread() {
            app.ai.invalidate();
        }
    }
}

fn composer(ui: &mut egui::Ui, app: &mut PdfCraftApp, send: &mut bool) {
    let busy = app.ai.job.is_some();
    let mut attach = false;
    ui.add_space(8.0);
    follow_up_buttons(ui, app);
    faint(ui, tl!("Each message is sent with the retrieved passages and this conversation."));
    pending_row(ui, app);
    ui.add_enabled_ui(!busy, |ui| {
        let response = ui.add(
            widgets::block(&mut app.ai.input)
                .id_salt("ai-composer")
                .desired_rows(2)
                .char_limit(2000)
                .hint_text(tl!("Ask a follow-up or paste an image"))
                .desired_width(f32::INFINITY),
        );
        app.ai.input_focused |= response.has_focus();
    });
    ui.add_space(6.0);
    if busy {
        ui.horizontal(|ui| {
            ui.spinner();
            let status = if app.ai.status.is_empty() { tl!("Working…").to_string() } else { app.ai.status.clone() };
            faint(ui, &status);
        });
        return;
    }
    ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
        attach = attach_button(ui);
        ui.add_space(6.0);
        ui.add_enabled_ui(!app.ai.input.trim().is_empty() || !app.ai.pending_images.is_empty(), |ui| {
            *send = widgets::pill_button_fill(ui, tl!("Send"), true).clicked();
        });
    });
    if attach {
        pick_chat_image(app);
    }
}

fn follow_up_buttons(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    if app.ai.follow_ups.is_empty() || app.ai.job.is_some() {
        return;
    }
    let suggestions = app.ai.follow_ups.clone();
    ui.horizontal_wrapped(|ui| {
        for suggestion in suggestions {
            if ui.small_button(&suggestion).clicked() {
                app.ai.input = suggestion;
            }
        }
    });
    ui.add_space(6.0);
}

fn attach_button(ui: &mut egui::Ui) -> bool {
    icons::button(ui, "image-plus", 26.0, false, tl!("Add image")).on_hover_text(tl!("Attach an image, or paste one into the chat box.")).clicked()
}

fn pick_chat_image(app: &mut PdfCraftApp) {
    #[cfg(not(target_arch = "wasm32"))]
    app.pick_image(tl!("Choose an image to send"), |app, _name, bytes| {
        if let Err(e) = app.ai.attach_encoded(bytes) {
            app.ai.status = e;
        }
    });
    #[cfg(target_arch = "wasm32")]
    {
        app.ai.status = tl!("Pasting an image works on this version; choosing a file needs the desktop app.").into();
    }
}

fn absorb_chat_images(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    if app.ai.job.is_some() {
        return;
    }
    let over = ui.rect_contains_pointer(ui.max_rect());
    if !app.ai.input_focused && !over {
        return;
    }
    let mut pasted = Vec::new();
    ui.ctx().input_mut(|i| {
        i.events.retain(|event| match event {
            egui::Event::PasteImage(image) => {
                pasted.push(image.clone());
                false
            }
            _ => true,
        });
    });
    if pasted.is_empty() && app.ai.input_focused {
        #[cfg(not(target_arch = "wasm32"))]
        if ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::V))
            && let Some(crate::create_ui::Clip::Image { width, height, rgba }) = crate::create_ui::read_clipboard()
        {
            match ChatImage::from_rgba(width, height, &rgba) {
                Ok(encoded) => {
                    let preview = color_image_from_chat(&encoded).unwrap_or_else(|_| egui::ColorImage::filled([1, 1], egui::Color32::TRANSPARENT));
                    if let Err(e) = app.ai.push_pending(encoded, preview) {
                        app.ai.status = e;
                    }
                }
                Err(e) => app.ai.status = e,
            }
            return;
        }
    }
    for image in pasted {
        if let Err(e) = app.ai.attach_color_image(&image) {
            app.ai.status = e;
        }
    }
}

fn pending_row(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    if app.ai.pending_images.is_empty() {
        return;
    }
    ui.add_space(8.0);
    faint(ui, tl!("Attached images are sent with this request."));
    let mut remove = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
        for (i, image) in app.ai.pending_images.iter().enumerate() {
            if pending_thumb(ui, i, image) {
                remove = Some(i);
            }
        }
    });
    if let Some(i) = remove {
        app.ai.pending_images.remove(i);
    }
}

fn pending_thumb(ui: &mut egui::Ui, index: usize, image: &PendingImage) -> bool {
    let t = Tokens::get(ui.ctx());
    let thumb = thumb_size(image.preview.size);
    let pad = 4.0;
    let (rect, _) = ui.allocate_exact_size(thumb + egui::vec2(pad * 2.0, pad * 2.0), egui::Sense::hover());
    ui.painter().rect(rect, CornerRadius::same(8), t.field, Stroke::new(1.0, t.border), egui::StrokeKind::Inside);
    let texture = ui.ctx().load_texture(
        format!("ai-pending-{index}-{}", image.encoded.bytes().len()),
        image.preview.as_ref().clone(),
        egui::TextureOptions::LINEAR,
    );
    egui::Image::new((texture.id(), thumb)).paint_at(ui, egui::Rect::from_center_size(rect.center(), thumb));
    let x_size = egui::vec2(20.0, 20.0);
    let x_rect = egui::Rect::from_min_size(egui::pos2(rect.right() - x_size.x - 2.0, rect.top() + 2.0), x_size);
    ui.scope_builder(egui::UiBuilder::new().max_rect(x_rect), |ui| icons::button(ui, "x", 18.0, false, tl!("Remove image")).clicked()).inner
}

fn show_turn_images(ui: &mut egui::Ui, images: &[ChatImage]) {
    if images.is_empty() {
        return;
    }
    ui.horizontal_wrapped(|ui| {
        for (i, image) in images.iter().enumerate() {
            if let Ok(preview) = color_image_from_chat(image) {
                let size = thumb_size(preview.size);
                let texture = ui.ctx().load_texture(format!("ai-turn-{i}-{}", image.bytes().len()), preview, egui::TextureOptions::LINEAR);
                ui.image((texture.id(), size));
            }
        }
    });
}

fn color_image_from_chat(image: &ChatImage) -> Result<egui::ColorImage, String> {
    let (w, h, rgba) = image.rgba8()?;
    Ok(egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba))
}

fn color_thumb(image: &egui::ColorImage) -> egui::ColorImage {
    const MAX: usize = 96;
    let [w, h] = image.size;
    if w == 0 || h == 0 {
        return egui::ColorImage::filled([1, 1], egui::Color32::TRANSPARENT);
    }
    let long = w.max(h).max(1);
    let nw = (w.saturating_mul(MAX) / long).max(1);
    let nh = (h.saturating_mul(MAX) / long).max(1);
    let mut pixels = Vec::with_capacity(nw.saturating_mul(nh));
    for y in 0..nh {
        let sy = y.saturating_mul(h) / nh;
        for x in 0..nw {
            let sx = x.saturating_mul(w) / nw;
            pixels.push(image.pixels.get(sy.saturating_mul(w).saturating_add(sx)).copied().unwrap_or(egui::Color32::TRANSPARENT));
        }
    }
    egui::ColorImage::new([nw, nh], pixels)
}

fn thumb_size(size: [usize; 2]) -> egui::Vec2 {
    const MAX: f32 = 56.0;
    let w = size[0].max(1) as f32;
    let h = size[1].max(1) as f32;
    let k = MAX / w.max(h);
    egui::vec2((w * k).max(1.0), (h * k).max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ai_settings_never_persist_keys_consent_or_document_text() {
        let app = PdfCraftApp::new();
        assert!(!app.persist().contains("ai_provider"), "untouched defaults are not written");
        let mut app = PdfCraftApp::new();
        app.ai.config = Config { enabled: true, model: "test-model".into(), ..Default::default() };
        app.ai.key = Zeroizing::new("synthetic-secret".into());
        app.ai.input = "private question".into();
        app.ai.messages = vec![
            Turn { role: Role::User, text: "private question".into(), images: Vec::new(), citations: Vec::new() },
            Turn { role: Role::Assistant, text: "private answer".into(), images: Vec::new(), citations: Vec::new() },
        ];
        app.ai.consent = true;
        let png = ChatImage::from_rgba(1, 1, &[255, 0, 0, 255]).unwrap();
        app.ai.pending_images.push(PendingImage { encoded: png.clone(), preview: Arc::new(egui::ColorImage::filled([1, 1], egui::Color32::RED)) });
        app.ai.starter_images.push(png);
        let saved = app.persist();
        assert!(!saved.contains("synthetic-secret") && !saved.contains("private question") && !saved.contains("private answer"));
        assert!(!saved.contains("\\x89PNG") && !saved.contains("iVBOR"), "chat images stay in memory");
        let mut restored = PdfCraftApp::new();
        restored.restore(&saved);
        assert_eq!(restored.ai.config.model, "test-model");
        assert!(restored.ai.config.enabled);
        assert!(restored.ai.key.is_empty());
        assert!(!restored.ai.consent);
        assert!(restored.ai.messages.is_empty());
        app.ai.config.base_url = "https://user:synthetic-secret@example.com/v1".into();
        assert!(!app.persist().contains("synthetic-secret"));
        assert!(!app.persist().contains("ai_provider"));
    }

    #[test]
    fn ai_controls_require_preview_and_clear_consent_on_changes() {
        let mut app = PdfCraftApp::new();
        app.ai.config.enabled = true;
        app.ai.settings_open = true;
        app.run_command("ai.ask");
        assert_eq!(app.right, Some(RightPanel::Ai));
        assert!(app.dialog.is_none());
        assert_eq!(app.ai.task, Task::Ask);
        assert!(!app.ai.settings_open, "Ask/Summarize/Translate close the settings view");
        app.ai.settings_open = true;
        app.run_command("ai.summary");
        assert!(!app.ai.settings_open);
        assert_eq!(app.ai.task, Task::Summary);
        app.right = None;
        app.set_option("dialog", "ai").unwrap();
        assert_eq!(app.right, Some(RightPanel::Ai));
        assert!(app.dialog.is_none());
        app.set_option("ai-consent", "true").unwrap();
        app.set_option("ai-model", "test").unwrap();
        assert!(!app.ai.consent);
        app.ai.key = Zeroizing::new("synthetic-secret".into());
        app.set_option("ai-enabled", "false").unwrap();
        assert!(app.ai.key.is_empty());
        assert!(app.set_option("ai-key", "secret").is_err());
        assert!(!app.ai.state().to_string().contains("secret"));
    }

    fn ai_harness(enabled: bool) -> egui_kittest::Harness<'static, PdfCraftApp> {
        let bytes = pdfcraft_engine::Session::new().create_from_text("t", "Hello").expect("text pdf").as_ref().clone();
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(move |_cc| {
            let mut app = PdfCraftApp::new();
            app.set_option("language", "en").unwrap();
            app.open_bytes("t.pdf", None, bytes.clone()).expect("opens");
            if enabled {
                app.set_option("ai-enabled", "true").unwrap();
                app.ai.settings_open = false;
            }
            app.set_option("panel", "ai").unwrap();
            app
        });
        h.run_steps(4);
        h
    }

    #[test]
    fn ai_panel_has_masked_key_and_explicit_connection_control() {
        use egui_kittest::kittest::Queryable;
        let h = ai_harness(false);
        h.get_by_label("Provider settings");
        h.get_by_label("API base URL");
        h.get_by_label("Model ID");
        h.get_by_label("API key (optional for local servers)");
        h.get_by_label("Test connection");
        assert!(h.query_by_label("Preview").is_none(), "settings view must not show the chat starter");
    }

    #[test]
    fn ai_chat_surface_hides_provider_fields() {
        use egui_kittest::kittest::Queryable;
        let h = ai_harness(true);
        h.get_by_label("Summarize");
        h.get_by_label("Preview");
        h.get_by_label("Add image");
        assert!(h.query_by_label("API base URL").is_none(), "chat view must not show provider settings");
    }

    #[test]
    fn ai_chat_pastes_clipboard_images_and_can_remove_them() {
        let mut app = PdfCraftApp::new();
        app.ai.config.enabled = true;
        app.ai.settings_open = false;
        let image = egui::ColorImage::filled([8, 6], egui::Color32::from_rgb(20, 40, 80));
        app.ai.attach_color_image(&image).unwrap();
        assert_eq!(app.ai.pending_images.len(), 1);
        assert_eq!(app.ai.pending_images[0].encoded.mime(), "image/png");
        app.ai.pending_images.clear();
        app.ai.attach_color_image(&image).unwrap();
        app.ai.pending_images.remove(0);
        assert!(app.ai.pending_images.is_empty());
        app.run_command("ai.ask");
        assert!(!app.ai.settings_open);
        let saved = app.persist();
        assert!(!saved.contains("iVBOR"));
    }

    #[test]
    fn preview_and_question_edits_keep_pending_images() {
        let mut app = PdfCraftApp::new();
        app.ai.config.enabled = true;
        let image = egui::ColorImage::filled([8, 6], egui::Color32::from_rgb(20, 40, 80));
        app.ai.attach_color_image(&image).unwrap();
        assert_eq!(app.ai.pending_images.len(), 1);
        app.ai.invalidate();
        assert_eq!(app.ai.pending_images.len(), 1, "Preview and form edits must not drop attached images");
        app.ai.reset_thread();
        assert!(app.ai.pending_images.is_empty(), "New conversation clears attachments");
        app.ai.attach_color_image(&image).unwrap();
        app.set_option("ai-enabled", "false").unwrap();
        assert!(app.ai.pending_images.is_empty(), "Disabling AI drops unsaved attachments");
    }

    fn open_chat_on_page(preview_pages: Vec<usize>) -> PdfCraftApp {
        let bytes = pdfcraft_engine::Session::new().create_from_text("t", "Hello from this page").expect("text pdf").as_ref().clone();
        let mut app = PdfCraftApp::new();
        app.open_bytes("t.pdf", None, bytes).expect("opens");
        app.poll_ai();
        app.ai.config.enabled = true;
        app.ai.settings_open = false;
        app.ai.task = Task::Ask;
        app.ai.current_page = true;
        app.ai.input = "solve".into();
        app.ai.starter = "first question".into();
        app.ai.messages = vec![
            Turn { role: Role::User, text: "first question".into(), images: Vec::new(), citations: Vec::new() },
            Turn { role: Role::Assistant, text: "first answer".into(), images: Vec::new(), citations: Vec::new() },
        ];
        app.ai.preview = Some(ai::Prepared { text: "old page".into(), pages: preview_pages });
        app
    }

    #[test]
    fn a_page_change_refreshes_the_preview_inside_the_same_chat() {
        let mut app = open_chat_on_page(vec![2]);
        let mut fresh = false;
        for _ in 0..100 {
            app.poll_ai();
            fresh = app.ai.preview.as_ref().is_some_and(|preview| preview.pages == vec![1] && preview.text.contains("Hello"));
            if fresh && app.ai.job.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(fresh, "status: {}", app.ai.status);
        assert_eq!(app.ai.messages.len(), 2, "the conversation stays open");
        assert_eq!(app.ai.messages[0].text, "first question");
        assert_eq!(app.ai.input, "solve");
        assert!(!app.ai.status.contains("current page changed"));
    }

    #[test]
    fn a_follow_up_sends_after_a_page_change_without_a_blocker() {
        let mut app = open_chat_on_page(vec![2]);
        app.ai.config.model = "test-model".into();
        app.ai.config.base_url = "http://127.0.0.1:9/v1".into();
        app.start_ai(true);
        assert!(app.ai.job.as_ref().is_some_and(|job| job.pending.is_some()), "status: {}", app.ai.status);
        assert!(!app.ai.status.contains("current page changed"));
        assert_eq!(app.ai.messages[0].text, "first question");
        assert_eq!(app.ai.input, "solve");
        app.ai.job = None;
    }

    #[test]
    fn a_first_summary_shows_the_answer_while_it_is_still_arriving() {
        use egui_kittest::kittest::Queryable;
        use std::sync::{Arc, Mutex};
        let bytes = pdfcraft_engine::Session::new().create_from_text("t", "Hello from this page").expect("text pdf").as_ref().clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        // The worker has not finished. Dropping the sender would look like a failed request.
        std::mem::forget(sender);
        let partial = Arc::new(Mutex::new("Partial summary of revenue".to_string()));
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(move |_cc| {
            let mut app = PdfCraftApp::new();
            app.set_option("language", "en").unwrap();
            app.open_bytes("t.pdf", None, bytes).expect("opens");
            app.poll_ai();
            app.set_option("ai-enabled", "true").unwrap();
            app.ai.settings_open = false;
            app.ai.task = Task::Summary;
            app.ai.current_page = true;
            app.ai.preview = Some(ai::Prepared { text: "Hello from this page".into(), pages: vec![1] });
            app.ai.streaming = "Partial summary of revenue".into();
            app.ai.job = Some(Job {
                revision: app.ai.revision,
                pending: Some((
                    Turn { role: Role::User, text: "Summarize this page".into(), images: Vec::new(), citations: Vec::new() },
                    String::new(),
                )),
                receiver,
                activity: Arc::new(Mutex::new(String::new())),
                partial,
            });
            app.set_option("panel", "ai").unwrap();
            app
        });
        h.run_steps(4);
        h.get_by_label_contains("Partial summary of revenue");
        h.get_by_label_contains("Summarize this page");
        assert!(h.query_by_label("Preview").is_none(), "the first summary stays in the chat while the reply arrives");
    }
}
