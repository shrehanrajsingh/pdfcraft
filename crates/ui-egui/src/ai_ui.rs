//! Settings and chat are exclusive inspector views. Background work is one job per window.
use crate::theme::{self, Tokens};
use crate::{Dialog, PdfCraftApp, RightPanel, icons, widgets};
use egui::{Align, CornerRadius, Layout, RichText, Stroke};
use pdfcraft_engine::{
    DocId,
    ai::{self, ChatImage, Config, Prepared, Role, Task, Turn},
};
use std::sync::Arc;
use zeroize::Zeroizing;

struct PendingImage {
    encoded: ChatImage,
    preview: Arc<egui::ColorImage>,
}

enum ResultData {
    Preview(Prepared),
    Answer(String),
    Connection(String),
}

struct Job {
    revision: u64,
    pending: Option<(Turn, String)>,
    receiver: std::sync::mpsc::Receiver<Result<ResultData, String>>,
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

    fn invalidate(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.preview = None;
        self.consent = false;
        self.messages.clear();
        self.starter.clear();
        self.status.clear();
        self.starter_images.clear();
        // Keep pending_images: Preview and question/source edits must not drop a paste the user
        // still intends to send. reset_thread and disabling AI clear them.
    }

    pub(crate) fn reset_thread(&mut self) {
        self.invalidate();
        self.pending_images.clear();
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
        work: impl FnOnce() -> Result<ResultData, String> + Send + 'static,
    ) {
        if self.job.is_some() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver) = std::sync::mpsc::channel();
            let job = Job { revision: self.revision, pending, receiver };
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
            let _ = (ctx, pending, work);
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
            self.ai.invalidate();
            self.ai.doc = doc;
        }
        let Some(job) = &self.ai.job else { return };
        let result = match job.receiver.try_recv() {
            Ok(result) => result,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("AI worker stopped unexpectedly.".into()),
        };
        let pending = job.pending.clone();
        if job.revision == self.ai.revision {
            self.ai.status.clear();
            match result {
                Ok(ResultData::Preview(preview)) => self.ai.preview = Some(preview),
                Ok(ResultData::Answer(answer)) => {
                    if let Some((user, starter)) = pending {
                        if self.ai.messages.is_empty() {
                            self.ai.starter = starter;
                        }
                        self.ai.messages.push(user);
                        self.ai.messages.push(Turn { role: Role::Assistant, text: answer, images: Vec::new() });
                        self.ai.input.clear();
                    }
                }
                Ok(ResultData::Connection(status)) | Err(status) => self.ai.status = status,
            }
        }
        self.ai.job = None;
    }

    pub(crate) fn start_ai(&mut self, send: bool) {
        self.poll_ai();
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
        let source = self.ai_source()?;
        self.ai.invalidate();
        self.ai.launch(self.ctx.clone(), None, move || source.prepare().map(ResultData::Preview));
        Ok(())
    }

    fn prepare_first_send(&mut self) -> Result<(), String> {
        self.ai.config.validate()?;
        self.ai.task.instruction(&self.ai.input)?;
        let view = self.active_ids().ok_or("Open a document first.")?.0;
        if !self.ai.consent {
            return Err("Review the text and allow this request before sending.".into());
        }
        self.check_preview_page(view)?;
        self.launch_completion(&[], self.ai.input.clone(), starter_label(self.ai.task, self.ai.current_page, &self.ai.input))
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
        let view = self.active_ids().ok_or("Open a document first.")?.0;
        self.check_preview_page(view)?;
        let images: Vec<ChatImage> = self.ai.pending_images.iter().map(|p| p.encoded.clone()).collect();
        let mut turns = self.ai.api_turns();
        turns.push(Turn { role: Role::User, text: follow.to_owned(), images });
        let label = follow.to_owned();
        let starter = self.ai.starter.clone();
        self.launch_completion(&turns, starter, label)
    }

    fn ai_source(&self) -> Result<ai::Source, String> {
        let (view, id) = self.active_ids().ok_or("Open a document first.")?;
        let doc = self.session.get(id).ok_or("Document is no longer open.")?;
        let pages = if self.ai.current_page { vec![self.views[view].current] } else { vec![] };
        doc.ai_source(&pages)
    }

    fn check_preview_page(&self, view: usize) -> Result<(), String> {
        let preview = self.ai.preview.as_ref().ok_or("Preview the document text before sending.")?;
        if self.ai.current_page && preview.pages != vec![self.views[view].current + 1] {
            return Err("The current page changed. Preview the text again.".into());
        }
        Ok(())
    }

    fn launch_completion(&mut self, turns: &[Turn], starter: String, user_text: String) -> Result<(), String> {
        let preview = self.ai.preview.as_ref().ok_or("Preview the document text before sending.")?;
        let snapshot_images: Vec<ChatImage> =
            if turns.is_empty() { self.ai.pending_images.iter().map(|p| p.encoded.clone()).collect() } else { self.ai.starter_images.clone() };
        let messages = preview.conversation_messages(self.ai.task, &starter, turns, &snapshot_images)?;
        let user_images = if turns.is_empty() { snapshot_images.clone() } else { turns.last().map(|t| t.images.clone()).unwrap_or_default() };
        let user = Turn { role: Role::User, text: user_text, images: user_images };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut provider = ai::OpenAiProvider::new(self.ai.config.clone(), self.ai.key.to_string())?;
            self.ai.pending_images.clear();
            if turns.is_empty() {
                self.ai.starter_images = snapshot_images;
            }
            self.ai.launch(self.ctx.clone(), Some((user, starter)), move || ai::complete_messages(&mut provider, &messages).map(ResultData::Answer));
            Ok(())
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (messages, user);
            Err("AI requests require the desktop app in this version.".into())
        }
    }

    pub(crate) fn test_ai(&mut self) {
        self.poll_ai();
        if self.ai.job.is_some() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        match ai::OpenAiProvider::new(self.ai.config.clone(), self.ai.key.to_string()) {
            Ok(mut provider) => self.ai.launch(self.ctx.clone(), None, move || ai::test_provider(&mut provider).map(ResultData::Connection)),
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
        changed |= ui.add(widgets::checkbox(&mut a.config.legacy_max_tokens, tl!("Use legacy max_tokens (older compatible servers)"))).changed();
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
    if app.ai.has_thread() {
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

fn pane_scroll(ui: &mut egui::Ui, rect: egui::Rect, salt: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.scope_builder(egui::UiBuilder::new().max_rect(rect).layout(Layout::top_down(Align::Min)), |ui| {
        egui::ScrollArea::vertical().id_salt(salt).auto_shrink([false, false]).show(ui, add);
    });
}

fn starter_view(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    let mut preview = false;
    let mut send = false;
    let mut attach = false;
    let (main, dock) = pane_regions(ui, 48.0);
    pane_scroll(ui, main, "ai-start", |ui| {
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
    let dock_h = if app.ai.pending_images.is_empty() { 128.0 } else { 188.0 };
    let (main, dock) = pane_regions(ui, dock_h);
    pane_scroll(ui, main, "ai-thread", |ui| {
        transcript(ui, app);
        if !app.ai.status.is_empty() {
            ui.add_space(6.0);
            faint(ui, &app.ai.status);
        }
    });
    ui.scope_builder(egui::UiBuilder::new().max_rect(dock).layout(Layout::top_down(Align::Min)), |ui| {
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
            let consent = if app.ai.pending_images.is_empty() {
                tl!("Allow sending this text for this request")
            } else {
                tl!("Allow sending this text and attached images for this request")
            };
            ui.add(widgets::checkbox(&mut app.ai.consent, consent));
            faint(
                ui,
                if app.ai.pending_images.is_empty() {
                    tl!("Sending shares this text and your question with the chosen provider, which may charge for usage. A local server may forward requests elsewhere.")
                } else {
                    tl!("Sending shares this text, your question, and the attached images with the chosen provider, which may charge for usage. A local server may forward requests elsewhere.")
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
            faint(ui, tl!("Working…"));
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
                assistant_bubble(ui, &mut app.ai.markdown, &text, first_assistant);
                first_assistant = false;
            }
            None => break,
        }
        i = i.saturating_add(1);
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

fn assistant_bubble(ui: &mut egui::Ui, cache: &mut widgets::markdown::Cache, text: &str, disclaimer: bool) {
    let t = Tokens::get(ui.ctx());
    let mut copy = false;
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
                faint(ui, tl!("AI-generated answer — verify against the document. Page citations are not independently verified."));
                ui.add_space(6.0);
            }
            widgets::markdown::show(ui, cache, text);
        });
    if copy {
        ui.ctx().copy_text(text.to_owned());
    }
}

fn composer(ui: &mut egui::Ui, app: &mut PdfCraftApp, send: &mut bool) {
    let busy = app.ai.job.is_some();
    let mut attach = false;
    ui.add_space(8.0);
    faint(ui, tl!("Each message is sent with the page text and this conversation."));
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
            faint(ui, tl!("Working…"));
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
            Turn { role: Role::User, text: "private question".into(), images: Vec::new() },
            Turn { role: Role::Assistant, text: "private answer".into(), images: Vec::new() },
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
}
