//! An explicit preview/send flow. Background work is bounded to one job per window.
use crate::theme::{self, Tokens};
use crate::{Dialog, PdfCraftApp, RightPanel, icons, widgets};
use egui::{Align, CornerRadius, Layout, RichText, Stroke};
use pdfcraft_engine::{
    DocId,
    ai::{self, Config, Prepared, Task},
};
use zeroize::Zeroizing;

enum ResultData {
    Preview(Prepared),
    Answer(String),
    Connection(String),
}

struct Job {
    revision: u64,
    receiver: std::sync::mpsc::Receiver<Result<ResultData, String>>,
}

#[derive(Default)]
pub(crate) struct Assistant {
    pub(crate) config: Config,
    key: Zeroizing<String>,
    task: Task,
    input: String,
    current_page: bool,
    doc: Option<(DocId, u64)>,
    preview: Option<Prepared>,
    consent: bool,
    answer: String,
    status: String,
    revision: u64,
    job: Option<Job>,
    /// Inspector gear: provider fields. Off by default; forced on while AI is disabled.
    pub(crate) settings_open: bool,
}

impl Assistant {
    pub(crate) fn state(&self) -> serde_json::Value {
        // The control channel never receives the key, prompt, extracted text, or answer.
        serde_json::json!({"enabled": self.config.enabled, "busy": self.job.is_some(),
            "preview_ready": self.preview.is_some(), "answer_ready": !self.answer.is_empty(),
            "consent": self.consent, "status": self.status})
    }
    fn invalidate(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.preview = None;
        self.consent = false;
        self.answer.clear();
        self.status.clear();
    }

    fn launch(&mut self, ctx: Option<egui::Context>, work: impl FnOnce() -> Result<ResultData, String> + Send + 'static) {
        if self.job.is_some() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (sender, receiver) = std::sync::mpsc::channel();
            let job = Job { revision: self.revision, receiver };
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
            let _ = (ctx, work);
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
        if command == "ai.settings" {
            self.ai.settings_open = true;
        }
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
        if job.revision == self.ai.revision {
            self.ai.status.clear();
            match result {
                Ok(ResultData::Preview(preview)) => self.ai.preview = Some(preview),
                Ok(ResultData::Answer(answer)) => self.ai.answer = answer,
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
        if send {
            self.ai.config.validate()?;
        } else if !self.ai.config.enabled {
            return Err("AI assistant is off. Enable it and configure a provider first.".into());
        }
        self.ai.task.instruction(&self.ai.input)?;
        let (view, id) = self.active_ids().ok_or("Open a document first.")?;
        let doc = self.session.get(id).ok_or("Document is no longer open.")?;
        let pages = if self.ai.current_page { vec![self.views[view].current] } else { vec![] };
        // Check permissions and pending redactions again at Send, not just at preview time.
        let source = doc.ai_source(&pages)?;
        if !send {
            self.ai.invalidate();
            self.ai.launch(self.ctx.clone(), move || source.prepare().map(ResultData::Preview));
            return Ok(());
        }
        if !self.ai.consent {
            return Err("Review the text and allow this request before sending.".into());
        }
        let preview = self.ai.preview.as_ref().ok_or("Preview the document text before sending.")?;
        // A page navigation through the control channel must not send a stale current-page preview.
        if self.ai.current_page && preview.pages != vec![self.views[view].current + 1] {
            return Err("The current page changed. Preview the text again.".into());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut provider = ai::OpenAiProvider::new(self.ai.config.clone(), self.ai.key.to_string())?;
            let prompt = preview.prompt(self.ai.task, &self.ai.input)?;
            self.ai.consent = false;
            self.ai.answer.clear();
            self.ai.launch(self.ctx.clone(), move || {
                use pdfcraft_engine::ai::complete_prompt;
                complete_prompt(&mut provider, &prompt).map(ResultData::Answer)
            });
        }
        #[cfg(target_arch = "wasm32")]
        return Err("AI requests require the desktop app in this version.".into());
        #[cfg(not(target_arch = "wasm32"))]
        Ok(())
    }

    pub(crate) fn test_ai(&mut self) {
        self.poll_ai();
        if self.ai.job.is_some() {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        match ai::OpenAiProvider::new(self.ai.config.clone(), self.ai.key.to_string()) {
            Ok(mut provider) => self.ai.launch(self.ctx.clone(), move || ai::test_provider(&mut provider).map(ResultData::Connection)),
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
        self.ai.invalidate();
        match key {
            "ai-enabled" => self.ai.config.enabled = value.parse().map_err(|_| "ai-enabled must be true or false")?,
            "ai-url" => self.ai.config.base_url = value.into(),
            "ai-model" => self.ai.config.model = value.into(),
            "ai-input" => self.ai.input = value.into(),
            "ai-current-page" => self.ai.current_page = value.parse().map_err(|_| "ai-current-page must be true or false")?,
            "ai-legacy-tokens" => self.ai.config.legacy_max_tokens = value.parse().map_err(|_| "ai-legacy-tokens must be true or false")?,
            _ => return Err("Unknown AI setting. Enter API keys only in the masked desktop field.".into()),
        }
        if !self.ai.config.enabled {
            self.ai.key = Zeroizing::new(String::new());
        }
        Ok(())
    }
}

fn faint(ui: &mut egui::Ui, text: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(text).font(theme::regular(11.0)).color(t.text_faint));
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
            }
        }
    });
    if !app.ai.status.is_empty() {
        faint(ui, &app.ai.status);
    }
    test
}

pub(crate) fn body(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    ui.spacing_mut().item_spacing.y = 10.0;
    let busy = app.ai.job.is_some();
    let mut preview = false;
    let mut send = false;
    let mut test = false;
    let show_settings = app.ai.settings_open || !app.ai.config.enabled;
    if show_settings {
        test = provider_settings(ui, app);
        ui.add_space(4.0);
    }
    ui.add_enabled_ui(!busy, |ui| {
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
            changed |= ui.add(widgets::block(&mut app.ai.input).desired_rows(3).char_limit(2000).desired_width(f32::INFINITY)).labelled_by(label.id).changed();
        } else if app.ai.task == Task::Translate {
            let label = ui.label(tl!("Translate into"));
            changed |= ui.add(widgets::block(&mut app.ai.input).desired_rows(3).char_limit(2000).desired_width(f32::INFINITY)).labelled_by(label.id).changed();
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
        if let Some(p) = &app.ai.preview {
            let page_s = crate::i18n::trn(crate::i18n::current(), p.pages.len() as u64, "{n} page", "{n} pages");
            faint(ui, &format!("{} · {}", page_s, crate::panels::human_size(p.text.len())));
            ui.collapsing(tl!("Text to send"), |ui| {
                ui.label(&p.text);
            });
            let dest = format!("{} · {}", app.ai.config.base_url, app.ai.config.model);
            ui.add(egui::Label::new(RichText::new(dest).font(theme::regular(11.0)).color(t.text_faint)).truncate());
        }
        if app.ai.preview.is_some() {
            ui.add(widgets::checkbox(&mut app.ai.consent, tl!("Allow sending this text for this request")));
            faint(
                ui,
                tl!("Sending shares this text and your question with the chosen provider, which may charge for usage. A local server may forward requests elsewhere."),
            );
        }
    });
    if busy {
        ui.horizontal(|ui| {
            ui.spinner();
            faint(ui, tl!("Working…"));
        });
    } else if app.ai.preview.is_none() {
        ui.add_enabled_ui(app.ai.config.enabled, |ui| {
            preview = widgets::pill_button_fill(ui, tl!("Preview"), true).clicked();
        });
    } else {
        ui.add_enabled_ui(app.ai.consent, |ui| {
            send = widgets::pill_button_fill(ui, tl!("Send"), true).clicked();
        });
    }
    if !app.ai.answer.is_empty() {
        answer_card(ui, app);
    }
    if !show_settings && !app.ai.status.is_empty() {
        faint(ui, &app.ai.status);
    }
    #[cfg(target_arch = "wasm32")]
    faint(ui, tl!("AI requests require the desktop app in this version."));
    if preview {
        app.start_ai(false);
    }
    if send {
        app.start_ai(true);
    }
    if test {
        app.test_ai();
    }
}

fn answer_card(ui: &mut egui::Ui, app: &mut PdfCraftApp) {
    let t = Tokens::get(ui.ctx());
    let mut copy = false;
    egui::Frame::NONE
        .fill(t.field)
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(egui::Margin::same(10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(tl!("Answer")).font(theme::semibold(12.5)));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if icons::button(ui, "copy", 22.0, false, tl!("Copy answer")).clicked() {
                        copy = true;
                    }
                });
            });
            faint(ui, tl!("AI-generated answer — verify against the document. Page citations are not independently verified."));
            ui.add_space(6.0);
            widgets::markdown::show(ui, &app.ai.answer);
        });
    if copy {
        ui.ctx().copy_text(app.ai.answer.clone());
    }
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
        app.ai.answer = "private answer".into();
        app.ai.consent = true;
        let saved = app.persist();
        assert!(!saved.contains("synthetic-secret") && !saved.contains("private question") && !saved.contains("private answer"));
        let mut restored = PdfCraftApp::new();
        restored.restore(&saved);
        assert_eq!(restored.ai.config.model, "test-model");
        assert!(restored.ai.config.enabled);
        assert!(restored.ai.key.is_empty());
        assert!(!restored.ai.consent);
        app.ai.config.base_url = "https://user:synthetic-secret@example.com/v1".into();
        assert!(!app.persist().contains("synthetic-secret"));
        assert!(!app.persist().contains("ai_provider"));
    }

    #[test]
    fn ai_controls_require_preview_and_clear_consent_on_changes() {
        let mut app = PdfCraftApp::new();
        app.run_command("ai.ask");
        assert_eq!(app.right, Some(RightPanel::Ai));
        assert!(app.dialog.is_none());
        assert_eq!(app.ai.task, Task::Ask);
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

    #[test]
    fn ai_panel_has_masked_key_and_explicit_connection_control() {
        use egui_kittest::kittest::Queryable;
        let bytes = pdfcraft_engine::Session::new().create_from_text("t", "Hello").expect("text pdf").as_ref().clone();
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(move |_cc| {
            let mut app = PdfCraftApp::new();
            app.set_option("language", "en").unwrap();
            app.open_bytes("t.pdf", None, bytes.clone()).expect("opens");
            app.set_option("panel", "ai").unwrap();
            app
        });
        h.run_steps(4);
        h.get_by_label("Provider settings");
        h.get_by_label("API base URL");
        h.get_by_label("Model ID");
        h.get_by_label("API key (optional for local servers)");
        h.get_by_label("Test connection");
        h.get_by_label("Preview");
    }
}
