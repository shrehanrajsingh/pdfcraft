//! Read-only CommonMark for model answers. HTML from the model is not rendered.

use egui::{FontId, RichText, Stroke};
use pulldown_cmark::{Event, HeadingLevel, Parser, Tag, TagEnd};

use crate::theme::{self, Tokens};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub text: String,
    pub strong: bool,
    pub emphasis: bool,
    pub code: bool,
    pub link: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Block {
    Heading { level: u8, spans: Vec<Span> },
    Paragraph(Vec<Span>),
    ListItem { depth: u8, ordered: bool, index: u32, spans: Vec<Span> },
    Code(String),
    Quote(Vec<Span>),
    Rule,
}

#[derive(Clone, Default)]
struct Inline {
    strong: bool,
    emphasis: bool,
    code: bool,
    link: Option<String>,
}

struct List {
    ordered: bool,
    next: u32,
}

/// Parse CommonMark into display blocks. Images and raw HTML are dropped.
pub(crate) fn parse(markdown: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut inline = Inline::default();
    let mut spans = Vec::new();
    let mut lists: Vec<List> = Vec::new();
    let mut heading: Option<u8> = None;
    let mut quote = 0_u8;
    let mut code: Option<String> = None;
    let mut skip_image = 0_u8;
    let mut item_open = false;

    for event in Parser::new(markdown) {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
                heading = Some(heading_level(level));
            }
            Event::End(TagEnd::Heading(_)) => {
                if !spans.is_empty() {
                    blocks.push(Block::Heading { level: heading.unwrap_or(1), spans: std::mem::take(&mut spans) });
                }
                heading = None;
            }
            Event::Start(Tag::Paragraph | Tag::HtmlBlock) => {}
            Event::End(TagEnd::Paragraph) => {
                flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
            }
            Event::End(TagEnd::HtmlBlock) => {}
            Event::Start(Tag::BlockQuote(_)) => {
                flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
                quote = quote.saturating_add(1);
            }
            Event::End(TagEnd::BlockQuote(_)) => {
                flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
                quote = quote.saturating_sub(1);
            }
            Event::Start(Tag::List(start)) => {
                flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
                let next = start.and_then(|n| u32::try_from(n).ok()).unwrap_or(1);
                lists.push(List { ordered: start.is_some(), next });
            }
            Event::End(TagEnd::List(_)) => {
                flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
                lists.pop();
            }
            Event::Start(Tag::Item) => {
                flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
                item_open = true;
            }
            Event::End(TagEnd::Item) => {
                let depth = lists.len().saturating_sub(1).min(8) as u8;
                if let Some(list) = lists.last_mut() {
                    let index = list.next;
                    list.next = list.next.saturating_add(1);
                    if !spans.is_empty() {
                        blocks.push(Block::ListItem { depth, ordered: list.ordered, index, spans: std::mem::take(&mut spans) });
                    }
                }
                item_open = false;
            }
            Event::Start(Tag::CodeBlock(_)) => {
                flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
                code = Some(String::new());
            }
            Event::End(TagEnd::CodeBlock) => {
                if let Some(text) = code.take() {
                    let text = text.trim_end_matches('\n').to_string();
                    if !text.is_empty() {
                        blocks.push(Block::Code(text));
                    }
                }
            }
            Event::Start(Tag::Emphasis) => inline.emphasis = true,
            Event::End(TagEnd::Emphasis) => inline.emphasis = false,
            Event::Start(Tag::Strong) => inline.strong = true,
            Event::End(TagEnd::Strong) => inline.strong = false,
            Event::Start(Tag::Link { dest_url, .. }) => inline.link = Some(dest_url.into_string()),
            Event::End(TagEnd::Link) => inline.link = None,
            Event::Start(Tag::Image { .. }) => skip_image = skip_image.saturating_add(1),
            Event::End(TagEnd::Image) => skip_image = skip_image.saturating_sub(1),
            Event::Rule => {
                flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
                blocks.push(Block::Rule);
            }
            Event::Text(text) => {
                if skip_image > 0 {
                    continue;
                }
                if let Some(buf) = code.as_mut() {
                    buf.push_str(text.as_ref());
                    continue;
                }
                push_text(&mut spans, &inline, text.as_ref());
            }
            Event::Code(text) => {
                if skip_image > 0 {
                    continue;
                }
                if let Some(buf) = code.as_mut() {
                    buf.push_str(text.as_ref());
                    continue;
                }
                let mut code_inline = inline.clone();
                code_inline.code = true;
                push_text(&mut spans, &code_inline, text.as_ref());
            }
            Event::SoftBreak => {
                if skip_image == 0 && code.is_none() {
                    push_text(&mut spans, &inline, " ");
                }
            }
            Event::HardBreak => {
                if skip_image == 0 && code.is_none() {
                    push_text(&mut spans, &inline, "\n");
                }
            }
            Event::Html(_)
            | Event::InlineHtml(_)
            | Event::InlineMath(_)
            | Event::DisplayMath(_)
            | Event::FootnoteReference(_)
            | Event::TaskListMarker(_) => {}
            Event::Start(_) | Event::End(_) => {}
        }
    }
    flush_paragraph(&mut blocks, &mut spans, heading, quote, item_open, lists.len());
    blocks
}

fn heading_level(level: HeadingLevel) -> u8 {
    match level {
        HeadingLevel::H1 => 1,
        HeadingLevel::H2 => 2,
        HeadingLevel::H3 => 3,
        HeadingLevel::H4 => 4,
        HeadingLevel::H5 => 5,
        HeadingLevel::H6 => 6,
    }
}

fn push_text(spans: &mut Vec<Span>, inline: &Inline, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(last) = spans.last_mut()
        && last.strong == inline.strong
        && last.emphasis == inline.emphasis
        && last.code == inline.code
        && last.link == inline.link
    {
        last.text.push_str(text);
        return;
    }
    spans.push(Span { text: text.to_string(), strong: inline.strong, emphasis: inline.emphasis, code: inline.code, link: inline.link.clone() });
}

fn flush_paragraph(blocks: &mut Vec<Block>, spans: &mut Vec<Span>, heading: Option<u8>, quote: u8, item_open: bool, _lists: usize) {
    if spans.is_empty() || heading.is_some() || item_open {
        return;
    }
    let taken = std::mem::take(spans);
    if quote > 0 {
        blocks.push(Block::Quote(taken));
    } else {
        blocks.push(Block::Paragraph(taken));
    }
}

/// Draw parsed CommonMark. Links show their URL on hover and are not opened.
pub(crate) fn show(ui: &mut egui::Ui, markdown: &str) {
    let t = Tokens::get(ui.ctx());
    for block in parse(markdown) {
        match block {
            Block::Heading { level, spans } => {
                let size = match level {
                    1 => 20.0,
                    2 => 16.5,
                    3 => 14.5,
                    _ => 13.5,
                };
                show_spans(ui, &spans, theme::semibold(size), t.text);
            }
            Block::Paragraph(spans) => show_spans(ui, &spans, theme::regular(13.0), t.text),
            Block::Quote(spans) => {
                egui::Frame::NONE.inner_margin(egui::Margin { left: 10, right: 0, top: 0, bottom: 0 }).show(ui, |ui| {
                    show_spans(ui, &spans, theme::regular(13.0), t.text_muted);
                });
            }
            Block::ListItem { depth, ordered, index, spans } => {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.add_space(16.0 * f32::from(depth));
                    let marker = if ordered { format!("{index}. ") } else { "• ".into() };
                    ui.label(RichText::new(marker).font(theme::regular(13.0)).color(t.text));
                    show_spans_inner(ui, &spans, theme::regular(13.0), t.text);
                });
            }
            Block::Code(text) => {
                egui::Frame::NONE.fill(t.field).stroke(Stroke::new(1.0, t.border)).inner_margin(egui::Margin::same(8)).corner_radius(4).show(
                    ui,
                    |ui| {
                        ui.add(egui::Label::new(RichText::new(&text).font(FontId::monospace(12.0)).color(t.text)).wrap());
                    },
                );
            }
            Block::Rule => {
                ui.add_space(4.0);
                ui.separator();
            }
        }
        ui.add_space(10.0);
    }
}

fn show_spans(ui: &mut egui::Ui, spans: &[Span], font: FontId, color: egui::Color32) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        show_spans_inner(ui, spans, font, color);
    });
}

fn show_spans_inner(ui: &mut egui::Ui, spans: &[Span], font: FontId, color: egui::Color32) {
    for span in spans {
        let font = if span.code { FontId::monospace(font.size) } else { font.clone() };
        let mut text = RichText::new(&span.text).font(font).color(color);
        if span.strong {
            text = text.strong();
        }
        if span.emphasis {
            text = text.italics();
        }
        if span.link.is_some() {
            text = text.underline();
        }
        let response = ui.label(text);
        if let Some(url) = &span.link {
            response.on_hover_text(url);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Block, Span, parse};

    fn text(s: &str) -> Span {
        Span { text: s.into(), strong: false, emphasis: false, code: false, link: None }
    }

    fn strong(s: &str) -> Span {
        Span { text: s.into(), strong: true, emphasis: false, code: false, link: None }
    }

    fn code(s: &str) -> Span {
        Span { text: s.into(), strong: false, emphasis: false, code: true, link: None }
    }

    #[test]
    fn parse_maps_headings_lists_bold_and_code() {
        let md = "# Title\n\n- one\n- **two**\n\nUse `code` here.\n\n[Page 3] is a citation, not a link.\n";
        let blocks = parse(md);
        assert_eq!(blocks[0], Block::Heading { level: 1, spans: vec![text("Title")] });
        assert_eq!(blocks[1], Block::ListItem { depth: 0, ordered: false, index: 1, spans: vec![text("one")] });
        assert_eq!(blocks[2], Block::ListItem { depth: 0, ordered: false, index: 2, spans: vec![strong("two")] });
        assert_eq!(blocks[3], Block::Paragraph(vec![text("Use "), code("code"), text(" here.")]));
        match &blocks[4] {
            Block::Paragraph(spans) => {
                assert!(spans.iter().all(|s| s.link.is_none()), "bracket page citations stay literal");
                let joined: String = spans.iter().map(|s| s.text.as_str()).collect();
                assert!(joined.contains("[Page 3]"));
            }
            other => panic!("expected paragraph, got {other:?}"),
        }
    }

    #[test]
    fn parse_skips_html_and_keeps_link_destinations() {
        let blocks = parse("See [docs](https://example.com/a) and <em>no html</em>.");
        let Block::Paragraph(spans) = &blocks[0] else { panic!("paragraph") };
        assert!(spans.iter().any(|s| s.link.as_deref() == Some("https://example.com/a")));
        let joined: String = spans.iter().map(|s| s.text.as_str()).collect();
        assert!(!joined.to_lowercase().contains("<em>"));
    }
}
