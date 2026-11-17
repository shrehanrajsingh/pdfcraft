//! Read-only CommonMark for model answers, rendered by `egui_commonmark`.
//!
//! `\[…\]` / `\(…\)` are rewritten to `$$…$$` / `$…$` so pulldown-cmark's math events fire;
//! TeX itself is laid out by RaTeX. HTML is not rendered. Markdown images are not fetched.

use egui_commonmark::{CommonMarkCache, CommonMarkViewer};

use super::math;

/// Persistent CommonMark widget cache (syntax, links) for one AI panel.
#[derive(Default)]
pub(crate) struct Cache {
    inner: CommonMarkCache,
}

/// Draw parsed CommonMark. Links keep their destinations; they are not rewritten.
pub(crate) fn show(ui: &mut egui::Ui, cache: &mut Cache, markdown: &str) {
    let prepared = with_tex_delimiters(markdown);
    let math_fn = |ui: &mut egui::Ui, tex: &str, inline: bool| {
        math::show(ui, tex, !inline);
    };
    CommonMarkViewer::new().explicit_image_uri_scheme(true).max_image_width(Some(0)).render_math_fn(Some(&math_fn)).show(
        ui,
        &mut cache.inner,
        &prepared,
    );
}

/// Map TeX `\[…\]` / `\(…\)` (outside code) onto pulldown-cmark's `$` / `$$` math.
pub(crate) fn with_tex_delimiters(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len());
    let mut rest = markdown;
    let mut at_line = true;
    while !rest.is_empty() {
        if at_line && let Some((taken, after)) = take_fenced(rest) {
            out.push_str(taken);
            rest = after;
            at_line = rest.starts_with('\n') || rest.is_empty();
            continue;
        }
        if let Some((taken, after)) = take_inline_code(rest) {
            out.push_str(taken);
            rest = after;
            at_line = false;
            continue;
        }
        if rest.starts_with("$$") {
            out.push_str("$$");
            rest = rest.get(2..).unwrap_or("");
            at_line = false;
            continue;
        }
        if rest.starts_with('$') {
            out.push('$');
            rest = rest.get(1..).unwrap_or("");
            at_line = false;
            continue;
        }
        if let Some((inner, after)) = delimited(rest, "\\[", "\\]") {
            out.push_str("$$");
            out.push_str(inner);
            out.push_str("$$");
            rest = after;
            at_line = false;
            continue;
        }
        if let Some((inner, after)) = delimited(rest, "\\(", "\\)") {
            out.push('$');
            out.push_str(inner);
            out.push('$');
            rest = after;
            at_line = false;
            continue;
        }
        let Some(ch) = rest.chars().next() else { break };
        out.push(ch);
        rest = rest.get(ch.len_utf8()..).unwrap_or("");
        at_line = ch == '\n';
    }
    out
}

fn take_fenced(rest: &str) -> Option<(&str, &str)> {
    let marker = rest.as_bytes().first().copied()?;
    if marker != b'`' && marker != b'~' {
        return None;
    }
    let open = rest.bytes().take_while(|b| *b == marker).count();
    if open < 3 {
        return None;
    }
    let after_open = rest.get(open..)?;
    let line_end = after_open.find('\n').map(|i| i.saturating_add(1)).unwrap_or(after_open.len());
    let body = after_open.get(line_end..)?;
    let mut offset = 0;
    while offset < body.len() {
        let line = body.get(offset..)?;
        let close = line.bytes().take_while(|b| *b == marker).count();
        if close >= open {
            let end = open.saturating_add(line_end).saturating_add(offset).saturating_add(close);
            return Some((rest.get(..end)?, rest.get(end..)?));
        }
        offset = offset.saturating_add(line.find('\n').map(|i| i.saturating_add(1)).unwrap_or(line.len()));
        if offset == 0 {
            break;
        }
    }
    Some((rest, ""))
}

fn take_inline_code(rest: &str) -> Option<(&str, &str)> {
    let ticks = rest.bytes().take_while(|b| *b == b'`').count();
    if ticks == 0 || ticks >= 3 {
        return None;
    }
    let needle = "`".repeat(ticks);
    let inner = rest.get(ticks..)?;
    let at = inner.find(&needle)?;
    let end = ticks.saturating_add(at).saturating_add(ticks);
    Some((rest.get(..end)?, rest.get(end..)?))
}

fn delimited<'a>(rest: &'a str, open: &str, close: &str) -> Option<(&'a str, &'a str)> {
    if !rest.starts_with(open) {
        return None;
    }
    let inner_start = open.len();
    let rel = rest.get(inner_start..)?.find(close)?;
    let inner_end = inner_start.saturating_add(rel);
    let after = inner_end.saturating_add(close.len());
    Some((rest.get(inner_start..inner_end)?, rest.get(after..)?))
}

#[cfg(test)]
mod tests {
    use super::{Cache, with_tex_delimiters};
    use pulldown_cmark::{Event, Options, Parser, Tag};

    #[test]
    fn parse_maps_pipe_tables() {
        let md = "| Load | Value |\n| --- | ---: |\n| Slab self-weight, (0.175\\times25) | 4.375 kN/m² |\n| Live load | 4.000 kN/m² |\n";
        let prepared = with_tex_delimiters(md);
        let mut saw_table = false;
        let mut texts = Vec::new();
        for event in Parser::new_ext(&prepared, Options::ENABLE_TABLES | Options::ENABLE_MATH) {
            match event {
                Event::Start(Tag::Table(_)) => saw_table = true,
                Event::Text(t) => texts.push(t.into_string()),
                _ => {}
            }
        }
        assert!(saw_table, "{texts:?}");
        assert!(texts.iter().any(|c| c == "Load"), "{texts:?}");
        assert!(texts.iter().any(|c| c == "Value"), "{texts:?}");
        assert!(!texts.iter().any(|c| c.contains("| Load |")), "{texts:?}");
    }

    #[test]
    fn parse_maps_tex_delimiters_and_keeps_page_citations() {
        let md = "Ratio:\n\n\\[ \\frac{9}{3.5}=2.57>2 \\]\n\nInline $w_u=14.82$ and [Page 365].\n";
        let prepared = with_tex_delimiters(md);
        assert!(prepared.contains("$$"), "{prepared}");
        assert!(prepared.contains("[Page 365]"), "{prepared}");
        let mut display = false;
        let mut inline = false;
        for event in Parser::new_ext(&prepared, Options::ENABLE_TABLES | Options::ENABLE_MATH) {
            match event {
                Event::DisplayMath(s) if s.contains(r"\frac{9}{3.5}") => display = true,
                Event::InlineMath(s) if s.contains("w_u") => inline = true,
                _ => {}
            }
        }
        assert!(display, "{prepared}");
        assert!(inline, "{prepared}");
        let mut texts = Vec::new();
        let mut link = false;
        for event in Parser::new_ext(&prepared, Options::ENABLE_TABLES | Options::ENABLE_MATH) {
            match event {
                Event::Start(Tag::Link { .. }) => link = true,
                Event::Text(t) => texts.push(t.into_string()),
                _ => {}
            }
        }
        let joined = texts.concat();
        assert!(joined.contains("[Page 365]") || joined.contains("Page 365"), "{texts:?}");
        assert!(!link, "{prepared}");
    }

    #[test]
    fn extract_does_not_eat_code_or_citations() {
        let out = with_tex_delimiters("use `$x$` and [Page 1]\n");
        assert!(out.contains("$x$"), "{out}");
        assert!(out.contains("[Page 1]"), "{out}");
        let prepared = with_tex_delimiters("use `$x$` and [Page 1]\n");
        let math = Parser::new_ext(&prepared, Options::ENABLE_TABLES | Options::ENABLE_MATH).any(|e| matches!(e, Event::InlineMath(_)));
        assert!(!math, "{prepared}");
    }

    #[test]
    fn show_uses_commonmark_for_tables() {
        let md = "| Load | Value |\n| --- | --- |\n| A | 1 |\n";
        let ctx = egui::Context::default();
        let output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let mut cache = Cache::default();
            super::show(ui, &mut cache, md);
        });
        let texts: Vec<String> = output
            .shapes
            .iter()
            .filter_map(|shape| if let egui::Shape::Text(t) = &shape.shape { Some(t.galley.job.text.clone()) } else { None })
            .collect();
        output.drop_without_applying_deltas();
        assert!(texts.iter().any(|t| t.contains("Load")), "{texts:?}");
        assert!(texts.iter().any(|t| t.contains("Value")), "{texts:?}");
        assert!(!texts.iter().any(|t| t.contains("| Load |")), "{texts:?}");
    }
}
