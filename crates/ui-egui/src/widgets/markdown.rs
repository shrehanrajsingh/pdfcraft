//! Read-only CommonMark for model answers, rendered by `egui_commonmark`.
//!
//! Pipe tables are laid out in this module. The commonmark viewer puts a table in a
//! horizontal scroller, which leaves an empty band under the last row and paints the
//! next paragraph on that border. `\[…\]` / `\(…\)` are rewritten to `$$…$$` / `$…$`
//! so pulldown-cmark's math events fire; TeX itself is laid out by RaTeX. HTML is not
//! rendered. Markdown images are not fetched.

use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use pulldown_cmark::{Alignment, Event, Options, Parser, Tag, TagEnd};

use super::math;

/// Persistent CommonMark widget cache (syntax, links) for one AI panel.
#[derive(Default)]
pub(crate) struct Cache {
    inner: CommonMarkCache,
}

/// A checked citation the answer can point at.
pub(crate) struct CiteRef<'a> {
    pub id: &'a str,
    /// 1-based page number.
    pub page: usize,
    pub quote: &'a str,
    pub page_level: bool,
}

/// Draw markdown, turning `[c1]` and `[Page N]` into buttons when `cites` knows them.
/// Returns the id of a citation the user clicked.
pub(crate) fn show_cited(ui: &mut egui::Ui, cache: &mut Cache, markdown: &str, cites: &[CiteRef<'_>]) -> Option<String> {
    limit_width(ui);
    let prepared = with_tex_delimiters(markdown);
    let mut clicked = None;
    for segment in segments(&prepared) {
        match segment {
            Seg::Prose(text) => {
                if let Some(id) = show_pieces(ui, cache, text, cites) {
                    clicked = Some(id);
                }
            }
            Seg::Table(table) => render_table(ui, cache, &table),
        }
    }
    clicked
}

fn show_pieces(ui: &mut egui::Ui, cache: &mut Cache, markdown: &str, cites: &[CiteRef<'_>]) -> Option<String> {
    let mut clicked = None;
    for piece in pieces(markdown) {
        match piece {
            Piece::Text(text) => {
                if !text.trim().is_empty() {
                    show_prose(ui, cache, text);
                }
            }
            Piece::Cite { lookup, label } => {
                let known = match lookup {
                    Lookup::Id(id) => cites.iter().find(|cite| cite.id == id),
                    Lookup::Page(page) => cites.iter().find(|cite| cite.id == format!("page-{page}")),
                };
                let response = ui.small_button(label);
                if let Some(cite) = known {
                    response.clone().on_hover_ui(|ui| {
                        ui.label(format!("Page {}", cite.page));
                        if cite.page_level {
                            ui.label("Page-level location. The exact passage box was not available.");
                        }
                        if !cite.quote.is_empty() {
                            ui.label(cite.quote);
                        }
                    });
                    if response.clicked() {
                        clicked = Some(cite.id.to_string());
                    }
                }
            }
        }
    }
    clicked
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lookup<'a> {
    Id(&'a str),
    Page(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Piece<'a> {
    Text(&'a str),
    Cite { lookup: Lookup<'a>, label: &'a str },
}

/// Split citation markers that sit outside code. `[c1]` looks up `c1`. `[Page 3]` looks up `page-3`.
fn pieces(markdown: &str) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    let bytes = markdown;
    let mut rest = bytes;
    let mut at_line = true;
    let mut text_from = bytes;
    while !rest.is_empty() {
        if at_line && let Some((_taken, after)) = take_fenced(rest) {
            rest = after;
            at_line = rest.starts_with('\n') || rest.is_empty();
            continue;
        }
        if let Some((_taken, after)) = take_inline_code(rest) {
            rest = after;
            at_line = false;
            continue;
        }
        if let Some((lookup, label, after)) = citation_at(rest) {
            push_text(&mut out, bytes, text_from, rest);
            out.push(Piece::Cite { lookup, label });
            rest = after;
            text_from = rest;
            at_line = false;
            continue;
        }
        let Some(ch) = rest.chars().next() else { break };
        rest = rest.get(ch.len_utf8()..).unwrap_or("");
        at_line = ch == '\n';
    }
    push_text(&mut out, bytes, text_from, rest);
    out
}

fn push_text<'a>(out: &mut Vec<Piece<'a>>, markdown: &'a str, from: &'a str, until: &str) {
    let end = markdown.len().saturating_sub(until.len());
    let start = markdown.len().saturating_sub(from.len());
    if let Some(text) = markdown.get(start..end)
        && !text.is_empty()
    {
        out.push(Piece::Text(text));
    }
}

fn citation_at(rest: &str) -> Option<(Lookup<'_>, &str, &str)> {
    if let Some(after) = rest.strip_prefix("[c") {
        let digits = after.chars().take_while(|ch| ch.is_ascii_digit()).count();
        if digits > 0 && after.get(digits..).is_some_and(|tail| tail.starts_with(']')) {
            let id_end = 2usize.saturating_add(digits);
            let end = id_end.saturating_add(1);
            return Some((Lookup::Id(rest.get(1..id_end)?), rest.get(..end)?, rest.get(end..)?));
        }
    }
    let after = rest.strip_prefix("[Page ")?;
    let digits = after.chars().take_while(|ch| ch.is_ascii_digit()).count();
    if digits == 0 || !after.get(digits..).is_some_and(|tail| tail.starts_with(']')) {
        return None;
    }
    let page = after.get(..digits)?.parse::<usize>().ok()?;
    let end = "[Page ".len().saturating_add(digits).saturating_add(1);
    Some((Lookup::Page(page), rest.get(..end)?, rest.get(end..)?))
}

/// Draw parsed CommonMark. Links keep their destinations; they are not rewritten.
pub(crate) fn show(ui: &mut egui::Ui, cache: &mut Cache, markdown: &str) {
    limit_width(ui);
    let prepared = with_tex_delimiters(markdown);
    for segment in segments(&prepared) {
        match segment {
            Seg::Prose(text) if !text.trim().is_empty() => show_prose(ui, cache, text),
            Seg::Table(table) => render_table(ui, cache, &table),
            Seg::Prose(_) => {}
        }
    }
}

fn limit_width(ui: &mut egui::Ui) {
    let width = ui.available_width();
    if width.is_finite() && width > 1.0 {
        ui.set_max_width(width);
    }
}

fn show_prose(ui: &mut egui::Ui, cache: &mut Cache, markdown: &str) {
    let math_fn = |ui: &mut egui::Ui, tex: &str, inline: bool| {
        math::show(ui, tex, !inline);
    };
    CommonMarkViewer::new().explicit_image_uri_scheme(true).max_image_width(Some(0)).render_math_fn(Some(&math_fn)).show(
        ui,
        &mut cache.inner,
        markdown,
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

const COL_GAP: f32 = 12.0;

#[derive(Debug)]
struct MdTable<'a> {
    aligns: Vec<Alignment>,
    header: Vec<&'a str>,
    rows: Vec<Vec<&'a str>>,
}

#[derive(Debug)]
enum Seg<'a> {
    Prose(&'a str),
    Table(MdTable<'a>),
}

fn parser_opts() -> Options {
    Options::ENABLE_TABLES
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_FOOTNOTES
        | Options::ENABLE_DEFINITION_LIST
        | Options::ENABLE_MATH
}

// `collect_table` keeps reading this same iterator, so this cannot be a `for` loop.
#[allow(clippy::while_let_on_iterator)]
fn segments(markdown: &str) -> Vec<Seg<'_>> {
    let mut events = Parser::new_ext(markdown, parser_opts()).into_offset_iter();
    let mut out = Vec::new();
    let mut prose_from = 0usize;
    while let Some((event, range)) = events.next() {
        let Event::Start(Tag::Table(aligns)) = event else { continue };
        push_prose(&mut out, markdown, prose_from, range.start);
        let (table, end) = collect_table(markdown, aligns, &mut events);
        prose_from = end;
        if !table.header.is_empty() || !table.rows.is_empty() {
            out.push(Seg::Table(table));
        }
    }
    push_prose(&mut out, markdown, prose_from, markdown.len());
    out
}

fn push_prose<'a>(out: &mut Vec<Seg<'a>>, markdown: &'a str, from: usize, to: usize) {
    if let Some(text) = markdown.get(from..to)
        && !text.is_empty()
    {
        out.push(Seg::Prose(text));
    }
}

// Same iterator as `segments`: a `for` loop would hold it for the whole table.
#[allow(clippy::while_let_on_iterator)]
fn collect_table<'a>(
    markdown: &'a str,
    aligns: Vec<Alignment>,
    events: &mut impl Iterator<Item = (Event<'a>, std::ops::Range<usize>)>,
) -> (MdTable<'a>, usize) {
    let mut header = Vec::new();
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut in_header = false;
    let mut end = markdown.len();
    while let Some((event, range)) = events.next() {
        end = range.end;
        match event {
            Event::Start(Tag::TableHead) => in_header = true,
            Event::End(TagEnd::TableHead) => {
                // pulldown-cmark emits the header as TableHead, not as a TableRow inside it.
                in_header = false;
                if header.is_empty() {
                    header = std::mem::take(&mut row);
                }
            }
            Event::Start(Tag::TableCell) => row.push(markdown.get(range).unwrap_or("").trim()),
            Event::End(TagEnd::TableRow) => {
                let finished = std::mem::take(&mut row);
                if in_header {
                    header = finished;
                } else if finished.iter().any(|cell| !cell.is_empty()) {
                    rows.push(finished);
                }
            }
            Event::End(TagEnd::Table) => break,
            _ => {}
        }
    }
    while rows.last().is_some_and(|cells| cells.iter().all(|cell| cell.is_empty())) {
        rows.pop();
    }
    (MdTable { aligns, header, rows }, end)
}

fn render_table(ui: &mut egui::Ui, cache: &mut Cache, table: &MdTable<'_>) {
    let cols = table.header.len().max(table.rows.iter().map(Vec::len).max().unwrap_or(0));
    if cols == 0 {
        return;
    }
    let stroke = ui.visuals().widgets.noninteractive.bg_stroke;
    egui::Frame::new().stroke(stroke).inner_margin(egui::Margin::symmetric(8, 6)).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 0.0;
        let inner = ui.available_width();
        let available = if inner.is_finite() { (inner - 8.0).max(1.0) } else { 480.0 };
        let widths = column_widths(ui, table, cols, available);
        let header = paint_row(ui, cache, &table.header, &widths, &table.aligns, true);
        ui.painter().hline(header.x_range(), header.bottom(), stroke);
        for (index, row) in table.rows.iter().enumerate() {
            paint_row(ui, cache, row, &widths, &table.aligns, index % 2 == 1);
        }
    });
    ui.add_space(8.0);
}

fn paint_row(ui: &mut egui::Ui, cache: &mut Cache, row: &[&str], widths: &[f32], aligns: &[Alignment], fill: bool) -> egui::Rect {
    let mut frame = egui::Frame::new().inner_margin(egui::Margin::symmetric(4, 3));
    if fill {
        frame = frame.fill(ui.visuals().faint_bg_color);
    }
    frame
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.x = COL_GAP;
            ui.horizontal(|ui| {
                for (col, width) in widths.iter().copied().enumerate() {
                    let text = row.get(col).copied().unwrap_or("");
                    let align = aligns.get(col).copied().unwrap_or(Alignment::None);
                    ui.vertical(|ui| show_cell(ui, cache, text, width, align));
                }
            });
        })
        .response
        .rect
}

fn show_cell(ui: &mut egui::Ui, cache: &mut Cache, text: &str, width: f32, align: Alignment) {
    if !text.is_empty() && !cell_is_plain(text) {
        ui.set_width(width);
        show_prose(ui, cache, text);
        return;
    }
    let font = egui::TextStyle::Body.resolve(ui.style());
    let color = ui.visuals().text_color();
    let shown = if text.is_empty() { " " } else { text };
    let galley = ui.fonts_mut(|fonts| fonts.layout(shown.to_owned(), font, color, width));
    let extra = (width - galley.size().x).max(0.0);
    let shift = match align {
        Alignment::Right => extra,
        Alignment::Center => extra * 0.5,
        Alignment::Left | Alignment::None => 0.0,
    };
    let pos = ui.cursor().left_top() + egui::vec2(shift, 0.0);
    ui.painter().galley(pos, galley.clone(), color);
    ui.allocate_exact_size(egui::vec2(width, galley.size().y.max(1.0)), egui::Sense::hover());
}

fn cell_is_plain(text: &str) -> bool {
    !text.contains(['*', '`', '$', '[', '\\', '_', '<'])
}

fn column_widths(ui: &egui::Ui, table: &MdTable<'_>, cols: usize, available: f32) -> Vec<f32> {
    let font = egui::TextStyle::Body.resolve(ui.style());
    let mut widths = vec![32.0f32; cols];
    let mut measure = |text: &str, col: usize| {
        let Some(slot) = widths.get_mut(col) else { return };
        let width = ui.fonts_mut(|fonts| fonts.layout_no_wrap(text.to_owned(), font.clone(), egui::Color32::WHITE).size().x) + 8.0;
        *slot = slot.max(width);
    };
    for (col, cell) in table.header.iter().enumerate() {
        measure(cell, col);
    }
    for row in &table.rows {
        for (col, cell) in row.iter().enumerate() {
            measure(cell, col);
        }
    }
    let gaps = COL_GAP * cols.saturating_sub(1) as f32;
    let content: f32 = widths.iter().sum();
    if content + gaps < available {
        let extra = available - gaps - content;
        let mut grow: Vec<usize> = (0..cols).filter(|col| !matches!(table.aligns.get(*col), Some(Alignment::Right | Alignment::Center))).collect();
        if grow.is_empty() {
            grow.push(0);
        }
        let share = extra / grow.len() as f32;
        for col in grow {
            if let Some(slot) = widths.get_mut(col) {
                *slot += share;
            }
        }
    } else if content + gaps > available {
        let budget = (available - gaps).max(cols as f32 * 32.0);
        let scale = budget / content.max(1.0);
        for slot in &mut widths {
            *slot = (*slot * scale).max(32.0);
        }
        let sum = widths.iter().sum::<f32>() + gaps;
        if sum > available
            && let Some(slot) = widths.iter_mut().max_by(|a, b| a.total_cmp(b))
        {
            *slot = (*slot - (sum - available)).max(24.0);
        }
    }
    widths
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
    fn citation_markers_split_outside_code() {
        let parts = super::pieces("See [c1] and [Page 2]. `not [c9]`\n");
        assert!(parts.iter().any(|part| matches!(part, super::Piece::Cite { lookup: super::Lookup::Id("c1"), .. })), "{parts:?}");
        assert!(parts.iter().any(|part| matches!(part, super::Piece::Cite { lookup: super::Lookup::Page(2), .. })), "{parts:?}");
        let code = parts.iter().find_map(|part| match part {
            super::Piece::Text(text) if text.contains("[c9]") => Some(*text),
            _ => None,
        });
        assert!(code.is_some(), "{parts:?}");
    }

    #[test]
    fn pipe_table_cells_keep_their_text_and_drop_a_blank_row() {
        let md = "| Load | Value |\n| --- | ---: |\n| Slab | 4 |\n| | |\n\nNext\n";
        let segs = super::segments(md);
        let super::Seg::Table(table) = &segs[0] else { panic!("{segs:?}") };
        assert_eq!(table.header, ["Load", "Value"]);
        assert_eq!(table.rows, [vec!["Slab", "4"]]);
        assert!(matches!(table.aligns.get(1), Some(pulldown_cmark::Alignment::Right)));
        assert!(segs.iter().any(|seg| matches!(seg, super::Seg::Prose(text) if text.contains("Next"))));
    }

    #[test]
    fn a_paragraph_after_a_table_starts_below_it() {
        let md = "| Load | Value |\n| --- | ---: |\n| Slab self-weight, 0.175(25) | 4 |\n| Total service load | 9.875 kN/m2 |\n\nThe ultimate design load is:\n";
        let ctx = egui::Context::default();
        let output = ctx.run_ui(egui::RawInput::default(), |ui| {
            ui.set_max_width(320.0);
            let mut cache = Cache::default();
            super::show(ui, &mut cache, md);
        });
        let mut last_bottom = 0.0f32;
        let mut paragraph_top = f32::MAX;
        let mut short_left = 0.0f32;
        let mut short_right = 0.0f32;
        let mut long_left = 0.0f32;
        let mut long_right = 0.0f32;
        for shape in &output.shapes {
            let egui::Shape::Text(text) = &shape.shape else { continue };
            let top = text.pos.y;
            let bottom = text.pos.y + text.galley.size().y;
            let left = text.pos.x;
            let right = text.pos.x + text.galley.size().x;
            let body = text.galley.job.text.as_str();
            if body.contains("Total service") || body.contains("9.875") {
                last_bottom = last_bottom.max(bottom);
            }
            if body.contains("9.875") {
                long_left = left;
                long_right = right;
            }
            if body == "4" {
                short_left = left;
                short_right = right;
            }
            if body.contains("ultimate design") {
                paragraph_top = paragraph_top.min(top);
            }
        }
        output.drop_without_applying_deltas();
        assert!(paragraph_top.is_finite(), "missing paragraph");
        assert!(paragraph_top + 1.0 >= last_bottom, "paragraph at {paragraph_top} overlaps the table bottom {last_bottom}");
        assert!(paragraph_top - last_bottom < 64.0, "gap of {} is the empty band under the table", paragraph_top - last_bottom);
        assert!((short_right - long_right).abs() < 8.0, "values are not right-aligned: {short_right} vs {long_right}");
        assert!(short_left > long_left + 8.0, "shorter value {short_left} should start after {long_left}");
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
