//! Lazy page index for the assistant. Span ids name glyph ranges from [`pdfcraft_render::PageText`].
//! The model is only allowed to cite those ids; coordinates stay here.
//!
//! An id is `p{page}-g{start}-{end}`: `page` is the physical 1-based page, and `start..end` is the
//! glyph range on that page. The same extract of the same bytes yields the same ids. The cache is
//! keyed by the displayed bytes, so an edit drops it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use pdfcraft_render::{PageRenderer, PageText, RenderRequest, RenderWarning, RequestKind};

use super::MAX_PAGES;
use crate::Document;
use crate::export::ExportSource;

/// How a span was classified. Untagged text stays a paragraph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    Paragraph,
    Heading,
    Table,
    Figure,
}

impl SpanKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Paragraph => "paragraph",
            Self::Heading => "heading",
            Self::Table => "table",
            Self::Figure => "figure",
        }
    }
}

/// One reading-order block and the view-space boxes that draw it.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub id: String,
    /// Zero-based page.
    pub page: usize,
    /// Glyph range on that page, end exclusive.
    pub start: usize,
    pub end: usize,
    pub text: String,
    /// Union of the glyph boxes, page view space (points, origin top-left, y down).
    pub rect: [f32; 4],
    /// One box per line, same space. A very long block keeps the union only.
    pub rects: Vec<[f32; 4]>,
    pub kind: SpanKind,
}

/// Bookmark or extracted heading the agent can name.
#[derive(Clone, Debug, PartialEq)]
pub struct OutlineEntry {
    pub title: String,
    pub page: Option<usize>,
    pub depth: usize,
}

/// A structure tag reduced to the text that can be matched to a span.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StructMark {
    page: Option<usize>,
    kind: SpanKind,
    text: String,
}

/// Spans of one page.
#[derive(Clone, Debug, PartialEq)]
pub struct PageIndex {
    pub page: usize,
    pub spans: Vec<Span>,
    /// The renderer cut this page's content, so the spans were left out of answers.
    pub truncated: bool,
}

const MAX_SPANS: usize = 2_000;
const MAX_LINE_RECTS: usize = 48;
const MAX_OUTLINE: usize = 2_000;
const MAX_CACHED_DOCS: usize = 4;

struct CachedDoc {
    stamp: u64,
    len: usize,
    pages: HashMap<usize, Arc<PageIndex>>,
    marks: Vec<StructMark>,
    marks_ready: bool,
}

fn cache() -> &'static Mutex<Vec<CachedDoc>> {
    static CACHE: Mutex<Vec<CachedDoc>> = Mutex::new(Vec::new());
    &CACHE
}

fn lock_cache() -> std::sync::MutexGuard<'static, Vec<CachedDoc>> {
    cache().lock().unwrap_or_else(PoisonError::into_inner)
}

/// Identity of the displayed bytes. A new working file, including an edit, misses the cache.
fn stamp(bytes: &[u8]) -> u64 {
    let mut hash = bytes.len() as u64;
    let head = bytes.get(..64.min(bytes.len())).unwrap_or(&[]);
    for (i, byte) in head.iter().enumerate() {
        hash = hash.wrapping_mul(16777619).wrapping_add(u64::from(*byte)).wrapping_add(i as u64);
    }
    let tail_at = bytes.len().saturating_sub(64);
    if let Some(tail) = bytes.get(tail_at..) {
        for byte in tail {
            hash = hash.wrapping_mul(16777619).wrapping_add(u64::from(*byte));
        }
    }
    hash
}

fn cached_page(source: &ExportSource, page: usize) -> Option<Arc<PageIndex>> {
    let stamp = stamp(&source.bytes);
    lock_cache().iter().find(|doc| doc.stamp == stamp && doc.len == source.bytes.len()).and_then(|doc| doc.pages.get(&page).cloned())
}

fn store_page(source: &ExportSource, page: Arc<PageIndex>) {
    let stamp = stamp(&source.bytes);
    let len = source.bytes.len();
    let mut cache = lock_cache();
    if let Some(doc) = cache.iter_mut().find(|doc| doc.stamp == stamp && doc.len == len) {
        if doc.pages.len() < MAX_PAGES {
            doc.pages.insert(page.page, page);
        }
        return;
    }
    if cache.len() >= MAX_CACHED_DOCS {
        cache.remove(0);
    }
    let mut pages = HashMap::new();
    pages.insert(page.page, page);
    cache.push(CachedDoc { stamp, len, pages, marks: Vec::new(), marks_ready: false });
}

fn cached_marks(source: &ExportSource) -> Option<Vec<StructMark>> {
    let stamp = stamp(&source.bytes);
    lock_cache().iter().find(|doc| doc.stamp == stamp && doc.len == source.bytes.len() && doc.marks_ready).map(|doc| doc.marks.clone())
}

fn store_marks(source: &ExportSource, marks: Vec<StructMark>) {
    let stamp = stamp(&source.bytes);
    let len = source.bytes.len();
    let mut cache = lock_cache();
    if let Some(doc) = cache.iter_mut().find(|doc| doc.stamp == stamp && doc.len == len) {
        doc.marks = marks;
        doc.marks_ready = true;
        return;
    }
    if cache.len() >= MAX_CACHED_DOCS {
        cache.remove(0);
    }
    cache.push(CachedDoc { stamp, len, pages: HashMap::new(), marks, marks_ready: true });
}

impl Document {
    /// Permission check shared with [`Document::ai_source`](super::Document::ai_source).
    pub(crate) fn ai_access(&self) -> Result<(), String> {
        self.check_copying().map_err(|e| e.to_string())?;
        if self.redaction_marks() != 0 {
            return Err("Apply or remove pending redaction marks before using AI; marked text is still extractable.".into());
        }
        Ok(())
    }

    /// Outline entries in tree order, capped, from the document's bookmarks.
    pub fn ai_outline(&self) -> Result<Vec<OutlineEntry>, String> {
        self.ai_access()?;
        Ok(flatten_outline(&self.info.outline))
    }

    /// Spans for one zero-based page. Empty when the page has no text. A truncated page is
    /// returned with `truncated` set and no spans, so incomplete text is not cited.
    pub fn ai_page(&self, page: usize) -> Result<Arc<PageIndex>, String> {
        self.ai_access()?;
        if page >= self.info.pages.len() {
            return Err(format!("Page {} is outside this document.", page + 1));
        }
        let mut source = self.export_source();
        source.config.system_fonts = false;
        page_index(&source, page, &flatten_outline(&self.info.outline), &structure_marks(&source))
    }

    /// Pages to read for a request. Empty means the first [`MAX_PAGES`] pages.
    pub(crate) fn ai_page_list(&self, pages: &[usize]) -> Result<Vec<usize>, String> {
        self.ai_access()?;
        let count = self.info.pages.len();
        if count == 0 {
            return Err("This document has no pages.".into());
        }
        let list = if pages.is_empty() {
            (0..count.min(MAX_PAGES)).collect::<Vec<_>>()
        } else {
            if pages.len() > MAX_PAGES || pages.iter().any(|page| *page >= count) {
                return Err(format!("Choose 1–{MAX_PAGES} existing pages."));
            }
            let mut pages = pages.to_vec();
            pages.sort_unstable();
            pages.dedup();
            pages
        };
        Ok(list)
    }
}

/// Build one page's spans. Uses the cache when `source` was indexed before.
pub(crate) fn page_index(source: &ExportSource, page: usize, outline: &[OutlineEntry], marks: &[StructMark]) -> Result<Arc<PageIndex>, String> {
    if let Some(hit) = cached_page(source, page) {
        return Ok(hit);
    }
    let built = crate::guard(|| build_page(source, page, outline, marks)).map_err(|_| format!("Page {} could not be indexed safely.", page + 1))?;
    let built = built?;
    let arc = Arc::new(built);
    store_page(source, Arc::clone(&arc));
    Ok(arc)
}

pub(crate) fn structure_marks(source: &ExportSource) -> Vec<StructMark> {
    if let Some(marks) = cached_marks(source) {
        return marks;
    }
    let marks = read_marks(source);
    store_marks(source, marks.clone());
    marks
}

fn read_marks(source: &ExportSource) -> Vec<StructMark> {
    let Ok(doc) = pdfcraft_cos::Document::open_with_password(source.bytes.clone(), source.config.password.as_deref()) else {
        return Vec::new();
    };
    pdfcraft_a11y::tags(&doc)
        .into_iter()
        .filter_map(|tag| {
            let kind = match tag.kind.as_str() {
                "Table" | "THead" | "TBody" | "TFoot" | "TR" | "TH" | "TD" => SpanKind::Table,
                "Figure" => SpanKind::Figure,
                "H" | "H1" | "H2" | "H3" | "H4" | "H5" | "H6" | "Title" => SpanKind::Heading,
                _ => return None,
            };
            let text = tag.actual_text.or(tag.alt).or(tag.title).unwrap_or_default();
            let text = normalize(&text);
            (text.chars().count() >= 3).then_some(StructMark { page: tag.page, kind, text })
        })
        .take(MAX_OUTLINE)
        .collect()
}

fn build_page(source: &ExportSource, page: usize, outline: &[OutlineEntry], marks: &[StructMark]) -> Result<PageIndex, String> {
    if page >= source.pages {
        return Err(format!("Page {} is outside this document.", page + 1));
    }
    let mut renderer = PageRenderer::new(source.bytes.clone(), source.config.clone());
    let rendered = renderer.render(RenderRequest { page, kind: RequestKind::Text, ..Default::default() });
    if rendered.error.is_some() {
        return Err(format!("Page {} could not be extracted completely; nothing was sent.", page + 1));
    }
    if rendered.warnings.contains(&RenderWarning::ContentTruncated) {
        return Ok(PageIndex { page, spans: Vec::new(), truncated: true });
    }
    let Some(text) = rendered.text else {
        return Ok(PageIndex { page, spans: Vec::new(), truncated: false });
    };
    Ok(PageIndex { page, spans: spans_of(page, &text, outline, marks), truncated: false })
}

fn spans_of(page: usize, text: &PageText, outline: &[OutlineEntry], marks: &[StructMark]) -> Vec<Span> {
    let lines = lines_of(text);
    let mut spans = Vec::new();
    let mut index = 0;
    while index < lines.len() && spans.len() < MAX_SPANS {
        let start_line = index;
        index += 1;
        while index < lines.len() && !paragraph_break(&lines[index - 1], &lines[index]) {
            index += 1;
        }
        let Some(first) = lines.get(start_line) else { break };
        let Some(last) = lines.get(index - 1) else { break };
        let glyph_start = first.start;
        let glyph_end = last.end;
        let body = text.text_of(glyph_start..glyph_end);
        if body.trim().is_empty() {
            continue;
        }
        let mut rect = first.rect;
        let mut rects = Vec::new();
        for line in lines.get(start_line..index).unwrap_or(&[]) {
            rect = union(rect, line.rect);
            if rects.len() < MAX_LINE_RECTS {
                rects.push(line.rect);
            }
        }
        if lines.get(start_line..index).map(|s| s.len()).unwrap_or(0) > MAX_LINE_RECTS {
            rects.clear();
            rects.push(rect);
        }
        let mut span = Span {
            id: format!("p{}-g{glyph_start}-{glyph_end}", page + 1),
            page,
            start: glyph_start,
            end: glyph_end,
            text: body,
            rect,
            rects,
            kind: SpanKind::Paragraph,
        };
        classify(&mut span, outline, marks);
        spans.push(span);
    }
    spans
}

struct LineSeg {
    start: usize,
    end: usize,
    rect: [f32; 4],
}

fn lines_of(text: &PageText) -> Vec<LineSeg> {
    let mut lines = Vec::new();
    let mut index = 0;
    while index < text.glyphs.len() {
        let line_id = text.line_of.get(index).copied().unwrap_or(0);
        let start = index;
        let mut rect = text.glyphs.get(index).map(|glyph| glyph.rect).unwrap_or([0.0; 4]);
        index += 1;
        while index < text.glyphs.len() && text.line_of.get(index).copied() == Some(line_id) {
            if let Some(glyph) = text.glyphs.get(index) {
                rect = union(rect, glyph.rect);
            }
            index += 1;
        }
        if rect.iter().all(|v| v.is_finite()) {
            lines.push(LineSeg { start, end: index, rect });
        }
    }
    lines
}

fn paragraph_break(prev: &LineSeg, next: &LineSeg) -> bool {
    let height = (prev.rect[3] - prev.rect[1]).abs().max(4.0);
    let gap = next.rect[1] - prev.rect[3];
    if gap > height * 0.65 {
        return true;
    }
    let overlaps = next.rect[1] < prev.rect[3] && next.rect[3] > prev.rect[1];
    overlaps && next.rect[0] > prev.rect[2] + 8.0
}

fn classify(span: &mut Span, outline: &[OutlineEntry], marks: &[StructMark]) {
    let text = normalize(&span.text);
    for entry in outline {
        if entry.page == Some(span.page) && title_matches(&text, &normalize(&entry.title)) {
            span.kind = SpanKind::Heading;
            break;
        }
    }
    for mark in marks {
        if mark.page.is_some() && mark.page != Some(span.page) {
            continue;
        }
        if text_matches(&text, &mark.text) {
            span.kind = mark.kind;
        }
    }
}

fn title_matches(span: &str, title: &str) -> bool {
    title.chars().count() >= 3 && (span.starts_with(title) || title.starts_with(span))
}

fn text_matches(span: &str, mark: &str) -> bool {
    mark.chars().count() >= 8 && (span.contains(mark) || mark.contains(span))
}

fn normalize(text: &str) -> String {
    let mut out = String::new();
    let mut space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            space = !out.is_empty();
            continue;
        }
        if space {
            out.push(' ');
            space = false;
        }
        for lower in ch.to_lowercase() {
            out.push(lower);
        }
    }
    out
}

fn union(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
}

pub(crate) fn flatten_outline(items: &[pdfcraft_render::OutlineItem]) -> Vec<OutlineEntry> {
    let mut out = Vec::new();
    flatten(items, 0, &mut out);
    out
}

fn flatten(items: &[pdfcraft_render::OutlineItem], depth: usize, out: &mut Vec<OutlineEntry>) {
    for item in items {
        if out.len() >= MAX_OUTLINE {
            return;
        }
        out.push(OutlineEntry { title: item.title.clone(), page: item.page, depth });
        flatten(&item.children, depth.saturating_add(1), out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(bytes: Vec<u8>) -> (crate::Session, crate::DocId) {
        let mut session = crate::Session::new();
        let id = session.open("Indexed", None, Arc::new(bytes), None).unwrap();
        (session, id)
    }

    #[test]
    fn span_ids_are_stable_and_repeat_text_keeps_distinct_boxes() {
        let body = "BT /F1 12 Tf 20 220 Td (Same sentence.) Tj 0 -40 Td (Same sentence.) Tj ET";
        let pdf = pdfcraft_testkit::Pdf::new()
            .obj("<< /Type /Catalog /Pages 2 0 R >>")
            .obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
            .obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 300] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>")
            .stream("", body.as_bytes())
            .obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
            .build();
        let (session, id) = open(pdf);
        let doc = session.get(id).unwrap();
        let first = doc.ai_page(0).unwrap();
        let second = doc.ai_page(0).unwrap();
        assert_eq!(first.spans.len(), second.spans.len());
        assert!(first.spans.len() >= 2, "{:?}", first.spans.iter().map(|s| &s.text).collect::<Vec<_>>());
        let (a, b) = (&first.spans[0], &first.spans[1]);
        assert_eq!(a.text.trim(), b.text.trim());
        assert_ne!(a.id, b.id);
        assert_ne!(a.rect[1], b.rect[1]);
        assert_eq!(a.id, second.spans[0].id);
        assert_eq!(a.rects, second.spans[0].rects);
        assert!(a.id.starts_with("p1-g"));
        assert!(!a.rects.is_empty());
    }

    #[test]
    fn index_refuses_copy_blocks_and_pending_redactions() {
        let pdf = |annotation: bool| {
            pdfcraft_testkit::Pdf::new()
                .obj("<< /Type /Catalog /Pages 2 0 R >>")
                .obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
                .obj(if annotation {
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 300] /Annots [4 0 R] >>"
                } else {
                    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 300] >>"
                })
                .obj("<< /Type /Annot /Subtype /Redact /Rect [20 20 100 40] >>")
                .build()
        };
        let (session, id) = open(pdf(true));
        assert!(session.get(id).unwrap().ai_page(0).unwrap_err().contains("redaction"));
        let mut owner = crate::Session::new();
        let id = owner.open("Restricted", None, Arc::new(crate::tests::fixture(1)), None).unwrap();
        owner
            .apply(
                id,
                crate::Edit::Protect(crate::Protection {
                    permissions_password: Some("synthetic-owner-password".into()),
                    copy: false,
                    ..Default::default()
                }),
            )
            .unwrap();
        let saved = owner.save_bytes(id).unwrap();
        let (reader, id) = open(saved.as_ref().clone());
        assert!(reader.get(id).unwrap().ai_page(0).unwrap_err().contains("copying"));
    }
}
