//! Keyword retrieval over the page index. Current-page and selection spans rank first and do
//! not hide hits on other pages. Nothing is sent to a provider here.

use std::sync::Arc;

use super::index::{OutlineEntry, PageIndex, Span, SpanKind, page_index, structure_marks};
use super::{MAX_TEXT_BYTES, Task};
use crate::export::ExportSource;

/// A passage the user pointed at, in addition to the question.
#[derive(Clone, Debug, Default)]
pub struct Focus {
    /// `(page, glyph start, glyph end)` ranges, end exclusive.
    pub ranges: Vec<(usize, usize, usize)>,
    /// A view-space rectangle on `region_page`, for a drawn region.
    pub region_page: Option<usize>,
    pub region: Option<[f32; 4]>,
    pub text: String,
}

/// What to gather before a request.
#[derive(Clone, Debug)]
pub struct EvidenceRequest {
    pub task: Task,
    pub question: String,
    pub pages: Vec<usize>,
    pub current_page: Option<usize>,
    pub focus: Focus,
}

/// The passages that may be shown to the model, already under the text cap.
#[derive(Clone, Debug)]
pub struct EvidencePack {
    pub text: String,
    pub pages: Vec<usize>,
    pub spans: Vec<Span>,
    pub activities: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Clone)]
struct Scored {
    span: Span,
    score: i32,
}

const STOP: &[&str] = &[
    "about", "all", "and", "are", "but", "can", "does", "for", "from", "has", "have", "how", "into", "its", "not", "page", "that", "the", "this",
    "was", "what", "why", "with", "you", "your",
];

/// Gather evidence from `source`. `pages` empty means the first [`super::MAX_PAGES`] pages.
pub(crate) fn gather(source: &ExportSource, outline: &[OutlineEntry], request: &EvidenceRequest, page_count: usize) -> Result<EvidencePack, String> {
    let pages = limited_pages(&request.pages, page_count)?;
    let marks = structure_marks(source);
    let mut activities = vec!["Searching the document".to_string()];
    let mut notes = Vec::new();
    let mut indexed = Vec::new();
    for page in &pages {
        match page_index(source, *page, outline, &marks) {
            Ok(index) if index.truncated => {
                notes.push(format!("Page {} was incomplete and was left out.", page + 1));
            }
            Ok(index) => {
                if !index.spans.is_empty() {
                    activities.push(format!("Reading page {}", page + 1));
                }
                indexed.push(index);
            }
            Err(err) => return Err(err),
        }
    }
    if indexed.iter().all(|page| page.spans.is_empty()) {
        return Err("No extractable text on these pages. Run OCR first for scanned documents.".into());
    }
    let terms = terms(&request.question);
    let mut scored = Vec::new();
    for page in &indexed {
        for span in &page.spans {
            let score = score(span, &terms, request);
            if score > 0 {
                scored.push(Scored { span: span.clone(), score });
            }
        }
    }
    force_focus(&indexed, request, &mut scored);
    if request.task == Task::Summary {
        force_summary(&indexed, request, &mut scored);
    } else if scored.iter().all(|hit| request.current_page != Some(hit.span.page)) {
        // A question with no hit on the open page still carries a little of that page.
        if let Some(page) = request.current_page {
            push_page_sample(&indexed, page, 2, 40, &mut scored);
        }
    }
    add_neighbors(&indexed, &mut scored);
    scored.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.span.page.cmp(&b.span.page)).then_with(|| a.span.start.cmp(&b.span.start)));
    let pages_hit = distinct_pages(&scored);
    if pages_hit > 1 {
        activities.push("Comparing sections".into());
    }
    if scored.iter().any(|hit| hit.span.kind == SpanKind::Figure) {
        activities.push("Inspecting figure".into());
    }
    let (spans, text) = pack(&scored, request, &notes)?;
    let mut pages_out = spans.iter().map(|span| span.page + 1).collect::<Vec<_>>();
    pages_out.sort_unstable();
    pages_out.dedup();
    Ok(EvidencePack { text, pages: pages_out, spans, activities, notes })
}

fn limited_pages(pages: &[usize], count: usize) -> Result<Vec<usize>, String> {
    if count == 0 {
        return Err("This document has no pages.".into());
    }
    let list = if pages.is_empty() {
        (0..count.min(super::MAX_PAGES)).collect::<Vec<_>>()
    } else {
        if pages.len() > super::MAX_PAGES || pages.iter().any(|page| *page >= count) {
            return Err(format!("Choose 1–{} existing pages.", super::MAX_PAGES));
        }
        let mut pages = pages.to_vec();
        pages.sort_unstable();
        pages.dedup();
        pages
    };
    Ok(list)
}

fn terms(question: &str) -> Vec<String> {
    let mut out = Vec::new();
    for word in question.split(|ch: char| !ch.is_alphanumeric()) {
        if word.chars().count() < 3 || STOP.binary_search(&word.to_ascii_lowercase().as_str()).is_ok() {
            continue;
        }
        let folded = pdfcraft_render::text::fold_case(word);
        if !folded.is_empty() && !out.contains(&folded) {
            out.push(folded);
        }
    }
    out
}

fn score(span: &Span, terms: &[String], request: &EvidenceRequest) -> i32 {
    let folded = pdfcraft_render::text::fold_case(&span.text);
    let mut score = 0;
    for term in terms {
        if folded.contains(term) {
            score += 10;
        }
    }
    let phrase = pdfcraft_render::text::fold_case(request.question.trim());
    if phrase.chars().count() >= 3 && folded.contains(&phrase) {
        score += 30;
    }
    if request.current_page == Some(span.page) && score > 0 {
        score += 25;
    }
    if focus_covers(span, &request.focus) {
        score += 1_000;
    }
    if request.task == Task::Summary && span.kind == SpanKind::Heading {
        score += 15;
    }
    score
}

fn focus_covers(span: &Span, focus: &Focus) -> bool {
    focus.ranges.iter().any(|(page, start, end)| *page == span.page && *start < span.end && *end > span.start)
        || focus.region_page == Some(span.page) && focus.region.is_some_and(|rect| rects_overlap(rect, span.rect))
}

fn rects_overlap(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3]
}

fn force_focus(pages: &[Arc<PageIndex>], request: &EvidenceRequest, scored: &mut Vec<Scored>) {
    for page in pages {
        for span in &page.spans {
            if focus_covers(span, &request.focus) && scored.iter().all(|hit| hit.span.id != span.id) {
                scored.push(Scored { span: span.clone(), score: 1_000 });
            }
        }
    }
}

fn force_summary(pages: &[Arc<PageIndex>], request: &EvidenceRequest, scored: &mut Vec<Scored>) {
    for page in pages {
        if let Some(span) = page.spans.first()
            && scored.iter().all(|hit| hit.span.id != span.id)
        {
            scored.push(Scored { span: span.clone(), score: 20 });
        }
        for span in &page.spans {
            if span.kind == SpanKind::Heading && scored.iter().all(|hit| hit.span.id != span.id) {
                scored.push(Scored { span: span.clone(), score: 35 });
            }
        }
    }
    if let Some(page) = request.current_page {
        push_page_sample(pages, page, usize::MAX, 50, scored);
    }
}

fn push_page_sample(pages: &[Arc<PageIndex>], page: usize, limit: usize, score: i32, scored: &mut Vec<Scored>) {
    let Some(index) = pages.iter().find(|item| item.page == page) else { return };
    for span in index.spans.iter().take(limit) {
        if scored.iter().all(|hit| hit.span.id != span.id) {
            scored.push(Scored { span: span.clone(), score });
        }
    }
}

fn add_neighbors(pages: &[Arc<PageIndex>], scored: &mut Vec<Scored>) {
    let seeds: Vec<(usize, usize)> = scored.iter().map(|hit| (hit.span.page, hit.span.start)).collect();
    for (page, start) in seeds {
        let Some(index) = pages.iter().find(|item| item.page == page) else { continue };
        let Some(at) = index.spans.iter().position(|span| span.start == start) else { continue };
        for neighbor in at.saturating_sub(1)..=at.saturating_add(1) {
            let Some(span) = index.spans.get(neighbor) else { continue };
            if scored.iter().all(|hit| hit.span.id != span.id) {
                scored.push(Scored { span: span.clone(), score: 5 });
            }
        }
    }
}

fn distinct_pages(scored: &[Scored]) -> usize {
    let mut pages: Vec<usize> = scored.iter().map(|hit| hit.span.page).collect();
    pages.sort_unstable();
    pages.dedup();
    pages.len()
}

fn pack(scored: &[Scored], request: &EvidenceRequest, notes: &[String]) -> Result<(Vec<Span>, String), String> {
    // Current page and the selection come first. Other pages keep the rest of the budget so a
    // long open page cannot crowd them out.
    let (mut primary, mut other) = (Vec::new(), Vec::new());
    for hit in scored {
        if request.current_page == Some(hit.span.page) || focus_covers(&hit.span, &request.focus) {
            primary.push(hit);
        } else {
            other.push(hit);
        }
    }
    let mut chosen = Vec::new();
    let mut text = String::new();
    for note in notes {
        push_piece(&mut text, note)?;
    }
    if !request.focus.text.trim().is_empty() {
        let page = request.focus.ranges.first().map(|r| r.0).or(request.focus.region_page).map(|p| p + 1).unwrap_or(0);
        let block = if page == 0 {
            format!("Selected text:\n{}", request.focus.text.trim())
        } else {
            format!("Selected text on page {page}:\n{}", request.focus.text.trim())
        };
        push_piece(&mut text, &block)?;
    }
    let primary_budget = MAX_TEXT_BYTES * 7 / 10;
    append_hits(&primary, &mut chosen, &mut text, primary_budget)?;
    append_hits(&other, &mut chosen, &mut text, MAX_TEXT_BYTES)?;
    if chosen.is_empty() && request.focus.text.trim().is_empty() {
        return Err("No extractable text on these pages. Run OCR first for scanned documents.".into());
    }
    Ok((chosen, text))
}

fn append_hits(hits: &[&Scored], chosen: &mut Vec<Span>, text: &mut String, budget: usize) -> Result<(), String> {
    for hit in hits {
        if chosen.iter().any(|span| span.id == hit.span.id) {
            continue;
        }
        if text.len() >= budget && !chosen.is_empty() {
            break;
        }
        let block = format!("[{} page {} {}]\n{}", hit.span.id, hit.span.page + 1, hit.span.kind.as_str(), hit.span.text.trim());
        if text.len().saturating_add(block.len()).saturating_add(2) > MAX_TEXT_BYTES {
            if chosen.is_empty() && request_too_big(&block) {
                return Err(format!("Selected text exceeds {MAX_TEXT_BYTES} bytes. Choose fewer pages; nothing was sent."));
            }
            break;
        }
        push_piece(text, &block)?;
        chosen.push(hit.span.clone());
    }
    Ok(())
}

fn request_too_big(block: &str) -> bool {
    block.len() > MAX_TEXT_BYTES
}

fn push_piece(text: &mut String, piece: &str) -> Result<(), String> {
    let extra = if text.is_empty() { 0 } else { 2 };
    if text.len().saturating_add(piece.len()).saturating_add(extra) > MAX_TEXT_BYTES {
        return Err(format!("Selected text exceeds {MAX_TEXT_BYTES} bytes. Choose fewer pages; nothing was sent."));
    }
    if extra == 2 {
        text.push_str("\n\n");
    }
    text.push_str(piece);
    Ok(())
}

/// Search span text. No network.
pub fn search_spans(pages: &[Arc<PageIndex>], query: &str) -> Vec<Span> {
    let request = EvidenceRequest { task: Task::Ask, question: query.to_string(), pages: Vec::new(), current_page: None, focus: Focus::default() };
    let terms = terms(query);
    let mut scored = Vec::new();
    for page in pages {
        for span in &page.spans {
            let score = score(span, &terms, &request);
            if score > 0 {
                scored.push(Scored { span: span.clone(), score });
            }
        }
    }
    scored.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.span.page.cmp(&b.span.page)).then_with(|| a.span.start.cmp(&b.span.start)));
    scored.into_iter().map(|hit| hit.span).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;

    fn doc(bytes: Vec<u8>) -> (crate::Session, crate::DocId) {
        let mut session = crate::Session::new();
        let id = session.open("Retrieve", None, std::sync::Arc::new(bytes), None).unwrap();
        (session, id)
    }

    fn two_pages() -> Vec<u8> {
        let page = |contents: &str| {
            format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 300] /Resources << /Font << /F1 3 0 R >> >> /Contents {contents} >>")
        };
        let a = "BT /F1 12 Tf 20 200 Td (Revenue on the open page.) Tj ET";
        let b = "BT /F1 12 Tf 20 200 Td (Revenue in the later section.) Tj ET";
        let mut objs = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [4 0 R 6 0 R] /Count 2 >>".to_string(),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
            page("5 0 R"),
            format!("<< /Length {} >>\nstream\n{a}\nendstream", a.len()),
            page("7 0 R"),
            format!("<< /Length {} >>\nstream\n{b}\nendstream", b.len()),
        ];
        let _ = &mut objs;
        let mut out = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        for (i, object) in objs.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", i + 1).as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
        for offset in offsets {
            out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).as_bytes());
        out
    }

    fn pack_of(document: &Document, request: EvidenceRequest) -> EvidencePack {
        let outline = crate::ai::index::flatten_outline(&document.info.outline);
        let mut source = document.export_source();
        source.config.system_fonts = false;
        gather(&source, &outline, &request, document.info.pages.len()).unwrap()
    }

    #[test]
    fn retrieval_ranks_the_current_page_and_keeps_the_other_page() {
        let (session, id) = doc(two_pages());
        let document = session.get(id).unwrap();
        let pack = pack_of(
            document,
            EvidenceRequest {
                task: Task::Ask,
                question: "Where is revenue discussed?".into(),
                pages: Vec::new(),
                current_page: Some(0),
                focus: Focus::default(),
            },
        );
        assert!(pack.pages.contains(&1) && pack.pages.contains(&2), "{:?}", pack.pages);
        let first = pack.spans.iter().position(|span| span.text.contains("open page")).unwrap();
        let second = pack.spans.iter().position(|span| span.text.contains("later section")).unwrap();
        assert!(first < second, "the open page ranks before the other hit");
        assert!(pack.text.contains("Revenue on the open page"));
        assert!(pack.text.contains("Revenue in the later section"));
        assert!(pack.text.len() <= MAX_TEXT_BYTES);
    }

    #[test]
    fn a_selection_brings_its_neighbor_and_not_the_whole_file() {
        let body = "BT /F1 12 Tf 20 320 Td (Alpha definition stays here.) Tj 0 -80 Td (Beta consequence follows.) Tj 0 -80 Td (Unrelated inventory count.) Tj ET";
        let pdf = pdfcraft_testkit::Pdf::new()
            .obj("<< /Type /Catalog /Pages 2 0 R >>")
            .obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
            .obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 400] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>")
            .stream("", body.as_bytes())
            .obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
            .build();
        let (session, id) = doc(pdf);
        let document = session.get(id).unwrap();
        let page = document.ai_page(0).unwrap();
        assert!(page.spans.len() >= 2, "{:?}", page.spans.iter().map(|s| s.text.clone()).collect::<Vec<_>>());
        let selected = &page.spans[0];
        let pack = pack_of(
            document,
            EvidenceRequest {
                task: Task::Ask,
                question: "Why is this so?".into(),
                pages: vec![0],
                current_page: Some(0),
                focus: Focus { ranges: vec![(0, selected.start, selected.end)], text: selected.text.clone(), ..Focus::default() },
            },
        );
        assert!(pack.text.contains("Alpha definition"));
        assert!(pack.text.contains("Beta consequence"), "the following paragraph is context");
        assert!(!pack.text.contains("Unrelated inventory"), "a span beyond the neighbor stays out: {}", pack.text);
        assert!(pack.text.contains(&selected.id));
        assert!(pack.text.len() < MAX_TEXT_BYTES);
    }
}
