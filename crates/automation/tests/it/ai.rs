//! Preview and consent gates for the optional AI tools, without paid inference.
use std::path::PathBuf;

use pdfcraft_automation::{Automation, Content, ToolError};
use serde_json::{Value, json};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pdfcraft-ai-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fixture() -> Vec<u8> {
    let body = "BT /F1 12 Tf 20 150 Td (Synthetic report: revenue is ten.) Tj ET";
    let objs = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 300] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".to_string(),
        format!("<< /Length {} >>\nstream\n{body}\nendstream", body.len()),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
    ];
    let mut out = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1).as_bytes());
    for o in offsets {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1).as_bytes());
    out
}

fn json_out(result: Result<Vec<Content>, ToolError>) -> Result<Value, ToolError> {
    result.map(|mut c| match c.remove(0) {
        Content::Json(v) => v,
        other => panic!("expected JSON, got {other:?}"),
    })
}

#[test]
fn ai_preview_extracts_text_and_network_tools_need_explicit_consent() {
    let dir = tmp("preview");
    std::fs::write(dir.join("source.pdf"), fixture()).unwrap();
    let mut a = Automation::new().with_root(&dir).unwrap();
    let doc = json_out(a.call("doc_open", &json!({"path":"source.pdf"}))).unwrap()["doc"].as_u64().unwrap();
    let preview = json_out(a.call("ai_preview", &json!({"doc":doc}))).unwrap();
    assert_eq!(preview["pages"], json!([1]));
    assert!(preview["text"].as_str().unwrap().contains("Synthetic report"));
    let denied = a.call("ai_ask", &json!({"doc":doc,"input":"What is revenue?"})).unwrap_err().to_string();
    assert!(denied.contains("allow_send"), "{denied}");
    let off = a.call("ai_ask", &json!({"doc":doc,"input":"What is revenue?","allow_send":true})).unwrap_err().to_string();
    assert!(off.contains("AI assistant is off"), "{off}");
    let test_denied = a.call("ai_provider_test", &json!({})).unwrap_err().to_string();
    assert!(test_denied.contains("allow_send"), "{test_denied}");
    let chat_denied = a.call("ai_chat", &json!({"doc":doc,"messages":[{"role":"user","content":"Say more"}]})).unwrap_err().to_string();
    assert!(chat_denied.contains("allow_send"), "{chat_denied}");
    let chat_off = a
        .call("ai_chat", &json!({"doc":doc,"task":"summary","messages":[{"role":"user","content":"Say more"}],"allow_send":true}))
        .unwrap_err()
        .to_string();
    assert!(chat_off.contains("AI assistant is off"), "{chat_off}");
    std::fs::write(dir.join("not-an-image.txt"), b"hello").unwrap();
    let bad_image = a
        .call(
            "ai_chat",
            &json!({
                "doc": doc,
                "task": "ask",
                "input": "What is this?",
                "messages": [{"role":"user","content":"What colour?","images":["not-an-image.txt"]}],
                "allow_send": true
            }),
        )
        .unwrap_err()
        .to_string();
    assert!(bad_image.contains("image") || bad_image.contains("AI assistant is off"), "{bad_image}");
}

#[test]
fn ai_index_and_search_return_stable_span_boxes_without_a_provider() {
    let dir = tmp("index");
    std::fs::write(dir.join("source.pdf"), fixture()).unwrap();
    let mut a = Automation::new().with_root(&dir).unwrap();
    let doc = json_out(a.call("doc_open", &json!({"path":"source.pdf"}))).unwrap()["doc"].as_u64().unwrap();
    let index = json_out(a.call("ai_index", &json!({"doc":doc}))).unwrap();
    let spans = index["pages"][0]["spans"].as_array().unwrap();
    assert!(spans.iter().any(|span| span["text"].as_str().unwrap().contains("Synthetic report")), "{spans:?}");
    let id = spans[0]["id"].as_str().unwrap();
    assert!(id.starts_with("p1-g"), "{id}");
    assert!(!spans[0]["rects"].as_array().unwrap().is_empty());
    let again = json_out(a.call("ai_index", &json!({"doc":doc}))).unwrap();
    assert_eq!(again["pages"][0]["spans"][0]["id"], id);
    let hits = json_out(a.call("ai_search", &json!({"doc":doc,"query":"revenue"}))).unwrap();
    let found = hits["hits"].as_array().unwrap();
    assert!(found.iter().any(|hit| hit["text"].as_str().unwrap().to_ascii_lowercase().contains("revenue")), "{found:?}");
    assert!(!found[0]["rects"].as_array().unwrap().is_empty());
}
