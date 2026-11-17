//! Real CLI -> automation table -> engine -> synthetic HTTP provider, without paid inference.
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::process::Command;

fn fixture() -> Vec<u8> {
    pdfcraft_testkit::Pdf::new()
        .obj("<< /Type /Catalog /Pages 2 0 R >>")
        .obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
        .obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 300] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>")
        .stream("", b"BT /F1 12 Tf 20 150 Td (Synthetic report: revenue is ten.) Tj ET")
        .obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        .build()
}

#[test]
fn ai_cli_round_trip_and_consent_gate() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("source.pdf"), fixture()).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let make_command = || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_pdfcraft-cli"));
        cmd.current_dir(dir.path())
            .args(["run", "--script", "steps.json", "--root", ".", "--report"])
            .env("PDFCRAFT_AI_ENABLED", "1")
            .env("PDFCRAFT_AI_BASE_URL", &url)
            .env("PDFCRAFT_AI_MODEL", "test-model")
            .env("PDFCRAFT_AI_API_KEY", "synthetic-api-key")
            .env_remove("PDFCRAFT_AI_LEGACY_TOKENS");
        cmd
    };
    let steps = |allow_send| {
        json!([
            {"tool":"doc_open","args":{"path":"source.pdf"}},
            {"tool":"ai_preview","args":{"doc":1}},
            {"tool":"ai_ask","args":{"doc":1,"input":"What is revenue?","allow_send":allow_send}},
            {"tool":"ai_summary","args":{"doc":1,"allow_send":allow_send}},
            {"tool":"ai_translate","args":{"doc":1,"input":"French","allow_send":allow_send}},
            {"tool":"ai_provider_test","args":{"allow_send":allow_send}}
        ])
    };
    std::fs::write(dir.path().join("steps.json"), steps(false).to_string()).unwrap();
    let denied = make_command().output().unwrap();
    assert!(!denied.status.success());
    assert!(String::from_utf8_lossy(&denied.stdout).contains("allow_send"));
    std::fs::write(dir.path().join("steps.json"), steps(true).to_string()).unwrap();
    let disabled = make_command().env("PDFCRAFT_AI_ENABLED", "0").output().unwrap();
    assert!(!disabled.status.success());
    assert!(String::from_utf8_lossy(&disabled.stdout).contains("AI assistant is off"));
    listener.set_nonblocking(true).unwrap();
    assert!(listener.accept().is_err(), "no request before consent");
    listener.set_nonblocking(false).unwrap();
    let worker = std::thread::spawn(move || {
        let mut requests = Vec::new();
        for index in 0..4 {
            let (mut socket, _) = listener.accept().unwrap();
            socket.set_read_timeout(Some(std::time::Duration::from_secs(15))).unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut headers = String::new();
            let mut len = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                    len = value.trim().parse::<usize>().unwrap();
                }
                headers.push_str(&line);
            }
            let mut body = vec![0; len];
            reader.read_exact(&mut body).unwrap();
            let response = if index == 3 {
                r#"{"data":[{"id":"test-model"}]}"#
            } else {
                r#"{"choices":[{"finish_reason":"stop","message":{"content":"Revenue is ten [Page 1]."}}]}"#
            };
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response.len(),
                response
            )
            .unwrap();
            requests.push((headers, String::from_utf8(body).unwrap()));
        }
        requests
    });
    std::fs::write(dir.path().join("steps.json"), steps(true).to_string()).unwrap();
    let output = make_command().output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let requests = worker.join().unwrap();
    assert!(requests.iter().all(|(headers, _)| headers.to_lowercase().contains("authorization: bearer synthetic-api-key")));
    assert!(requests[0].1.contains("Synthetic report") && requests[0].1.contains("What is revenue?"));
    assert!(requests[1].1.contains("Summarize"));
    assert!(requests[2].1.contains("French"));
    assert!(requests[3].0.starts_with("GET /v1/models ") && requests[3].1.is_empty());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Revenue is ten [Page 1]."));
    assert!(!stdout.contains("synthetic-api-key"));
    assert_eq!(std::fs::read(dir.path().join("source.pdf")).unwrap(), fixture(), "read-only assistance");
}
