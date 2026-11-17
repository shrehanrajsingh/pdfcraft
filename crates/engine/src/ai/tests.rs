use super::*;

#[test]
fn ai_is_off_by_default_and_rejects_unsafe_endpoints() {
    assert!(Config::default().validate().is_err());
    assert!(Config::default().validate_connection().is_err());
    assert!(!Config::default().should_persist());
    for base_url in [
        "http://example.com/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?key=secret",
        "https://example.com/v1#secret",
        "file:///tmp/model",
        "http://localhost.example.com/v1",
        "https://example.com/v1/chat/completions",
    ] {
        assert!(Config { base_url: base_url.into(), ..Default::default() }.endpoint().is_err(), "{base_url}");
    }
    for base_url in ["https://api.openai.com/v1", "http://localhost:1234/v1", "http://127.0.0.1:8080/v1", "http://[::1]:8080/v1"] {
        assert!(Config { base_url: base_url.into(), ..Default::default() }.endpoint().is_ok(), "{base_url}");
    }
}

#[test]
fn ai_rejects_chatgpt_product_name_as_an_openai_model_id() {
    let openai = |model: &str| Config { enabled: true, model: model.into(), ..Default::default() };
    let err = openai("ChatGPT").require_model().unwrap_err();
    assert!(err.contains("gpt-4o-mini"), "{err}");
    assert!(openai("gpt-4o-mini").require_model().is_ok());
    assert!(openai("gpt-4o").require_model().is_ok());
    let local = Config { enabled: true, base_url: "http://localhost:11434/v1".into(), model: "ChatGPT".into(), ..Default::default() };
    assert!(local.require_model().is_ok(), "a local server may name a model anything");
}

#[test]
fn ai_prompts_are_bounded_and_use_source_page_numbers() {
    let mut session = crate::Session::new();
    let id = session.open("Synthetic", None, std::sync::Arc::new(crate::tests::fixture(2)), None).unwrap();
    let doc = session.get(id).unwrap();
    let prepared = doc.ai_source(&[1]).unwrap().prepare().unwrap();
    assert_eq!(prepared.pages, [2]);
    assert!(prepared.text.contains("[Page 2]"));
    let prompt = prepared.prompt(Task::Ask, "What page is this?").unwrap();
    assert!(prompt.contains("untrusted data"));
    assert!(prompt.contains("What page is this?"));
    assert!(prepared.prompt(Task::Ask, " ").is_err());
    assert!(prepared.prompt(Task::Translate, &"é".repeat(MAX_PROMPT_BYTES)).is_err());
    assert!(doc.ai_source(&[2]).is_err());
    assert!(doc.ai_source(&vec![0; MAX_PAGES + 1]).is_err());
    let huge = Prepared { text: "é".repeat(MAX_TEXT_BYTES), pages: vec![1] };
    assert!(huge.prompt(Task::Summary, "").is_err());
}

#[test]
fn ai_refuses_pending_redactions_and_empty_scans() {
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
    let mut session = crate::Session::new();
    let id = session.open("Pending redaction", None, std::sync::Arc::new(pdf(true)), None).unwrap();
    assert!(session.get(id).unwrap().ai_source(&[]).err().unwrap().contains("redaction"));
    let id = session.open("Empty scan", None, std::sync::Arc::new(pdf(false)), None).unwrap();
    assert!(session.get(id).unwrap().ai_source(&[]).unwrap().prepare().err().unwrap().contains("OCR"));
}

#[test]
fn ai_respects_document_copy_permissions() {
    let mut owner = crate::Session::new();
    let id = owner.open("Restricted", None, std::sync::Arc::new(crate::tests::fixture(1)), None).unwrap();
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
    let mut reader = crate::Session::new();
    let id = reader.open("Restricted", None, saved, None).unwrap();
    assert!(reader.get(id).unwrap().ai_source(&[]).err().unwrap().contains("copying"));
}

#[cfg(not(target_arch = "wasm32"))]
mod http {
    use super::*;
    use craft_ai::Provider;
    use std::io::{BufRead, BufReader, Read, Write};

    fn server(status: &str, body: &str, headers: &str) -> (Config, std::thread::JoinHandle<String>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let config = Config {
            enabled: true,
            base_url: format!("http://{}/v1", listener.local_addr().unwrap()),
            model: "test-model".into(),
            ..Default::default()
        };
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}",
            body.len()
        );
        let worker = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
            let mut reader = BufReader::new(socket.try_clone().unwrap());
            let mut request = String::new();
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse::<usize>().unwrap();
                }
                request.push_str(&line);
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            request.push_str("\r\n");
            request.push_str(&String::from_utf8(body).unwrap());
            // Oversized-response tests deliberately close the socket before the entire body.
            let _ = socket.write_all(response.as_bytes());
            request
        });
        (config, worker)
    }

    #[test]
    fn ai_http_round_trip_uses_bearer_model_and_page_context() {
        let (config, worker) = server("200 OK", r#"{"choices":[{"finish_reason":"stop","message":{"content":"Result [Page 3]"}}]}"#, "");
        let mut provider = OpenAiProvider::new(config, "synthetic-api-key".into()).unwrap();
        let prepared = Prepared { text: "[Page 3]\nSynthetic document".into(), pages: vec![3] };
        assert_eq!(prepared.complete(&mut provider, Task::Ask, "What is this?").unwrap(), "Result [Page 3]");
        let request = worker.join().unwrap();
        assert!(request.starts_with("POST /v1/chat/completions "));
        assert!(request.to_lowercase().contains("authorization: bearer synthetic-api-key"));
        let body: serde_json::Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["max_completion_tokens"], 2048);
        assert!(body.get("temperature").is_none());
        assert!(body["messages"][0]["content"].as_str().unwrap().contains("[Page 3]"));
    }

    #[test]
    fn ai_connection_test_never_sends_document_text_or_empty_auth() {
        let (config, worker) = server("200 OK", r#"{"data":[{"id":"test-model"}]}"#, "");
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        assert_eq!(provider.test().unwrap().models, ["test-model"]);
        let request = worker.join().unwrap();
        assert!(request.starts_with("GET /v1/models "));
        assert!(!request.to_lowercase().contains("authorization"));
    }

    #[test]
    fn ai_connection_test_does_not_require_a_model_id() {
        let (mut config, worker) = server("200 OK", r#"{"data":[{"id":"test-model"}]}"#, "");
        config.model.clear();
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        assert_eq!(provider.test().unwrap().models, ["test-model"]);
        worker.join().unwrap();
        let err = provider.complete("hello", &Default::default()).unwrap_err().to_string();
        assert!(err.contains("model"), "{err}");
    }

    #[test]
    fn ai_errors_never_echo_credentials_or_response_bodies() {
        for status in ["401 Unauthorized", "429 Too Many Requests", "500 Server Error", "307 Temporary Redirect"] {
            let (config, worker) = server(status, "synthetic-secret private document", "Location: http://127.0.0.1:1/stolen\r\n");
            let mut provider = OpenAiProvider::new(config, "synthetic-secret".into()).unwrap();
            let error = provider.complete("private document", &Default::default()).unwrap_err().to_string();
            assert!(error.contains(status.split_whitespace().next().unwrap()));
            assert!(!error.contains("synthetic-secret"));
            assert!(!error.contains("private document"));
            worker.join().unwrap();
        }
    }

    #[test]
    fn ai_rejects_malformed_oversized_empty_and_incomplete_answers() {
        for body in [
            "not JSON".into(),
            "x".repeat(1024 * 1024 + 1),
            r#"{"choices":[{"message":{"content":""}}]}"#.into(),
            r#"{"choices":[{"finish_reason":"length","message":{"content":"partial"}}]}"#.into(),
            r#"{"choices":[{"finish_reason":"tool_calls","message":{"content":"tool"}}]}"#.into(),
        ] {
            let (config, worker) = server("200 OK", &body, "");
            let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
            assert!(provider.complete("hello", &Default::default()).is_err());
            worker.join().unwrap();
        }
    }

    #[test]
    fn ai_legacy_tokens_and_reflected_keys_are_handled() {
        let (mut config, worker) = server("200 OK", r#"{"choices":[{"message":{"content":"synthetic-secret"}}]}"#, "");
        config.legacy_max_tokens = true;
        let mut provider = OpenAiProvider::new(config, "synthetic-secret".into()).unwrap();
        assert_eq!(provider.complete("hello", &Default::default()).unwrap(), "[API key removed]");
        let request = worker.join().unwrap();
        assert!(request.contains("\"max_tokens\""));
        assert!(!request.contains("max_completion_tokens"));
    }
}
