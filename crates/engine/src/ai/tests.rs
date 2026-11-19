use super::*;

#[test]
fn ai_is_off_by_default_and_rejects_unsafe_endpoints() {
    assert!(Config::default().stream_responses);
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
    assert!(prompt.contains("own knowledge"));
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

    fn server_queue(bodies: Vec<String>) -> (Config, std::thread::JoinHandle<Vec<String>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut config = Config {
            enabled: true,
            base_url: format!("http://{}/v1", listener.local_addr().unwrap()),
            model: "test-model".into(),
            ..Default::default()
        };
        config.stream_responses = false;
        let worker = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for body in bodies {
                let (mut socket, _) = listener.accept().unwrap();
                socket.set_read_timeout(Some(std::time::Duration::from_secs(10))).unwrap();
                let mut reader = BufReader::new(socket.try_clone().unwrap());
                let mut request = String::new();
                let mut length = 0usize;
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
                let mut payload = vec![0; length];
                reader.read_exact(&mut payload).unwrap();
                request.push_str("\r\n");
                request.push_str(&String::from_utf8(payload).unwrap());
                let response =
                    format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                socket.write_all(response.as_bytes()).unwrap();
                requests.push(request);
            }
            requests
        });
        (config, worker)
    }

    fn chat_json(content: &str) -> String {
        serde_json::json!({"choices":[{"finish_reason":"stop","message":{"content": content}}]}).to_string()
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
        assert!(body.get("max_completion_tokens").is_none());
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["stream"], false);
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
            r#"{"choices":[{"message":{"content":""}}]}"#,
            r#"{"choices":[{"finish_reason":"length","message":{"content":""}}]}"#,
            r#"{"choices":[{"finish_reason":"tool_calls","message":{"content":"tool"}}]}"#,
        ] {
            let (config, worker) = server("200 OK", body, "");
            let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
            assert!(provider.complete("hello", &Default::default()).is_err());
            worker.join().unwrap();
        }
    }

    #[test]
    fn an_unreadable_body_is_shown_as_text() {
        let (config, worker) = server("200 OK", "Plain summary, not JSON.", "");
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        assert_eq!(provider.complete("hello", &Default::default()).unwrap(), "Plain summary, not JSON.");
        worker.join().unwrap();

        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Kept\"}}]}\n\n",
            "data: this line is not json\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" answer\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (mut config, worker) = server("200 OK", body, "");
        config.stream_responses = true;
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        let mut seen = Vec::new();
        let answer = provider
            .complete_chat_reporting(&[ChatMessage { role: "user".into(), text: "Hi".into(), images: Vec::new() }], StreamKind::Plain, &mut |text| {
                seen.push(text.to_string());
            })
            .unwrap();
        assert_eq!(answer, "Kept\nthis line is not json answer");
        assert!(seen.iter().any(|text| text.contains("this line is not json")), "{seen:?}");
        worker.join().unwrap();
    }

    #[test]
    fn ai_legacy_tokens_and_reflected_keys_are_handled() {
        let (mut config, worker) = server("200 OK", r#"{"choices":[{"message":{"content":"synthetic-secret"}}]}"#, "");
        config.legacy_max_tokens = true;
        let mut provider = OpenAiProvider::new(config, "synthetic-secret".into()).unwrap();
        assert_eq!(provider.complete("hello", &Default::default()).unwrap(), "[API key removed]");
        let request = worker.join().unwrap();
        assert!(!request.contains("\"max_tokens\""));
        assert!(!request.contains("max_completion_tokens"));
    }

    #[test]
    fn ai_conversation_is_bounded_and_keeps_page_context() {
        let prepared = Prepared { text: "[Page 3]\nSynthetic document".into(), pages: vec![3] };
        let first = prepared.conversation_messages(Task::Summary, "", &[], &[]).unwrap();
        assert_eq!(first[0].role, "system");
        assert!(first[0].text.contains("own knowledge"));
        assert!(first[0].images.is_empty());
        assert_eq!(first[1].role, "user");
        assert!(first[1].text.contains("[Page 3]"));
        let follow = vec![
            Turn { role: Role::Assistant, text: "A short summary [Page 3].".into(), images: Vec::new(), citations: Vec::new() },
            Turn { role: Role::User, text: "What about revenue?".into(), images: Vec::new(), citations: Vec::new() },
        ];
        let messages = prepared.conversation_messages(Task::Summary, "", &follow, &[]).unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[2].role, "assistant");
        assert!(messages[2].text.contains("A short summary"));
        assert_eq!(messages[3].text, "What about revenue?");
        let too_many: Vec<Turn> = (0..=MAX_TURNS)
            .map(|i| Turn {
                role: if i % 2 == 0 { Role::Assistant } else { Role::User },
                text: "x".into(),
                images: Vec::new(),
                citations: Vec::new(),
            })
            .collect();
        let err = prepared.conversation_messages(Task::Summary, "", &too_many, &[]).unwrap_err();
        assert!(err.contains("too long"), "{err}");
        let huge = vec![Turn { role: Role::User, text: "é".repeat(MAX_HISTORY_BYTES + 1), images: Vec::new(), citations: Vec::new() }];
        assert!(prepared.conversation_messages(Task::Summary, "", &huge, &[]).is_err());
        let not_user = vec![Turn { role: Role::Assistant, text: "only a reply".into(), images: Vec::new(), citations: Vec::new() }];
        assert!(prepared.conversation_messages(Task::Summary, "", &not_user, &[]).unwrap_err().contains("last message"));
        let image = ChatImage::from_rgba(2, 2, &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255]).unwrap();
        let pictured = vec![Turn { role: Role::User, text: String::new(), images: vec![image.clone()], citations: Vec::new() }];
        let with_image = prepared.conversation_messages(Task::Ask, "What is this?", &pictured, &[]).unwrap();
        assert_eq!(with_image.last().map(|m| m.images.len()), Some(1));
        let snapshot = prepared.conversation_messages(Task::Summary, "", &[], std::slice::from_ref(&image)).unwrap();
        assert_eq!(snapshot[1].images.len(), 1);
    }

    #[test]
    fn ai_follow_up_http_sends_prior_turns_and_page_markers() {
        let (config, worker) = server("200 OK", r#"{"choices":[{"finish_reason":"stop","message":{"content":"Revenue is ten [Page 3]."}}]}"#, "");
        let mut provider = OpenAiProvider::new(config, "synthetic-api-key".into()).unwrap();
        let prepared = Prepared { text: "[Page 3]\nSynthetic document".into(), pages: vec![3] };
        let turns = vec![
            Turn { role: Role::Assistant, text: "A short summary [Page 3].".into(), images: Vec::new(), citations: Vec::new() },
            Turn { role: Role::User, text: "What about revenue?".into(), images: Vec::new(), citations: Vec::new() },
        ];
        assert_eq!(prepared.complete_conversation(&mut provider, Task::Summary, "", &turns, &[]).unwrap(), "Revenue is ten [Page 3].");
        let request = worker.join().unwrap();
        let body: serde_json::Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert!(messages.len() > 1, "{messages:?}");
        assert_eq!(messages[0]["role"], "system");
        assert!(messages[1]["content"].as_str().unwrap().contains("[Page 3]"));
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["content"], "A short summary [Page 3].");
        assert_eq!(messages[3]["content"], "What about revenue?");
        assert!(!request.contains("synthetic-api-key") || request.to_lowercase().contains("authorization: bearer synthetic-api-key"));
    }

    #[test]
    fn ai_follow_up_http_sends_image_data_urls() {
        let (config, worker) = server("200 OK", r#"{"choices":[{"finish_reason":"stop","message":{"content":"A red square."}}]}"#, "");
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        let prepared = Prepared { text: "[Page 1]\nSynthetic document".into(), pages: vec![1] };
        let image = ChatImage::from_rgba(2, 2, &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255]).unwrap();
        let turns = vec![Turn { role: Role::User, text: "What colour?".into(), images: vec![image], citations: Vec::new() }];
        assert_eq!(prepared.complete_conversation(&mut provider, Task::Ask, "What colour?", &turns, &[]).unwrap(), "A red square.");
        let request = worker.join().unwrap();
        let body: serde_json::Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        let content = body["messages"].as_array().unwrap().last().unwrap()["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "What colour?");
        let url = content[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"), "{url}");
        assert!(body["messages"][1]["content"].as_str().unwrap().contains("[Page 1]"));
    }

    #[test]
    fn ai_stream_delivers_each_piece_and_still_returns_the_full_answer() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (mut config, worker) = server("200 OK", body, "");
        config.stream_responses = true;
        let mut provider = OpenAiProvider::new(config, "synthetic-secret".into()).unwrap();
        let mut seen = Vec::new();
        let answer = provider
            .complete_chat_reporting(&[ChatMessage { role: "user".into(), text: "Hi".into(), images: Vec::new() }], StreamKind::Plain, &mut |text| {
                seen.push(text.to_string());
            })
            .unwrap();
        assert_eq!(answer, "Hello");
        assert_eq!(seen, ["", "Hel", "Hello"]);
        let request = worker.join().unwrap();
        let body: serde_json::Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["stream"], true);
        assert!(body.get("max_completion_tokens").is_none());
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn a_reply_cut_off_by_the_output_limit_is_kept() {
        let (config, worker) = server("200 OK", r#"{"choices":[{"finish_reason":"length","message":{"content":"partial summary"}}]}"#, "");
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        assert_eq!(provider.complete("hello", &Default::default()).unwrap(), "partial summary");
        worker.join().unwrap();

        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Kept summary\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (mut config, worker) = server("200 OK", body, "");
        config.stream_responses = true;
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        let answer = provider
            .complete_chat_reporting(&[ChatMessage { role: "user".into(), text: "Hi".into(), images: Vec::new() }], StreamKind::Plain, &mut |_| {})
            .unwrap();
        assert_eq!(answer, "Kept summary");
        worker.join().unwrap();
    }

    #[test]
    fn ai_stream_shows_a_json_answer_as_it_arrives_and_redacts_the_key() {
        let event = |content: &str| {
            let payload = serde_json::json!({"choices":[{"delta":{"content": content}}]});
            format!("data: {payload}\n\n")
        };
        let body = format!(
            "{}{}{}data: {{\"choices\":[{{\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
            event(r#"{"answer":"Revenu"#),
            event("e is synthetic-secret."),
            event("\"}"),
        );
        let (mut config, worker) = server("200 OK", &body, "");
        config.stream_responses = true;
        let mut provider = OpenAiProvider::new(config, "synthetic-secret".into()).unwrap();
        let mut seen = Vec::new();
        let answer = provider
            .complete_chat_reporting(&[ChatMessage { role: "user".into(), text: "Hi".into(), images: Vec::new() }], StreamKind::Answer, &mut |text| {
                seen.push(text.to_string())
            })
            .unwrap();
        assert!(answer.contains("\"answer\""), "{answer}");
        assert!(!answer.contains("synthetic-secret"), "{answer}");
        assert!(answer.contains("[API key removed]"));
        assert!(seen.iter().any(|text| text == "Revenue is [API key removed]."), "{seen:?}");
        assert!(seen.iter().all(|text| !text.contains("synthetic-secret")));
        worker.join().unwrap();
    }

    #[test]
    fn ai_stream_accepts_a_server_that_returns_one_json_body() {
        let (mut config, worker) = server("200 OK", r#"{"choices":[{"finish_reason":"stop","message":{"content":"All at once."}}]}"#, "");
        config.stream_responses = true;
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        let mut seen = Vec::new();
        let answer = provider
            .complete_chat_reporting(&[ChatMessage { role: "user".into(), text: "Hi".into(), images: Vec::new() }], StreamKind::Plain, &mut |text| {
                seen.push(text.to_string());
            })
            .unwrap();
        assert_eq!(answer, "All at once.");
        assert_eq!(seen, ["", "All at once."]);
        worker.join().unwrap();
    }

    #[test]
    fn ai_stream_can_be_turned_off() {
        let (mut config, worker) = server("200 OK", r#"{"choices":[{"finish_reason":"stop","message":{"content":"Waited."}}]}"#, "");
        config.stream_responses = false;
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        let mut seen = Vec::new();
        let answer = provider
            .complete_chat_reporting(&[ChatMessage { role: "user".into(), text: "Hi".into(), images: Vec::new() }], StreamKind::Plain, &mut |text| {
                seen.push(text.to_string());
            })
            .unwrap();
        assert_eq!(answer, "Waited.");
        assert!(seen.is_empty());
        let request = worker.join().unwrap();
        let body: serde_json::Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn a_page_tool_is_fetched_and_kept_for_the_next_turn() {
        let mut session = crate::Session::new();
        let id = session.open("Pages", None, std::sync::Arc::new(crate::tests::fixture(2)), None).unwrap();
        let document = session.get(id).unwrap();
        let mut source = document.export_source();
        source.config.system_fonts = false;
        let outline = super::index::flatten_outline(&document.info.outline);
        let marks = super::index::structure_marks(&source);
        let span_id = super::index::page_index(&source, 0, &outline, &marks).unwrap().spans[0].id.clone();
        let cited = format!(r#"{{"answer":"Page one [c1].","citations":[{{"id":"c1","span_id":"{span_id}"}}],"viewer_actions":[],"follow_ups":[]}}"#);
        let (config, worker) = server_queue(vec![
            chat_json(r#"{"tool":"get_pdf_page_text","pages":"1-2"}"#),
            chat_json(&cited),
            chat_json(r#"{"answer":"Still page one.","citations":[],"viewer_actions":[],"follow_ups":[]}"#),
        ]);
        let mut provider = OpenAiProvider::new(config, String::new()).unwrap();
        let pack = EvidencePack { text: "Other.".into(), pages: vec![], spans: vec![], activities: vec![], notes: vec![] };
        let mut notebook = PageNotebook::default();
        let first = complete_evidence(
            &mut provider,
            &source,
            &outline,
            pack.clone(),
            &mut notebook,
            Task::Ask,
            "Explain the design.",
            &[],
            &[],
            None,
            |_| Err("no search".into()),
            |_| {},
            |_| {},
        )
        .unwrap();
        assert!(first.answer.contains("Page one"), "{}", first.answer);
        assert_eq!(first.citations.first().map(|citation| citation.span_id.as_str()), Some(span_id.as_str()));
        assert!(notebook.text.contains("Page 1"), "{}", notebook.text);
        assert!(notebook.text.contains("Page 2"), "{}", notebook.text);
        let turns = vec![Turn { role: Role::User, text: "What does the first page say?".into(), images: Vec::new(), citations: Vec::new() }];
        let second = complete_evidence(
            &mut provider,
            &source,
            &outline,
            pack,
            &mut notebook,
            Task::Ask,
            "Explain the design.",
            &turns,
            &[],
            None,
            |_| Err("no search".into()),
            |_| {},
            |_| {},
        )
        .unwrap();
        assert!(second.answer.contains("Still page one"), "{}", second.answer);
        let requests = worker.join().unwrap();
        assert!(requests[1].contains("Page 1"), "{}", requests[1]);
        assert!(requests[1].contains(&span_id), "{}", requests[1]);
        assert!(requests[2].contains("Page 1"), "{}", requests[2]);
    }
}
