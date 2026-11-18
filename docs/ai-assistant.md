# Bring your own model

Open **Preferences → AI assistant**, **All tools → AI assistant**, or the sparkles item on
the right-hand rail. Summarize, Ask and Translate open as a side panel beside the page, so the
document stays readable. Enable the assistant, enter a base URL and model ID, and paste the
provider's API key into the masked field in the app. The key is optional for local servers that
do not require authentication.
It stays in memory for this session; closing the app, clearing the key, or disabling AI removes
the stored key. The URL, model ID and non-secret options are saved with preferences.

Examples of base URLs:

| Provider | Base URL | Model ID |
| --- | --- | --- |
| OpenAI | `https://api.openai.com/v1` | A chat-completions model ID such as `gpt-4o-mini` or `gpt-4o` (not ChatGPT) |
| Ollama | `http://localhost:11434/v1` | An installed model's exact name |
| LM Studio | `http://localhost:1234/v1` | A model served by its local API server |
| Other compatible service | Its HTTPS API root | Its documented model ID |

**Test connection** calls `/models` without sending document text and without requiring a model
ID. A successful listing does not prove that a particular model supports generation. You can
still enter the model manually if the service does not implement model listing. Requests use
`/chat/completions` and Bearer authentication. No output-token limit is sent. Temperature is omitted for
compatibility with reasoning models. An OpenAI API key and API billing are separate from a
ChatGPT subscription.

Provider settings (URL, model, API key, **Test connection**) live behind the panel gear, and
also in **Preferences → AI assistant**. The chat surface is a separate view: **Summarize**,
**Ask**, or **Translate**, **This page** or **Document**, then **Preview**. Review the extracted
text, check **Allow sending this text for this request**, and press **Send**. Ask and Summarize
preview the passages retrieval will send (the open page and a selection rank first, and other
matching pages are still included, within the same byte cap). Translate still previews the
selected pages in full. The reply appears as it is written. Turn off **Stream the answer as it is written** in the provider settings to wait for the complete answer instead.
The finished reply is a chat turn (CommonMark via egui_commonmark, including GitHub pipe tables, and TeX math via RaTeX; copy uses the original markdown). Markers the index can check, `[c1]` and `[Page N]`,
become citation chips: hover shows the page and a short excerpt, and a click opens that passage
and flashes its boxes for about two seconds. A marker the evidence does not contain is removed.
A docked **Ask a follow-up** field
sends later messages with a fresh retrieval and the conversation so far; each send is
explicit. Suggested follow-ups fill that field. Paste an image into the chat box (or use **Add image**) to send it with that turn.
Previewing the page text keeps the attachments. Selecting text offers **Ask AI**, **Explain**, and
**Summarize**; the selection stays, and the quoted passage stays attached until you clear it.
PNG and JPEG are sent as-is when they already fit, and other still images are re-encoded as PNG.
At most four pictures per message and eight in the conversation; each is downscaled to 2048 pixels
on a side and 2 MiB encoded. **New conversation** clears the thread. Nothing is sent on opening a
PDF, restoring settings, enabling AI, or previewing text. A hosted provider may charge for usage;
a loopback server may itself forward to another service. The assistant never edits or saves the PDF.

## Limits and privacy

- Native desktop and CLI/MCP only; the web build remains usable and explains that AI requests
  need the desktop app.
- At most 100 pages and 120,000 UTF-8 bytes per request, with no silent truncation. The UI offers
  the document or current page; automation also accepts an explicit list of pages. Ask and
  Summarize send retrieved spans, not a silent dump of every page. Scans need OCR
  first. Extracted page text is sent, plus any pictures you attach or paste in the chat (not PDF
  bytes or other attachments). **Send PDF page images** (Preferences, and only when **This model
  accepts images** is on) also sends a downscaled render of a page or selected region when the
  question is visual or a region is selected. Text questions do not attach those renders. If the
  model is not marked as accepting images, the toggle stays off and the panel says why.
  Extractable text may include invisible/OCR text: review the
  preview, not only the page image. Attached pictures are session-only and are never written to
  preferences. The panel shows a one-line activity status while a request runs. There is no
  automatic retry of a failed request.
- Questions/languages are bounded to 8,000 UTF-8 bytes. No output-token limit is sent, and the
  reply is not cut off at a byte size. The provider stops at its own context window. A reply that
  already contains text is kept if the provider still stops for length. An empty or refused answer
  is reported as an error. Long translations may require smaller page selections.
- Copy restrictions and pending redactions are checked before extraction and again before the
  UI sends. Apply or remove redaction marks first: a mark alone leaves its underlying text intact.
- Model answers can still be wrong. Ask and Summarize keep a citation only when its span id or
  page is in the evidence for that request, and the chip uses the index's boxes, not coordinates
  from the model. A page with no glyph boxes is marked page-level. Translation produces text, not
  a translated PDF, and its page markers are not checked.
- Follow-up turns for Ask and Summarize retrieve again (at most 24 turns and 32,000 UTF-8 bytes of
  history on top of the evidence). One Send may run up to three provider rounds when the model
  asks for another search. Translate follow-ups reuse the previewed snapshot. Over those limits,
  start a new conversation; nothing is truncated silently. History is session-only and is not
  written to preferences. Layout without structure tags is line and paragraph geometry. Math is
  the extracted characters, or a crop when page images are on.
- No autonomous editing, local model downloads, stored API keys, vector database, or
  native Anthropic/Azure-specific authentication. Services must
  implement the OpenAI-compatible chat-completions route and optional Bearer authentication.
  Embeddings are not used.
- HTTPS is required except on loopback. URL credentials, queries and fragments are refused.
  Redirects are refused. Connecting times out after 10 seconds, and waiting for response headers
  times out after 120 seconds. A generation itself is not cut off by a total time limit. Text
  already received is kept if the connection then stops, and a body or stream event that is not
  JSON is shown as text. Provider error bodies are not echoed. The UI runs one worker at a time;
  closing the panel does not cancel an already-sent request. Changing documents discards its
  result. No automatic retries incur repeated charges.

## CLI and MCP

Configure the process launching `pdfcraft-cli` with these environment variables (use your shell
or secret manager; never put real keys in scripts committed to the repository):

- `PDFCRAFT_AI_ENABLED=1`
- `PDFCRAFT_AI_BASE_URL=https://api.openai.com/v1`
- `PDFCRAFT_AI_MODEL=<model ID>` (required for generate; optional for `ai_provider_test`)
- `PDFCRAFT_AI_API_KEY=<API key>` (omit for an unauthenticated local server)
- `PDFCRAFT_AI_MODEL_IMAGES=1` when the configured model accepts images
- `PDFCRAFT_AI_PAGE_IMAGES=1` to allow a page or region render on a visual question
- `PDFCRAFT_AI_STREAM=1` to ask the provider for a streamed reply; the tool still returns the finished text

The destination and credentials are launcher configuration, never model-controlled tool arguments.
`--root` still confines document reads and writes; it does not disable these explicitly enabled
provider requests. The MCP server stays opt-in and is never registered or started automatically.

Example script for `pdfcraft-cli run --script steps.json --root <document folder>`:

```json
[
  {"tool":"doc_open","args":{"path":"report.pdf"}},
  {"tool":"ai_preview","args":{"doc":1,"pages":[1,2]}},
  {"tool":"ai_ask","args":{"doc":1,"pages":[1,2],"input":"What are the main findings?","allow_send":true}}
]
```

Use `ai_summary` for summaries, `ai_translate` with `input` set to a language, `ai_chat` for a
follow-up (`task`, original `input`, and `messages` ending with the new user turn), and
`ai_provider_test` with `allow_send:true` to test model listing. `ai_ask` and `ai_summary` accept
an optional `page` (1-based, ranked first) and `selection` (`text`, `page`, glyph `start`/`end`,
or a view-space `rect`). They return `citations` whose span ids were in the evidence, with page
and view-space `rects`. `ai_index` and `ai_search` do the same lookup with no network and no
`allow_send`. User turns on `ai_chat` may
include `images` (paths under `--root`); optional top-level `images` go with the document
text. All network tools require explicit `allow_send:true`; call `ai_preview` or `ai_index`
first when approval of the extracted text is needed. They return text and source pages without
writing files. An MCP client's approval policy must treat the network tools as external requests, even
though they leave the PDF unchanged.

The opt-in UI control channel also exposes `ai.settings`, `ai.summary`, `ai.ask`, `ai.translate`,
`ai.preview`, `ai.send`, `ai.new`, and `ai.test`. After a thread exists, `ai.send` sends the
follow-up in `ai-input`. `ui.set` accepts `panel=ai` (and `dialog=ai` as an alias that opens the
panel), `ai-enabled`, `ai-url`, `ai-model`, `ai-input`, `ai-current-page`, `ai-legacy-tokens`,
and `ai-consent`. API keys are entered only in the masked desktop field; `ui.state` excludes the
key, document content, and conversation text.

## Design and shared-library boundary

`pdfcraft_engine::ai` checks document permissions, captures the working document, and calls the
shared `craft-ai::Provider` contract. Ask and Summarize build a lazy span index from the page
text, retrieve a bounded set of passages, and drop any citation whose span id was not in that
set. Translate still sends the selected pages with `[Page N]` markers. The UI owns
session-only credentials and background jobs; CLI/MCP uses the same engine path. The model
answers as a normal assistant: the open document is context, and it uses its own knowledge when
the pages do not cover the question. Document text stays in the user message.

The dependency is pinned to the upstream merge of
[craft-libs#1181](https://github.com/storytold/craft-libs/pull/1181), under `MIT OR Apache-2.0`.
Its `OpenAiCompat` does not expose an injectable HTTP client/redirect policy, response-size policy,
or a choice of modern versus legacy token-limit parameters. PdfCraft therefore implements a
small transport adapter over that shared trait using the already present `ureq` (`MIT OR Apache-2.0`)
and `url` crates. Moving these policy knobs into the shared provider would allow retiring the
adapter without changing the PDF, UI or automation layers. Local-model fetching/inference is not
enabled. No model files, fonts, images or other assets are added.

Protocol reference: [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
