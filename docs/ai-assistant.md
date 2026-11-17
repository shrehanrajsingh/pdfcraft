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
`/chat/completions`, Bearer authentication, and `max_completion_tokens`. Enable **Use legacy
max_tokens** only for servers that require the older parameter. Temperature is omitted for
compatibility with reasoning models. An OpenAI API key and API billing are separate from a
ChatGPT subscription.

Provider settings (URL, model, API key, **Test connection**) live behind the panel gear, and
also in **Preferences → AI assistant**. The chat surface is a separate view: **Summarize**,
**Ask**, or **Translate**, **This page** or **Document**, then **Preview**. Review the extracted
text, check **Allow sending this text for this request**, and press **Send**. The reply appears
as a chat turn (CommonMark via egui_commonmark, including GitHub pipe tables, and TeX math via RaTeX; copy uses the original markdown). A docked **Ask a follow-up** field
sends later messages with the same page snapshot and the conversation so far; each send is
explicit. Paste an image into the chat box (or use **Add image**) to send it with that turn.
Previewing the page text keeps the attachments.
PNG and JPEG are sent as-is when they already fit, and other still images are re-encoded as PNG.
At most four pictures per message and eight in the conversation; each is downscaled to 2048 pixels
on a side and 2 MiB encoded. **New conversation** clears the thread. Nothing is sent on opening a
PDF, restoring settings, enabling AI, or previewing text. A hosted provider may charge for usage;
a loopback server may itself forward to another service. The assistant never edits or saves the PDF.

## Limits and privacy

- Native desktop and CLI/MCP only; the web build remains usable and explains that AI requests
  need the desktop app.
- At most 100 pages and 120,000 UTF-8 bytes per request, with no silent truncation. The UI offers
  the document or current page; automation also accepts an explicit list of pages. Scans need OCR
  first. Extracted page text is sent, plus any pictures you attach or paste in the chat (not PDF
  bytes or other attachments). Extractable text may include invisible/OCR text: review the
  preview, not only the page image. Attached pictures are session-only and are never written to
  preferences.
- Questions/languages are bounded to 8,000 UTF-8 bytes; output is capped at 2,048 tokens and
  responses at 1 MiB. Incomplete/refused answers are reported as errors. Long translations may
  require smaller page selections.
- Copy restrictions and pending redactions are checked before extraction and again before the
  UI sends. Apply or remove redaction marks first: a mark alone leaves its underlying text intact.
- Model answers and page citations can be wrong. Citations use physical, one-based page numbers;
  they are not independently verified or clickable. Translation produces text, not a translated PDF.
- Follow-up turns reuse the previewed snapshot (at most 24 turns and 32,000 UTF-8 bytes of
  history on top of the page text). Over those limits, start a new conversation; nothing is
  truncated silently. History is session-only and is not written to preferences.
- No autonomous editing, streaming, retrieval over large documents, local model downloads,
  stored API keys, or native Anthropic/Azure-specific authentication. Services must
  implement the OpenAI-compatible chat-completions route and optional Bearer authentication.
- HTTPS is required except on loopback. URL credentials, queries and fragments are refused.
  Redirects are refused, requests time out after 120 seconds, and provider error bodies are not
  echoed. The UI runs one worker at a time; closing the panel does not cancel an already-sent
  request. Changing documents discards its result. No automatic retries incur repeated charges.

## CLI and MCP

Configure the process launching `pdfcraft-cli` with these environment variables (use your shell
or secret manager; never put real keys in scripts committed to the repository):

- `PDFCRAFT_AI_ENABLED=1`
- `PDFCRAFT_AI_BASE_URL=https://api.openai.com/v1`
- `PDFCRAFT_AI_MODEL=<model ID>` (required for generate; optional for `ai_provider_test`)
- `PDFCRAFT_AI_API_KEY=<API key>` (omit for an unauthenticated local server)
- `PDFCRAFT_AI_LEGACY_TOKENS=1` only for older compatible servers

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
`ai_provider_test` with `allow_send:true` to test model listing. User turns on `ai_chat` may
include `images` (paths under `--root`); optional top-level `images` go with the document
snapshot. All network tools require explicit `allow_send:true`; call `ai_preview` separately
first when approval of the extracted text is needed. They return text and source pages without
writing files. An MCP client's approval policy must treat these as external requests, even
though they leave the PDF unchanged.

The opt-in UI control channel also exposes `ai.settings`, `ai.summary`, `ai.ask`, `ai.translate`,
`ai.preview`, `ai.send`, `ai.new`, and `ai.test`. After a thread exists, `ai.send` sends the
follow-up in `ai-input`. `ui.set` accepts `panel=ai` (and `dialog=ai` as an alias that opens the
panel), `ai-enabled`, `ai-url`, `ai-model`, `ai-input`, `ai-current-page`, `ai-legacy-tokens`,
and `ai-consent`. API keys are entered only in the masked desktop field; `ui.state` excludes the
key, document content, and conversation text.

## Design and shared-library boundary

`pdfcraft_engine::ai` checks document permissions, captures the working document, extracts bounded
text with source-page markers, and calls the shared `craft-ai::Provider` contract. The UI owns
session-only credentials and background jobs; CLI/MCP uses the same engine path.

The dependency is pinned to the upstream merge of
[craft-libs#1181](https://github.com/storytold/craft-libs/pull/1181), under `MIT OR Apache-2.0`.
Its `OpenAiCompat` does not expose an injectable HTTP client/redirect policy, response-size policy,
or a choice of modern versus legacy token-limit parameters. PdfCraft therefore implements a
small transport adapter over that shared trait using the already present `ureq` (`MIT OR Apache-2.0`)
and `url` crates. Moving these policy knobs into the shared provider would allow retiring the
adapter without changing the PDF, UI or automation layers. Local-model fetching/inference is not
enabled. No model files, fonts, images or other assets are added.

Protocol reference: [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
