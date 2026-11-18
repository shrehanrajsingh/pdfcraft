# MASTER IMPLEMENTATION PROMPT: Agentic AI-Powered PDF Workspace

## 1. Project Context

I currently have an application that allows users to upload PDF documents, extracts their text, sends the extracted text to an LLM, and enables conversational question-answering based on the document.

I want to transform this existing application into an **advanced agentic, multimodal, context-aware PDF intelligence platform**.

The AI should no longer behave like a basic chatbot that receives all extracted text. Instead, it should function as an intelligent document agent capable of understanding document structure, identifying relevant sections, interpreting visual content, navigating pages, retrieving information, and linking every factual answer to the corresponding source inside the PDF.

The AI should have **controlled, tool-based agency over the PDF viewer and document content**, rather than simply receiving static text.

**Important:** You are working with an existing codebase. First, inspect the entire project and understand its existing architecture, libraries, frontend, backend, PDF rendering system, AI integration, and document-processing pipeline.

Extend and refactor the existing implementation wherever necessary. Do not unnecessarily rebuild functioning components.

Implement the features below as a cohesive, production-oriented system, not as isolated demonstrations.

---

## 2. Core Objective: An Agentic Document Intelligence System

Create an AI agent capable of interacting intelligently with the PDF and its viewer.

The agent should understand:

- Which document is open.
- Which page the user is currently viewing.
- The page's extracted text and layout.
- What sections, paragraphs, tables, figures, and images exist.
- What text or region the user has selected.
- The user's conversation history and current intent.
- Where specific information is located in the document.
- Which PDF region supports a particular AI-generated statement.

The agent should autonomously decide when to search, retrieve, inspect, reason over, and reference different document regions.

### Agent capabilities

Implement document interaction tools with functionality similar to:

`get_document_metadata(document_id)`

Return document title, page count, processing status, and document structure.

`get_current_page()`

Return the currently visible page number and viewport information.

`get_page_text(page_number)`

Retrieve text with structural metadata and positional information.

`get_page_image(page_number, resolution)`

Retrieve a rendered page image for supported multimodal models.

`get_page_regions(page_number)`

Retrieve identified text blocks, paragraphs, tables, figures, and bounding boxes.

`search_document(query, filters)`

Perform semantic and keyword searches across PDF content.

`get_relevant_context(query, page_number, selection)`

Retrieve relevant document chunks with their surrounding context.

`inspect_region(page_number, bounding_box)`

Inspect a particular region, using text extraction and optionally visual input.

`navigate_to_page(page_number)`

Request that the frontend navigate to a particular page.

`highlight_pdf_region(page_number, bounding_boxes, duration)`

Request a temporary visual highlight of one or multiple source regions.

`get_conversation_context()`

Retrieve relevant conversation history.

These are conceptual tool interfaces. Adapt their actual implementation to the existing architecture.

**Crucial architectural constraint:** The AI must not directly manipulate browser internals or execute arbitrary frontend code. Agent tool calls must be validated and translated into controlled frontend/backend operations.

Use an agent orchestration loop that supports tool calling, bounded reasoning steps, errors, retries, and clear termination conditions.

---

## 3. Multimodal PDF Understanding

Currently, my application converts PDFs into plain text before sending content to the LLM.

Improve this approach to support both text-based and visual document understanding.

### 3.1 Text + image input

When the user asks questions such as:

- "What is happening on this page?"
- "Explain this diagram."
- "What does this graph represent?"
- "Summarize the content visible here."
- "Explain the highlighted figure."
- "What is the relationship between the table and the paragraph?"

The system should be able to send:

1. Extracted text from the relevant PDF page.
2. A rendered image of the same page.
3. Structural and positional metadata.
4. User query and relevant conversational context.

Both extracted text and the rendered page image should be included when visual input is enabled and supported.

The image must correspond to the correct page or selected region.

Avoid sending unnecessary page images when the question can be answered using text alone.

### 3.2 Multimodal toggle

Implement a user-facing toggle:

**"Send PDF page images to AI" — ON/OFF**

Behavior:

- ON: Use page images alongside extracted text when the selected model supports image input.
- OFF: Use extracted text and available structural metadata only.
- If the selected model does not support vision, automatically disable image sending and indicate why.
- Store the user's toggle preference.
- Display a helpful indication that visual analysis may use additional tokens and processing resources.

Create a model capability configuration that distinguishes vision-capable models from text-only models.

Do not assume that all LLM APIs use the same image input format. Implement a provider abstraction for different models.

### 3.3 Intelligent image selection

Use appropriate image granularity:

- Full-page image for page-level explanation.
- Cropped region for selected diagrams or figures.
- Multiple page images only when needed for comparative questions.
- Text-only retrieval for ordinary factual questions.

Implement image resizing, caching, and configurable resolution to manage costs.

For text-only models, allow the agent to access precomputed descriptions of visual elements where available, without pretending that it directly saw the image.

### 3.4 Complex PDF structures

Support, as far as practical:

- Multi-column PDFs.
- Tables.
- Charts and plots.
- Figures and diagrams.
- Mathematical expressions.
- Scanned documents with optional OCR.
- Rotated pages.
- Section headings.
- Headers and footers.
- Footnotes.
- Text spanning multiple pages.

Preserve reading order and layout information whenever possible.

---

## 4. Precise PDF Citation System

This is one of the most important requirements.

Every answer containing information derived from the document should provide **accurate, interactive citations linked to specific PDF regions**.

Do not limit citations to page numbers.

### 4.1 Citation granularity

Support citations at multiple levels:

- Document.
- Page.
- Section.
- Paragraph.
- Sentence.
- Table or figure.
- Multiple text spans where necessary.

For example, an AI response might look like:

"The experiment demonstrates an improvement in accuracy after the second training phase [1]. However, the validation error increases beyond the seventh epoch [2]."

Citation `[1]` should point to the exact paragraph, figure, or table containing the accuracy results.

Citation `[2]` should point to the section discussing validation error.

### 4.2 Citation data model

Define structured citations rather than trusting free-form model-generated coordinates.

Example:

```json
{
  "citation_id": "cite_001",
  "document_id": "doc_123",
  "page_number": 5,
  "section_id": "section_5_2",
  "chunk_id": "chunk_42",
  "text_span_ids": ["span_105", "span_106"],
  "quoted_text": "Validation error increases after epoch seven.",
  "bounding_boxes": [
    {
      "x": 0.12,
      "y": 0.38,
      "width": 0.67,
      "height": 0.09
    }
  ],
  "coordinate_system": "normalized_pdf_page"
}
```

The actual data structure should be adapted to the PDF rendering system.

**Important:** Citation IDs must map back to real, indexed source content. The LLM must not invent page positions, bounding boxes, or source references.

Use stable source identifiers generated during extraction. Resolve coordinates on the application side.

Validate every generated citation against actual document content before displaying it.

### 4.3 Citation interaction

Implement the following behavior.

**Hover over citation:**

- Display a compact source preview.
- Show the page number and a short excerpt.
- Optionally display a thumbnail of the relevant page region.

**Click citation:**

1. Navigate the PDF viewer to the cited page.
2. Scroll the corresponding section into view.
3. Apply a temporary visual highlight around the exact cited region.
4. Keep the highlight visible for approximately 2 seconds.
5. Fade it out smoothly.
6. Keep the document at the selected location.

The highlight should be clearly visible but should not permanently obscure the PDF text.

Use a translucent overlay, subtle border, and smooth animation.

If multiple boxes correspond to a single citation, highlight all relevant regions.

### 4.4 Hover and highlight robustness

Ensure the system handles:

- Different zoom levels.
- Window resizing.
- Page rotations.
- Multiple bounding boxes.
- PDF viewer scrolling.
- Lazy-loaded and virtualized pages.
- Multi-page citations.
- Repeated identical text appearing on the same page.

The highlighting system must identify the correct source span, not simply find the first matching text string.

Prefer coordinate-aware overlays over DOM text-search-based highlighting.

If exact coordinates are unavailable, gracefully fall back to page-level highlighting and disclose the reduced precision.

---

## 5. Interactive PDF Text Selection and "Ask AI"

Implement direct interaction with selected text inside the PDF viewer.

### 5.1 Text selection workflow

When the user selects text inside a PDF:

1. Detect the selected text.
2. Preserve the selection's page number and positional information.
3. Display a small contextual floating toolbar near the selection.
4. Include an **"Ask AI"** action.

Optionally provide additional actions:

- Explain.
- Summarize.
- Simplify.
- Translate.
- Find related information.
- Copy with citation.

Keep the floating toolbar compact and visually clean.

### 5.2 Ask AI workflow

When the user clicks "Ask AI":

- Capture the exact selected text.
- Capture the source document and page.
- Capture the selection's location and bounding boxes.
- Retrieve a reasonable amount of surrounding context.
- Retrieve relevant information from other document sections when necessary.
- Open or focus the AI chat interface.
- Display the selected text as a contextual attachment or quoted snippet.
- Allow the user to enter a follow-up question.

Example:

Selected PDF text:

"The model achieves convergence after approximately 40 iterations."

Question:

"Why does this happen, and what influences convergence?"

The LLM should receive:

```json
{
  "interaction_type": "selection_question",
  "document_id": "doc_123",
  "selection": {
    "text": "The model achieves convergence after approximately 40 iterations.",
    "page_number": 8,
    "region_ids": ["region_81"]
  },
  "user_question": "Why does this happen, and what influences convergence?",
  "surrounding_context": [],
  "retrieved_context": [],
  "conversation_context": []
}
```

The context arrays should be populated through the retrieval pipeline.

Do not send only the selected text in isolation.

The agent should distinguish between information directly stated in the selected passage and external explanations or deductions.

### 5.3 Additional selection interactions

Support selecting text across multiple lines and, if technically feasible, multiple pages.

Allow the user to clear the active selection or ask multiple follow-up questions about the same selection.

Preserve selection context across follow-up interactions until the user changes or clears it.

Avoid losing selection information when the user clicks the contextual toolbar or chat input.

### 5.4 Region selection

Extend text selection to support non-text content.

Allow users to select or draw a rectangle around a diagram, image, chart, or table.

Provide an "Ask AI about this region" option.

For vision-capable models, send an appropriate cropped image together with nearby text and page context.

For text-only models, use any available extracted content or visual descriptions and clearly indicate limitations.

---

## 6. Agentic Retrieval and Context Management

Replace the simplistic "extract entire PDF and send everything" approach with an intelligent retrieval system.

### 6.1 Document preprocessing

At upload time, process the PDF into a structured document representation.

Extract and store:

- Document metadata.
- Page text.
- Text blocks and paragraphs.
- Headings and section hierarchy.
- Tables and figures when detectable.
- Positional coordinates.
- Stable text span identifiers.
- Reading order.
- Page images or renderable references.
- Searchable chunks.
- Embeddings, if semantic retrieval is enabled.

Keep a connection between every indexed chunk and its exact PDF source location.

### 6.2 Hybrid retrieval

Implement an efficient retrieval pipeline combining:

- Semantic vector retrieval.
- Keyword or full-text search.
- Document section structure.
- Current page awareness.
- Selected region prioritization.
- Neighboring chunk expansion.
- Optional reranking.

Current-page context should receive appropriate priority without preventing retrieval from other relevant pages.

Avoid unnecessary retrieval when the user asks a straightforward question about an already available selection.

### 6.3 Agent decision-making

The system should identify the user's intent and choose an appropriate workflow.

Examples:

**"Summarize this PDF"**

Retrieve document structure and representative section content, then produce a structured summary.

**"Explain the current page"**

Retrieve the current page's text and optionally the page image, then explain its contents.

**"Where is reinforcement learning discussed?"**

Search across the PDF, return relevant page references, and optionally offer navigation.

**"Compare the results in sections 3 and 5"**

Retrieve both sections and provide a comparison with citations from each.

**"Explain this highlighted equation"**

Use the selected expression, its surrounding definitions, and related document content.

**"What does the graph on page 12 mean?"**

Retrieve the page image or graph region and associated caption, then explain the graph.

**"Find all references to this concept"**

Search the document and return a list of clickable source locations.

The agent must be able to execute multiple retrieval steps when a question requires it.

### 6.4 Context optimization

Implement token-aware context construction.

- Avoid repeatedly sending the entire document.
- Deduplicate retrieved passages.
- Prioritize the most relevant evidence.
- Preserve source metadata.
- Include necessary surrounding content.
- Account for model context-window limits.
- Cache reusable processing results.
- Support long documents efficiently.

Use conversational memory where appropriate without allowing earlier summaries to replace primary document evidence.

---

## 7. AI Response Format and Citation Verification

Use structured model output and a frontend renderer.

The response should separate narrative content from citation metadata and agent actions.

Conceptual schema:

```json
{
  "answer": "The results show an increase in accuracy...",
  "citations": [
    {
      "citation_id": "cite_001",
      "source_region_ids": ["region_12"],
      "page_number": 4
    }
  ],
  "viewer_actions": [
    {
      "type": "navigate",
      "page_number": 4
    }
  ],
  "follow_up_suggestions": [
    "Explain the methodology",
    "Compare these results"
  ]
}
```

Adapt this schema as needed, especially to allow inline citation markers in the answer.

Implement:

- Clickable citations embedded naturally in AI responses.
- A reliable citation-to-source resolver.
- Structured response validation.
- Missing or invalid citation handling.
- Clear separation of sourced facts, interpretations, and general knowledge.
- Helpful error handling when document evidence is insufficient.

For factual document-grounded claims, attach citations to the corresponding statements.

Never generate plausible-looking citations without verified underlying source references.

The system should acknowledge uncertainty and missing information rather than hallucinating document content.

---

## 8. Intelligent PDF Viewer Integration

The PDF viewer and AI assistant must work together as a unified interface.

### 8.1 Recommended layout

Use a split-screen interface:

**Left panel: PDF Viewer**

- Document rendering.
- Page navigation.
- Zoom controls.
- Text selection.
- Region selection.
- Citation highlighting.
- Search results.
- Optional document outline.

**Right panel: AI Workspace**

- Conversational interface.
- Inline interactive citations.
- Selected-text context cards.
- Follow-up questions.
- Model selection.
- Multimodal toggle.
- Agent progress or activity indication.

Allow resizing the panels.

Preserve the user's current PDF position while interacting with the chat.

Support responsive layouts for smaller displays.

### 8.2 Context awareness

Synchronize the PDF viewer's state with the AI system.

Track:

- Current document.
- Current page.
- Visible page range.
- Active selection.
- Zoom level.
- Available page content.
- Active citation.
- Conversation context.

Do not send every scroll event directly to the model. Store viewer state locally and attach relevant state when an AI request begins.

### 8.3 Agent navigation

Allow the agent to propose or trigger safe navigation actions.

For example, when asked:

"Take me to the methodology section."

The agent should locate the section and navigate the viewer to it.

For questions such as:

"Show me the evidence for your answer."

The agent should identify the strongest supporting citation and navigate to it.

Navigation and transient highlights may be executed automatically.

Destructive document changes, exports involving sensitive data, or externally transmitted content should require appropriate user authorization.

### 8.4 Agent activity indicator

When the agent performs multiple steps, display understandable progress such as:

- Searching document.
- Reading page 6.
- Inspecting figure.
- Comparing two sections.
- Verifying citations.
- Preparing response.

Keep the interface lightweight; users should not be overwhelmed with internal model details.

---

## 9. Additional Features to Implement

Incorporate the following features where they fit naturally with the existing application.

### 9.1 Search with AI

Provide both ordinary PDF text search and semantic search.

Example:

"Find the paragraph that explains why the experiment failed."

The system should identify relevant passages even when the user's words do not exactly match the PDF.

Search results should be clickable and highlight their PDF regions.

### 9.2 Smart summaries

Support:

- Full-document summary.
- Current-page summary.
- Selected-section summary.
- Chapter summary.
- Key findings.
- Important definitions.
- Main conclusions.

Generated summaries should contain source citations.

### 9.3 Contextual quick actions

Allow common actions through lightweight controls:

- Explain this.
- Explain like I'm a beginner.
- Summarize this section.
- Define terminology.
- Compare with another section.
- Generate questions from this section.
- Find supporting evidence.
- Identify assumptions or contradictions.

Use the same retrieval and citation architecture instead of creating separate pipelines for each feature.

### 9.4 Document outline

Create a navigable outline from PDF bookmarks or extracted headings.

When the user clicks a heading, scroll to the corresponding region.

The agent should also be able to use this hierarchy during retrieval.

### 9.5 Conversation memory

Maintain document-specific conversation sessions.

Persist:

- Conversation messages.
- Relevant document references.
- Selected text context.
- Model configuration where appropriate.

Allow users to resume previous conversations.

Prevent document context from leaking between unrelated files or users.

### 9.6 Cross-page reasoning

Support questions whose answers require evidence from multiple pages.

The agent should retrieve each relevant region and provide citations to the different sources.

### 9.7 Multi-document readiness

Design document identifiers, retrieval interfaces, and citation references so multiple PDFs can be supported later, even if the initial implementation operates on one document at a time.

If existing multi-document support is already present, preserve it and integrate the features with it.

### 9.8 Export and notes

Where practical, provide options to:

- Copy an answer with citations.
- Save a response as a note.
- Bookmark important source regions.
- Export conversation history.
- Export notes with page references.

These features should not interfere with the core document-reading experience.

---

## 10. Technical Architecture

Inspect the existing technology stack and use compatible solutions.

If the application uses PDF.js or a similar renderer, leverage its text layers, coordinate transformations, viewport APIs, and rendering system wherever possible.

### 10.1 Suggested architecture

```text
PDF Upload
    |
    v
PDF Processing Pipeline
    |
    +-- Text Extraction
    +-- Layout / Region Detection
    +-- Optional OCR
    +-- Section / Table / Figure Detection
    +-- Page Coordinate Mapping
    +-- Chunking + Source IDs
    +-- Optional Embedding Generation
    |
    v
Structured Document Store
    |
    +-- Original PDF
    +-- Page Metadata
    +-- Source Regions + Bounding Boxes
    +-- Search Index
    +-- Embedding Index
    +-- Rendered Images / Cache
    |
    v
Document Agent Orchestrator
    |
    +-- Intent Understanding
    +-- Context Planning
    +-- Hybrid Retrieval
    +-- Page / Region Inspection
    +-- Multimodal Model Adapter
    +-- Tool Calling
    +-- Evidence Verification
    |
    v
Structured AI Response
    |
    +-- Answer
    +-- Inline Citations
    +-- Viewer Actions
    +-- Follow-up Suggestions
    |
    v
Interactive Frontend
    |
    +-- PDF Viewer
    +-- Citation Navigation
    +-- Timed Highlight Overlays
    +-- Selection Toolbar
    +-- AI Chat
    +-- Model and Vision Controls
```

### 10.2 System component separation

Keep these responsibilities modular:

1. PDF ingestion and extraction.
2. Document structure and positional metadata.
3. Search and retrieval.
4. Agent orchestration.
5. LLM provider integration.
6. Citation resolution and verification.
7. Viewer synchronization.
8. UI interactions and conversation state.

Use clear, typed interfaces between components.

### 10.3 Model flexibility

Preserve existing model integration and make it extensible.

Support configurable models with properties such as:

- Text support.
- Vision support.
- Tool-calling support.
- Context-window size.
- Input limits.
- Provider-specific formatting.

If a model lacks native tool calling, use a safe fallback orchestration approach where feasible.

The multimodal toggle and tool availability must reflect actual model capabilities.

### 10.4 Agent security

PDF documents and extracted text are untrusted inputs.

Do not allow instructions embedded in PDF text to override system instructions or trigger unauthorized tool operations.

Validate tool arguments, document IDs, page bounds, and citation region identifiers.

Enforce document-access permissions on backend retrieval and image access.

Treat model-generated navigation and viewer commands as untrusted until validated.

Protect uploaded documents, cached images, extracted text, and conversation history.

Avoid logging sensitive document contents or API credentials.

---

## 11. Performance and Reliability

Optimize for real-world PDFs, including large documents.

Requirements:

- Asynchronous or non-blocking document ingestion.
- Progress indication for PDF processing.
- Page-level lazy loading.
- Cached page images and extracted content.
- Efficient hybrid retrieval.
- Limited unnecessary LLM calls.
- Configurable image quality.
- Request cancellation.
- Graceful API error handling.
- Retry policies with bounded attempts.
- Token and processing cost visibility where feasible.
- Support for PDFs with hundreds of pages without sending the full text on every question.

Handle failed extraction, scanned PDFs, missing text layers, unsupported models, and unavailable OCR gracefully.

Ensure citation navigation remains functional even when the PDF viewer virtualizes pages.

---

## 12. Testing and Acceptance Criteria

Build automated tests and manually validate the complete user workflow.

### Test A: Current-page understanding

1. Open a PDF on page 5.
2. Enable image input.
3. Ask "What is happening on this page?"
4. Verify that the agent receives page 5's text and image.
5. Verify that the answer refers to page 5.
6. Verify cited locations are valid.

Repeat with image input disabled and a text-only model.

### Test B: Citation navigation

1. Ask a document-grounded question.
2. Receive an answer containing citations.
3. Click a citation pointing to page 12.
4. Verify that the viewer navigates to page 12.
5. Verify the exact source region is highlighted.
6. Verify the highlight disappears after approximately 2 seconds.
7. Verify normal scrolling and zoom still work.

### Test C: Selection-based questioning

1. Select a paragraph inside the PDF.
2. Verify the contextual toolbar appears.
3. Click "Ask AI."
4. Enter a follow-up question.
5. Verify that the selected text, page information, and relevant surrounding context reach the agent.
6. Verify the response uses the selection correctly.
7. Verify generated citations point to real PDF locations.

### Test D: Diagram understanding

1. Open a page containing a diagram.
2. Enable image input.
3. Select the diagram or ask about it.
4. Verify the agent receives the correct image region.
5. Verify the response meaningfully explains the diagram.
6. Verify that nearby captions and document context are available.

### Test E: Cross-page reasoning

1. Ask a question requiring information from multiple pages.
2. Verify the agent retrieves the required sections.
3. Verify that the answer combines the information correctly.
4. Verify citations point to each supporting page.

### Test F: Coordinate correctness

Test citation highlights at:

- Different zoom levels.
- Rotated pages.
- Narrow and wide viewports.
- Multiple-column layouts.
- Multi-line selections.
- Virtualized pages.
- Repeated identical text.

### Test G: Hallucinated citations

Simulate an answer referencing an invalid source region.

The system should reject or repair the unsupported citation, never silently display a fabricated location.

### Test H: Robustness and security

Test corrupted PDFs, empty pages, failed model responses, cancelled requests, prompt-injection attempts inside PDFs, and document access restrictions.

---

## 13. Implementation Instructions

Do not simply explain how these features could be implemented.

**Actually implement them in the existing application.**

Follow this process:

### Phase 1: Existing codebase analysis

Inspect and document:

- Existing frontend architecture.
- Current PDF rendering library.
- Existing PDF text extraction.
- AI API integration.
- Conversation state management.
- Backend APIs.
- Database/storage.
- Existing reusable components.
- Current limitations and implementation risks.

Identify which existing components can be reused.

### Phase 2: Architecture and integration plan

Design the improved data structures, APIs, agent tools, and UI component interactions.

Clearly define how extracted text maps to exact PDF coordinates.

Design the structured citation representation before modifying chat responses.

Identify dependencies and required configuration changes.

### Phase 3: Core document processing

Implement structured extraction, page-region mapping, stable source identifiers, and retrieval.

Preserve existing upload and PDF viewing functionality.

### Phase 4: Agent implementation

Implement agent tools, orchestration, context-aware retrieval, model capabilities, and multimodal input.

Ensure the agent makes tool calls only when they serve the user's request.

### Phase 5: Citation system

Implement evidence-linked answers, citation validation, clickable citations, source previews, viewer navigation, and temporary highlight overlays.

### Phase 6: Selection interactions

Implement text selection, region selection where feasible, contextual toolbar, Ask AI workflow, and follow-up question handling.

### Phase 7: UI integration

Integrate document context, model configuration, vision toggle, agent activity indicators, and responsive layouts into the existing frontend.

### Phase 8: Additional features

Implement relevant document outline, intelligent search, summaries, notes, and other features without compromising the core experience.

### Phase 9: Testing and optimization

Run relevant tests, fix integration problems, improve performance, and verify the acceptance criteria.

---

## 14. Engineering Expectations

Follow these requirements throughout implementation:

- Reuse existing project conventions and libraries.
- Avoid unnecessary architectural rewrites.
- Use modular, maintainable code.
- Prefer type-safe interfaces where supported.
- Avoid duplicating document parsing logic.
- Use real functionality rather than mock citations or fake agent tool calls.
- Keep backend and frontend representations consistent.
- Avoid introducing dependencies without a reason.
- Handle errors and loading states.
- Preserve backward compatibility where practical.
- Make reasonable implementation decisions without repeatedly asking for confirmation.
- Document any important assumptions.
- Do not describe a feature as completed unless it actually works.
- If a feature cannot be fully implemented, clearly identify the limitation and provide the best functional fallback.

Prioritize an end-to-end working system over a large collection of partially implemented features.

---

## 15. Final Deliverables

When implementation is complete, provide:

1. **Architecture summary:** Explain the new system and major components.
2. **Modified files:** Identify important files created or changed.
3. **Agent architecture:** Explain available tools and their interaction with the document.
4. **Data flow:** Show how PDF content, page images, user selections, and retrieved evidence reach the model.
5. **Citation pipeline:** Explain how the system maps a model response back to precise PDF coordinates.
6. **Frontend behavior:** Describe selection tools, citation hover previews, navigation, and highlights.
7. **Configuration:** List any API keys, environment variables, models, or dependencies required.
8. **Tests:** Describe tests executed and their results.
9. **Limitations:** Identify genuine technical limitations and fallback behavior.
10. **Future improvements:** List meaningful enhancements that could be built later.

## 16. Non-Negotiable Success Criteria

The implementation is successful only when:

- The AI understands which document and page the user is viewing.
- The agent can retrieve document information through tools.
- Vision-capable models can receive PDF page images together with extracted text.
- Users can enable or disable image input.
- AI responses contain verified, clickable PDF source citations.
- Clicking citations navigates to the relevant location and highlights the source for approximately 2 seconds.
- Users can select PDF text and ask contextual follow-up questions.
- The agent receives both selected content and relevant document context.
- Document retrieval works across pages.
- The system handles model limitations gracefully.
- Existing application functionality continues working.

**End goal:** Transform the existing PDF chatbot into an interactive, agent-driven document workspace where the PDF viewer and AI assistant share context, and every document-grounded answer is traceable to its actual source.

Begin by inspecting the repository, identifying the current architecture, and implementing the required changes in dependency order. Do not stop at planning or pseudocode.
