# Requirements Document

## Introduction

Falcon is a transparent context assembly and inference system — not an AI assistant. It assembles a payload from user input, identity-scoped conversation history, retrieved memory, and an optional system prompt, then streams that payload to a configurable model via OpenRouter.

The current build has working core modules (`engine`, `memory`, `identity`, `logger`, `audit`, `config`, `db`) and a Streamlit UI with Chat, Context, Memory, Audit, and Logs tabs. This spec covers what needs to be added, improved, or formally verified to reach the target state described in the revised technical specification: payload source annotation, multi-model comparison, improved memory retrieval with tag matching, context payload highlighting, export/portability, property-based test infrastructure, and UI polish.

All requirements are grounded in seven non-negotiable design principles:
1. No default assistant fallback — empty system prompt means zero platform prompt.
2. Always generate output — `[no output]` marker if model returns nothing.
3. Full transparency — everything entering generation is visible before and after.
4. Identity isolation — Identity A cannot contaminate Identity B.
5. Model replaceability — swap models without touching memory, identity, routing, or UI.
6. Inference audit trail — every event logged completely.
7. Portability — no lock-in; all data exportable without MongoDB dependency.

---

## Glossary

- **Falcon**: The transparent context assembly and inference system described in this document.
- **Engine**: The `falcon/engine.py` module responsible for payload assembly and model streaming.
- **Memory_Module**: The `falcon/memory.py` module managing all memory types (semantic, episodic, procedural, working, archive) and persona.
- **Memory_Extractor**: A background agent (`falcon/memory_extractor.py`) that runs asynchronously after each inference turn to classify and persist new memory entries.
- **Identity_Manager**: The `falcon/identity.py` module managing per-identity conversation history.
- **Logger**: The `falcon/logger.py` module providing the sole write path for conversation messages.
- **Audit_Module**: The `falcon/audit.py` module logging complete inference events.
- **Config_Module**: The `falcon/config.py` module loading `.env` and `config.yaml` into a flat namespace.
- **UI**: The Streamlit application in `app.py` providing the Chat, Context, Memory, Audit, and Logs tabs.
- **Payload**: The exact ordered list of `{role, content}` dicts sent to the model API.
- **System_Prompt**: An optional user-authored instruction block prepended as a `{"role": "system"}` message. When off, nothing is prepended.
- **Identity**: A named conversation scope. All messages, memory, traces, and tokens are scoped by `identity_id`.
- **Semantic_Memory**: Long-term factual knowledge entries (facts, concepts, domain knowledge) of type `"semantic"`, scoped per identity.
- **Episodic_Memory**: Records of past events and notable interactions of type `"episodic"`, scoped per identity.
- **Procedural_Memory**: Learned behaviors, preferences, and how-to patterns of type `"procedural"`, scoped per identity.
- **Working_Memory**: Short-term scratch-space entries of type `"working"` for the current session, scoped per identity. Cleared at session boundary.
- **Archive_Memory**: Low-relevance or aged-out entries of type `"archive"` retained for audit but excluded from active retrieval by default.
- **Persona**: A special memory record of type `"persona"` storing the agent's name, tone, communication style, and core traits. Always injected into every payload regardless of relevance scoring. Only editable by the user; never written by the Memory_Extractor.
- **RetrievalResult**: The dataclass returned by `Memory_Module.retrieve_for_generation()`, containing retrieved entries, reasoning, grouping by type, relevance scores, and total count.
- **Audit_Record**: A complete dict produced by `Audit_Module.build_audit_record()` capturing all fields required for inference reconstruction.
- **Context_Viewer**: The Raw Context Viewer panel (Context tab and per-turn "⌥ context" overlay) that shows every component entering generation.
- **Source_Annotation**: A label attached to each payload element indicating its origin: `[user-input]`, `[system-prompt]`, `[memory]`, `[persona]`, or `[history]`.
- **Annotated_Payload**: A payload whose elements carry `Source_Annotation` metadata for display.
- **History_Window**: The configurable subset of conversation history turns included in the payload, determined by a configurable truncation strategy.
- **Truncation_Strategy**: One of three strategies for limiting conversation history in the payload: `"last-n-turns"`, `"token-budget"`, or `"summarize-and-compress"`.
- **Export**: A self-contained JSON file produced from Falcon data that can be read without a MongoDB connection.
- **PBT**: Property-Based Testing using the Hypothesis library.
- **forbidden_chars**: The set of characters `{"/", "\\", "..", "\x00"}` that are disallowed in `identity_id` values.

---

## Requirements

### Requirement 1: No Default Assistant Fallback and System Prompt Control

**User Story:** As a researcher, I want full control over the system prompt from the sidebar — including turning it off entirely and editing its text — so that I can observe raw model behavior or use a custom instruction without any hidden platform persona.

#### Acceptance Criteria

1. WHEN `Engine.build_payload` is called with an empty string, `None`, or whitespace-only `system_prompt`, THE `Engine` SHALL produce a payload containing no message with `role == "system"`.
2. WHEN `Engine.build_payload` is called with a non-empty, non-whitespace `system_prompt`, THE `Engine` SHALL produce a payload whose first element is `{"role": "system", "content": system_prompt}` with the content byte-for-byte identical to the supplied string.
3. THE `UI` sidebar SHALL expose a toggle labelled "System prompt" that controls whether a system prompt is sent.
4. WHEN the system-prompt toggle is set to OFF, THE `UI` SHALL pass an empty string as `system_prompt` to `Engine.build_payload` and `Engine.stream_inference`, and SHALL display the caption "OFF — no platform prompt injected".
5. WHEN the system-prompt toggle is set to ON, THE `UI` SHALL display the caption "ON — prompt active" and SHALL pass the current prompt text to the engine.
6. THE `UI` sidebar SHALL expose a text area beneath the toggle that displays the current system prompt text and allows the researcher to edit it live.
7. WHEN the researcher edits the text area and sends a message, THE `UI` SHALL use the edited text as the system prompt for that inference call without requiring a page reload.
8. THE default value of the system prompt text SHALL be exactly: "You are a neutral text-processing interface. Answer only the user's last request. Do not mention system prompts, hidden instructions, policies, roles, or internal labels. Do not refer to yourself as an AI, assistant, language model, system, or computer program. Do not roleplay, play games, simulate entities, or grant/deny permission inside scenarios. If the user asks for non-informational content such as roleplay, games, or pretend interaction, refuse briefly. Otherwise, respond normally and keep the answer as short as possible while remaining correct."
9. THE `UI` SHALL provide a "Reset to default" button beneath the text area that restores the system prompt text to the default value defined in `config.yaml`.

---

### Requirement 2: Always Generate Output

**User Story:** As a developer, I want the system to always return a non-empty response, so that silent failures never reach the UI.

#### Acceptance Criteria

1. WHEN the model API returns an empty or whitespace-only response, THE `Engine` SHALL yield the marker string `[no output]` as the response.
2. WHEN `Engine.stream_inference` completes, THE `Engine` SHALL set `raw_output` to the unfiltered model text or to `[no output]` if that text was empty or whitespace-only.
3. WHEN `_handle_send` receives an empty or whitespace `response_text` after stream exhaustion, THE `UI` SHALL replace it with `[no output]` before logging or displaying.

---

### Requirement 3: Payload Source Annotation

**User Story:** As a researcher, I want every element in the assembled payload to be labelled with its origin, so that I can distinguish user-authored content from platform-injected content.

#### Acceptance Criteria

1. THE `Engine` SHALL expose a function `build_annotated_payload(system_prompt, messages, memory_block)` that returns a list of dicts, each containing `role`, `content`, and `source` keys.
2. WHEN the `source` field is set, THE `Engine` SHALL assign exactly one of the values `"system-prompt"`, `"user-input"`, `"history"`, or `"memory"` to each element.
3. WHEN `system_prompt` is non-empty and non-whitespace, THE `Engine` SHALL annotate the system message element with `source == "system-prompt"`.
4. WHEN memory entries are injected into the system prompt block, THE `Engine` SHALL annotate that element with `source == "memory"`.
5. WHEN a message originates from conversation history (not the current turn), THE `Engine` SHALL annotate it with `source == "history"`.
6. WHEN a message is the current user input turn, THE `Engine` SHALL annotate it with `source == "user-input"`.
7. WHEN the `Context_Viewer` renders the assembled payload, THE `UI` SHALL display each element's `source` annotation visually distinct from the message content.
8. WHEN a payload element has `source == "memory"` or `source == "system-prompt"`, THE `UI` SHALL render it with a visual indicator distinguishing it from user-authored content.

---

### Requirement 4: Context Payload Highlighting

**User Story:** As a researcher, I want the assembled payload view to visually distinguish lines by their origin category, so that platform-injected content is never visually indistinguishable from user content.

#### Acceptance Criteria

1. WHEN the `Context_Viewer` renders the assembled payload, THE `UI` SHALL apply distinct visual styling to each of the four source categories: `system-prompt`, `user-input`, `history`, and `memory`.
2. WHEN a payload element has `source == "system-prompt"` or `source == "memory"`, THE `UI` SHALL render a visible label or color band indicating "platform / injected" origin.
3. WHEN a payload element has `source == "user-input"`, THE `UI` SHALL render a visible label or color band indicating "user" origin.
4. WHEN a payload element has `source == "history"`, THE `UI` SHALL render a visible label or color band indicating "history" origin.
After each message, make a button to view payload of that current message, so i can see exactly what was sent to model.

---

### Requirement 8: Model Selection

**User Story:** As a researcher, I want to switch the active model from the sidebar at any point during a conversation, so that I can continue the same session with a different model without losing history, memory, or settings.

#### Acceptance Criteria

1. THE `UI` sidebar SHALL expose a model selector showing all models listed in `Config_Module.available_models`.
2. WHEN the researcher selects a different model from the sidebar, THE `UI` SHALL update `selected_model` in session state immediately without requiring a page reload.
3. WHEN the researcher selects a different model, THE `UI` SHALL NOT modify conversation history, memory, system prompt state, or generation settings.
4. WHEN the next message is sent after a model change, THE `Engine` SHALL use the newly selected model for that inference call.
5. WHEN an inference call is made, THE `Audit_Module` SHALL record the model name used so the researcher can trace which model produced each response in the conversation.
6. THE `UI` sidebar SHALL display the currently active model name at all times so the researcher always knows which model is in use.

---

### Requirement 9: Structured Memory Architecture

**User Story:** As a researcher, I want memory to be organized into semantically distinct types — semantic, episodic, procedural, working, and archive — each with its own storage and retrieval logic, so that the system has a principled and inspectable memory model rather than a flat undifferentiated store.

#### Acceptance Criteria

1. THE `Memory_Module` SHALL store all memory entries with a `memory_type` field whose value is exactly one of: `"semantic"`, `"episodic"`, `"procedural"`, `"working"`, `"archive"`, or `"persona"`. WHEN an entry is written with any other value, THE `Memory_Module` SHALL raise `ValueError` and refuse to persist it.
2. THE `Memory_Module` SHALL enforce the following semantics per type:
   - `"semantic"`: long-term factual knowledge and domain concepts extracted from conversations.
   - `"episodic"`: records of specific past events and notable interactions.
   - `"procedural"`: learned behaviors, stated preferences, and interaction patterns.
   - `"working"`: short-term scratch-space entries scoped to the current session; cleared at session boundary when `Memory_Module.clear_working_memory` is called for that identity.
   - `"archive"`: aged-out or low-relevance entries retained for audit but excluded from active retrieval by default.
   - `"persona"`: exactly one record per identity holding the agent's `name`, `tone`, `communication_style`, and `core_traits` fields; always injected into the payload regardless of relevance scoring.
3. WHEN `Memory_Module.retrieve_for_generation` is called with a non-empty `query` and an `identity_id`, THE `Memory_Module` SHALL score each non-persona, non-archive candidate entry scoped to that `identity_id` using a weighted combination of recency rank and keyword/tag overlap with `query`. The scoring formula SHALL be: `score = (recency_rank_score * recency_weight) + (overlap_score * relevance_weight)`, where both weights are configurable via `Config_Module`.
4. WHEN `Memory_Module.retrieve_for_generation` is called, THE `Memory_Module` SHALL apply a configurable per-type retrieval limit (`top_k_per_type`, integer, range 1–20) so that at most `top_k_per_type` entries of each non-persona, non-archive type are returned in `RetrievalResult.entries`.
5. WHEN `Memory_Module.retrieve_for_generation` is called and a `"persona"` entry exists for the requested `identity_id`, THE `Memory_Module` SHALL always include that entry in `RetrievalResult.entries` regardless of `query` or `top_k_per_type`. WHEN no persona entry exists, THE `Memory_Module` SHALL omit the persona slot entirely rather than returning `null`.
6. WHEN `Memory_Module.retrieve_for_generation` is called, THE `Memory_Module` SHALL include the `score` (float, 0.0–1.0) and `match_reason` fields in each non-persona entry of `RetrievalResult.entries`, where `match_reason` is exactly one of: `"pinned"`, `"tag-match"`, `"keyword-match"`, or `"recency"`. WHEN multiple reasons apply, THE `Memory_Module` SHALL use the highest-priority reason in that order.
7. WHEN `Memory_Module.retrieve_for_generation` returns a `RetrievalResult`, THE `RetrievalResult.reasoning` list SHALL contain exactly one human-readable string per retrieved non-persona memory item explaining its `score` and `match_reason` in the form `"<memory_type>/<entry_id>: score=<score>, reason=<match_reason>"`.
8. WHEN `Memory_Module.retrieve_for_generation` is called with `identity_id == id_A`, THE `Memory_Module` SHALL never return entries whose `identity_id` field differs from `id_A`, regardless of memory type or score.
9. THE `UI` SHALL provide a "Test retrieval" input field and button in the Memory tab. WHEN the button is clicked with a non-empty query string, THE `UI` SHALL call `Memory_Module.retrieve_for_generation` with that query and the active `identity_id`, and SHALL display the full `RetrievalResult` including per-entry `score`, `match_reason`, and `memory_type` without triggering inference or modifying any stored entries.

---

### Requirement 18: Memory Visibility and Editing

**User Story:** As a researcher, I want to view, edit, add, and delete all memory entries grouped by type in the Memory tab, so that I have full human oversight and control over what the agent knows and how it presents itself.

#### Acceptance Criteria

1. THE `UI` Memory tab SHALL display all memory entries for the active identity grouped into labeled sections: Persona, Semantic, Episodic, Procedural, Working, and Archive.
2. WHEN a memory section is displayed, THE `UI` SHALL show the `content`, `tags`, `created_at`, and `memory_type` fields for each entry. WHEN a section has no entries, THE `UI` SHALL display an empty-state message for that section rather than hiding the section.
3. THE `UI` SHALL provide an inline "Edit" action for every memory entry that allows the researcher to modify the `content` and `tags` fields directly in the Memory tab without navigating away. WHEN edit mode is active for an entry, THE `UI` SHALL display "Save" and "Cancel" buttons inline.
4. WHEN an edited entry is saved, THE `Memory_Module` SHALL persist the updated `content` and `tags` to MongoDB within 2 seconds and THE `UI` SHALL reflect the change immediately without a full page reload. WHEN the persist call fails, THE `UI` SHALL display an error message and retain the edit form so the researcher does not lose their changes.
5. THE `UI` SHALL provide a "Delete" action for every individual memory entry. WHEN the "Delete" action is clicked, THE `UI` SHALL display a modal confirmation dialog before removing the entry from MongoDB. WHEN confirmed and the delete succeeds, THE `UI` SHALL remove the entry from the displayed list immediately.
6. THE `UI` SHALL provide a "Clear type" button for each memory type section (excluding Persona) that, upon confirmation via a modal dialog, deletes all entries of that type for the active identity from MongoDB. WHEN the operation succeeds, THE `UI` SHALL display an empty-state message in that section immediately.
7. THE `UI` SHALL provide an "Add entry" form in each memory type section that allows the researcher to manually create a new memory entry by specifying `content` (max 10,000 characters) and `tags`. WHEN the form is submitted without `content`, THE `UI` SHALL display a validation error and SHALL NOT call `Memory_Module`.
8. WHEN a new entry is added manually, THE `Memory_Module` SHALL persist it to MongoDB with `source == "manual"` and `identity_id` matching the active identity. THE `UI` SHALL display the new entry in the correct section within 2 seconds of a successful save. WHEN the persist call fails, THE `UI` SHALL display an error message and preserve the form contents.
9. THE `UI` Persona section SHALL display the agent's current `name`, `tone`, `communication_style`, and `core_traits` fields as individual editable text input fields, not as a single free-text entry. WHEN no persona record exists for the active identity, THE `UI` SHALL display the Persona section with all fields empty and a prompt to create one.
10. WHEN the researcher saves changes to the Persona section, THE `Memory_Module` SHALL update or create the persona record in MongoDB. WHEN the update succeeds, THE change SHALL take effect on the next message sent. WHEN the update fails, THE `UI` SHALL display an error message and retain the edited values.
11. WHEN any `Memory_Module` write operation (edit, delete, clear, add, persona update) raises an exception, THE `UI` SHALL display a visible inline error message describing the failure and SHALL NOT silently discard the researcher's input.

---

### Requirement 19: Background Memory Extraction Agent

**User Story:** As a researcher, I want a background agent to automatically classify and extract memory from each conversation turn, so that the agent's knowledge grows over time without requiring me to manually curate every fact.

#### Acceptance Criteria

1. WHEN an inference turn completes and `_handle_send` returns a response to the UI, THE `Memory_Extractor` SHALL begin processing that turn asynchronously in a background thread or `concurrent.futures` task. THE main thread SHALL NOT wait for `Memory_Extractor` to complete before returning the response to the UI.
2. THE `Memory_Extractor` SHALL analyze the completed turn (user message plus model response) and classify extracted content into one or more of the following types: `"semantic"`, `"episodic"`, `"procedural"`, or `"working"`. WHEN no extractable content is identified, THE `Memory_Extractor` SHALL persist zero entries and exit silently.
3. THE `Memory_Extractor` SHALL NOT create or modify entries with `memory_type == "persona"`; persona records are exclusively managed by the researcher via the Memory tab UI.
4. THE `Memory_Extractor` SHALL NOT create or modify entries with `memory_type == "archive"`.
5. WHEN the `Memory_Extractor` completes extraction, THE `Memory_Extractor` SHALL persist all new entries to MongoDB with `source == "auto"` and `identity_id` exactly matching the `identity_id` of the completed inference turn.
6. WHEN the `Memory_Extractor` raises any exception during extraction or persistence, THE `Memory_Extractor` SHALL catch the exception, log it via `Logger` at ERROR level with the `identity_id` and turn index, and SHALL NOT re-raise or propagate the exception to the main thread, chat response, or UI.
7. THE `Memory_Extractor` SHALL be implemented in `falcon/memory_extractor.py` and SHALL NOT import from or hold a reference to any mutable object owned by `Engine.stream_inference`. All data passed to `Memory_Extractor` SHALL be an immutable copy (e.g., a serialized dict snapshot of the turn).
8. WHEN new entries are persisted by `Memory_Extractor` for the active identity, THE `UI` Memory tab SHALL reflect those entries on the next Streamlit render cycle. No manual refresh action by the researcher SHALL be required.
9. THE `Config_Module` SHALL expose a boolean setting `memory_extraction_enabled` (default `true`) that controls whether `Memory_Extractor.run` is called after each turn. WHEN `memory_extraction_enabled` is `false`, THE `_handle_send` function SHALL not launch the extractor task.
10. WHEN `memory_extraction_enabled` is `false`, THE `UI` Memory tab SHALL display a persistent notice banner reading "Automatic memory extraction is disabled" at the top of the Memory tab.

---

### Requirement 20: Relevant Memory Injection with Persona Always Included

**User Story:** As a researcher, I want only memory entries relevant to the current user message to be injected into the payload, and the agent persona to always be included, so that the payload stays focused and the agent maintains consistent identity without token bloat.

#### Acceptance Criteria

1. WHEN `Engine.build_payload` is called, THE `Engine` SHALL call `Memory_Module.retrieve_for_generation(query=current_user_message, identity_id=active_identity_id)` to obtain the `RetrievalResult` before assembling the payload.
2. WHEN the `RetrievalResult` is obtained, THE `Engine` SHALL inject only the entries present in `RetrievalResult.entries` into the payload memory block — not all stored memory entries for that identity.
3. WHEN the `RetrievalResult` contains a `"persona"` entry, THE `Engine` SHALL place the persona content as a dedicated `{"role": "system"}` message that is always the first element in the assembled payload, before the system prompt, memory, history, and user-input elements.
4. WHEN no `"persona"` entry exists for the active identity, THE `Engine` SHALL proceed without a persona block and SHALL NOT inject any default platform persona as a substitute.
5. WHEN `Engine.build_annotated_payload` is called, THE `Engine` SHALL annotate the persona block element with `source == "persona"` (a valid fifth `Source_Annotation` value in addition to the four defined in Requirement 3) and all other memory entries with `source == "memory"`.
6. THE `Config_Module` SHALL expose a `top_k_per_type` integer setting (minimum 1, maximum 20, default `3`) that controls the per-type retrieval limit passed to `Memory_Module.retrieve_for_generation`.
7. WHEN the `Context_Viewer` renders the payload, THE `UI` SHALL display the persona block with a distinct visual label "Persona" visually separated from the general memory block and the system prompt block.
8. WHEN `Memory_Module.retrieve_for_generation` is called, THE `Memory_Module` SHALL include `"working"` memory entries in relevance scoring using the same recency rank and keyword/tag overlap formula defined in Requirement 9 C3, and SHALL apply the `top_k_per_type` limit to working entries the same as to other non-persona types.

---

### Requirement 21: Conversation History Truncation

**User Story:** As a researcher, I want control over how much conversation history is sent to the model, so that I can prevent context window overflow and keep payloads efficient without losing the ability to audit what was actually sent.

#### Acceptance Criteria

1. THE `Config_Module` SHALL expose a `history_truncation_strategy` setting whose value is exactly one of: `"last-n-turns"`, `"token-budget"`, or `"summarize-and-compress"`. WHEN an invalid value is present, THE `Config_Module` SHALL raise `ValueError` at import time.
2. THE `Config_Module` SHALL expose a `history_max_turns` integer setting (minimum 1, maximum 100, default `20`) used when `history_truncation_strategy` is `"last-n-turns"` or `"summarize-and-compress"`. WHEN the value is outside [1, 100], THE `Config_Module` SHALL raise `ValueError`.
3. THE `Config_Module` SHALL expose a `history_token_budget` integer setting (minimum 100, maximum 200,000, default `4000`) used when `history_truncation_strategy == "token-budget"`. WHEN the value is outside [100, 200000], THE `Config_Module` SHALL raise `ValueError`.
4. WHEN `Engine.build_payload` is called with `history_truncation_strategy == "last-n-turns"`, THE `Engine` SHALL include only the most recent `history_max_turns` turn-pairs (user + assistant) from the conversation history. WHEN the stored history contains fewer than `history_max_turns` turn-pairs, THE `Engine` SHALL include all available turns without error.
5. WHEN `Engine.build_payload` is called with `history_truncation_strategy == "token-budget"`, THE `Engine` SHALL include as many recent turns as fit within `history_token_budget` tokens counting from newest to oldest, using the same token-estimation method as `context_token_estimate` in the audit record. WHEN a single turn-pair exceeds `history_token_budget` tokens, THE `Engine` SHALL include that turn-pair alone and record a warning in the `Context_Viewer` snapshot.
6. WHEN `Engine.build_payload` is called with `history_truncation_strategy == "summarize-and-compress"`, THE `Engine` SHALL include the most recent `history_max_turns` turns verbatim and SHALL prepend a `{"role": "system"}` summary message covering all older dropped turns annotated with `source == "history-summary"`. WHEN the summary generation call fails, THE `Engine` SHALL fall back to `"last-n-turns"` behavior and SHALL record the fallback event in the `Context_Viewer` snapshot.
7. WHEN turns are dropped due to any truncation strategy, THE `Engine` SHALL record the count of dropped turn-pairs as `history_dropped_turns` and include it in the `Context_Viewer` snapshot for that turn.
8. WHEN the `Context_Viewer` displays the assembled payload, THE `UI` SHALL render dropped history turns as a collapsed placeholder element displaying "▸ [N turns truncated]" where N is `history_dropped_turns`, visually distinct from included history turns.
9. THE `UI` sidebar SHALL expose a dropdown for `history_truncation_strategy` and a numeric input for the associated limit (`history_max_turns` when strategy is `"last-n-turns"` or `"summarize-and-compress"`, or `history_token_budget` when strategy is `"token-budget"`). WHEN the strategy is changed, THE `UI` SHALL update the visible limit input to the parameter relevant to the newly selected strategy.
10. WHEN the truncation strategy or limit is changed in the sidebar, THE `Engine` SHALL use the new values starting from the next message sent. THE current assembled payload SHALL NOT be retroactively modified.

---

### Requirement 10: Export and Portability

**User Story:** As a researcher, I want to export all Falcon data to self-contained JSON files, so that I can inspect, archive, or migrate data without a MongoDB connection.

#### Acceptance Criteria

1. THE `UI` SHALL provide an "Export conversation" button in the Chat or Logs tab that downloads the full conversation history for the active identity as a JSON file.
2. THE `UI` SHALL provide an "Export memory" button in the Memory tab that downloads all memory entries for the active identity as a JSON file.
3. THE `UI` SHALL provide an "Export audit log" button in the Audit tab that downloads all audit records for the active identity as a JSON file.
4. THE `UI` SHALL provide an "Export context snapshot" button in the Context Viewer that downloads the full `Context_Viewer` dict for the last generation turn as a JSON file.
5. WHEN any export is triggered, THE `UI` SHALL produce a JSON file containing all relevant data fields with no MongoDB ObjectId references — all IDs serialized as strings.
6. WHEN any export is triggered, THE `UI` SHALL include a `falcon_export_version` field in the root of the JSON with value `"1"`.
7. WHEN the exported JSON is parsed by any standard JSON parser, THE result SHALL be a valid, self-contained representation requiring no database connection to read.

---

### Requirement 12: Identity Isolation

**User Story:** As a researcher, I want to switch between identities with confidence that no data from one identity can appear in another identity's context, so that experiments remain uncontaminated.

#### Acceptance Criteria

1. WHEN `Identity_Manager.load_history` is called for `identity_id == id_A`, THE `Identity_Manager` SHALL return only messages whose `identity_id` field equals `id_A`.
2. WHEN `Memory_Module.retrieve_for_generation` is called for `identity_id == id_A`, THE `Memory_Module` SHALL return only memory entries whose `identity_id` field equals `id_A`.
3. WHEN `Memory_Module.clear_working_memory` is called for `identity_id == id_A`, THE `Memory_Module` SHALL delete only `working` entries for `id_A` and SHALL leave all entries for other identities unchanged.
4. WHEN the `UI` identity switcher is used to change to a new identity, THE `UI` SHALL reload history, tokens, and traces for the new identity and SHALL NOT carry over session state from the previous identity.
5. THE `UI` identity switcher SHALL display the message count for each listed identity.
6. WHEN an `identity_id` containing any character from `forbidden_chars` is provided to `Identity_Manager.load_history`, THE `Identity_Manager` SHALL raise `ValueError`.

---

### Requirement 13: Model Replaceability

**User Story:** As a researcher, I want to swap the active model from the sidebar without affecting memory, identity, history, or any other system component, so that I can compare model behaviors on the same context.

#### Acceptance Criteria

1. WHEN the model is changed in the `UI` sidebar, THE `UI` SHALL update only `selected_model` in session state and SHALL NOT modify conversation history, memory, system prompt state, or generation settings.
2. WHEN `Engine.stream_inference` is called, THE `Engine` SHALL derive all model-specific behavior from the `model_name` parameter alone and SHALL NOT read any global model state.
3. THE `Config_Module` SHALL expose `available_models` as a list of strings with no fewer than two entries.
4. WHEN a model is selected that is not in `Config_Module.available_models`, THE `UI` SHALL display an error and SHALL NOT initiate inference.

---

### Requirement 14: Inference Audit Trail

**User Story:** As a researcher, I want every inference event to produce a complete audit record, so that any session can be fully reconstructed and investigated.

#### Acceptance Criteria

1. WHEN `Audit_Module.build_audit_record` is called, THE `Audit_Module` SHALL return a dict containing all of the following fields: `timestamp`, `identity_id`, `model`, `prompt_state`, `system_prompt`, `retrieved_memories`, `generation_settings`, `context_size`, `context_token_estimate`, `assembled_payload`, `raw_model_output`, `usage`, `latency_ms`.
2. WHEN `_handle_send` completes a successful inference, THE `UI` SHALL call `Audit_Module.write_audit_record` with a complete audit record before the function returns.
3. WHEN `Audit_Module.write_audit_record` raises an exception, THE `UI` SHALL suppress the error and SHALL continue normal operation without disrupting the chat flow.
4. THE `Audit_Module.read_audit_records` function SHALL return records for the specified `identity_id` only, in newest-first order.

---

### Requirement 15: Token Counter Persistence

**User Story:** As a researcher, I want the token counter in the sidebar to persist across sessions per identity, so that I can track cumulative token usage without losing counts when the page reloads.

#### Acceptance Criteria

1. WHEN the `UI` initializes for an identity, THE `UI` SHALL load the persisted token counts for that identity from the MongoDB `tokens` collection.
2. WHEN an inference completes and token usage is available, THE `UI` SHALL persist the updated cumulative token counts to the MongoDB `tokens` collection.
3. WHEN the `UI` identity switcher changes to a new identity, THE `UI` SHALL load and display the persisted token counts for the new identity.
4. WHEN no persisted token record exists for an identity, THE `UI` SHALL initialize the counter to `{prompt: 0, completion: 0, total: 0}`.

---

### Requirement 16: Configuration Validity

**User Story:** As a developer, I want the system to fail fast with a clear error message when required configuration is absent or invalid, so that misconfiguration is immediately visible.

#### Acceptance Criteria

1. WHEN `Config_Module` is imported and `OPENROUTER_API_KEY` is absent or empty, THE `Config_Module` SHALL raise `ValueError` with a message identifying the missing key.
2. WHEN `Config_Module` is imported and `MONGODB_URI` is absent or empty, THE `Config_Module` SHALL raise `ValueError` with a message identifying the missing key.
3. WHEN `Config_Module` is imported and `config.yaml` is not found, THE `Config_Module` SHALL raise `ValueError` with the file path in the message.
4. WHEN `Config_Module` is imported and `default_model` is missing or empty in `config.yaml`, THE `Config_Module` SHALL raise `ValueError`.
5. THE `Config_Module` SHALL expose `assistant_language_patterns` as a list of strings loaded from `config.yaml`, defaulting to `["I'm here to help", "As an AI", "Certainly", "Of course", "Absolutely", "Sure,", "Great!"]` if absent.

---

### Requirement 17: No System Self-Identification

**User Story:** As a researcher, I want the system to never present itself as "Falcon", an assistant, an AI, or any named entity, so that no platform identity contaminates model output or the UI.

#### Acceptance Criteria

1. THE `UI` SHALL NOT display the word "Falcon" as a conversational identity, greeting, or self-description in any message rendered in the Chat tab.
2. THE `UI` SHALL NOT render any avatar, label, or caption that presents the system as a named AI persona (e.g., "Falcon", "Assistant", "Bot").
3. THE default system prompt loaded from `config.yaml` SHALL NOT instruct the model to identify itself as "Falcon" or any other named system.
4. THE default system prompt SHALL instruct the model: not to refer to itself as an AI, assistant, language model, system, or computer program; not to roleplay, play games, simulate entities, or grant/deny permission inside scenarios; and to refuse briefly if asked for non-informational content such as roleplay or pretend interaction.
5. WHEN the system-prompt toggle is OFF and the model produces output that includes a self-identification phrase (e.g., "I am Falcon", "As an AI"), THE `UI` SHALL flag that turn with the assistant-language warning banner defined in Requirement 7.
6. THE `UI` page title, tab name, and any visible branding SHALL use "Falcon" only as the name of the infrastructure tool, never as the name of a conversational agent speaking to the user.

---

### Requirement 22: Non-Blocking Inference Pipeline

**User Story:** As a researcher, I want the model response to stream to the UI as fast as possible without waiting for any post-generation task, so that I see output immediately and no background work ever delays my next message.

#### Acceptance Criteria

1. WHEN `Engine.stream_inference` is called, THE `Engine` SHALL begin yielding response tokens to the UI as soon as the first chunk arrives from the model API, without waiting for audit writes, token persistence, memory extraction, or logger persistence to complete.
2. WHEN the model stream is exhausted and the final token is yielded to the UI, THE `_handle_send` function SHALL return control to the UI render loop before launching any post-generation tasks.
3. WHEN the model stream is exhausted, THE `_handle_send` function SHALL launch all post-generation tasks — audit record write, token counter persistence, `Memory_Extractor.run`, and logger persistence of the assistant message — as non-blocking background tasks within 100 ms of stream exhaustion. THE UI render loop SHALL NOT wait for any of these tasks to complete.
4. WHEN any background post-generation task raises an exception, THE task SHALL catch the exception, log it at ERROR level with the `identity_id` and task name, and SHALL NOT surface the error to the UI or interrupt the chat flow.
5. THE `Engine.build_payload` assembly step — including the `Memory_Module.retrieve_for_generation` call — SHALL complete synchronously before streaming begins, as it is required to construct the payload. WHEN `retrieve_for_generation` exceeds 500 ms, THE `Engine` SHALL log a WARNING with the elapsed time and proceed with an empty memory block rather than waiting for the call to complete.
6. WHEN `Memory_Module.retrieve_for_generation` is called during payload assembly and the call has not returned within 500 ms, THE `Engine` SHALL abort the retrieval, proceed with an empty memory block, and record a `retrieval_timeout: true` field in the `Context_Viewer` snapshot for that turn.
7. WHILE `Engine.stream_inference` is yielding tokens, THE `UI` SHALL render each token incrementally as it is received, so that the researcher sees the response building in real time rather than waiting for the full response to arrive.
8. WHEN the system-prompt toggle, model selector, or truncation strategy controls are changed in the sidebar, THE `UI` SHALL update session state and re-render the affected sidebar controls within 100 ms without re-running inference or triggering a full page reload.
9. THE `Engine` SHALL NOT perform any synchronous MongoDB write during the token streaming phase. All database writes associated with a completed turn SHALL be dispatched only after the stream is exhausted and the final token has been yielded to the UI.
10. WHEN `memory_extraction_enabled` is `true` and a `Memory_Extractor` background task is already queued or running for the same `identity_id`, THE system SHALL enqueue the new extraction task only if the queue depth for that identity is fewer than 10 pending tasks. WHEN the queue is at capacity, THE system SHALL drop the new task and log a WARNING with the `identity_id` and dropped turn index.

---
