# Implementation Plan: Falcon Transparent Inference

## Overview

Implement the full Falcon transparent context assembly and inference system on top of the existing working core. The work is organized into eight areas: config extension, memory rewrite, engine payload assembly and truncation, background memory extractor, identity validation, audit completeness, full app.py UI additions, and Hypothesis property-based tests covering all 19 correctness properties.

All code is Python. UI is Streamlit. Database is MongoDB (via pymongo / mongomock for tests). PBT framework is Hypothesis.

---

## Tasks

- [x]  `falcon/config.py` with new settings and fail-fast validation
  - [x] 1.1 Add new config fields and validators to `falcon/config.py`
    - Add `top_k_per_type` (int, default 3, range [1, 20])
    - Add `recency_weight` (float, default 0.4) and `relevance_weight` (float, default 0.6)
    - Add `history_truncation_strategy` (str, one of `"last-n-turns"`, `"token-budget"`, `"summarize-and-compress"`)
    - Add `history_max_turns` (int, default 20, range [1, 100])
    - Add `history_token_budget` (int, default 4000, range [100, 200000])
    - Add `memory_extraction_enabled` (bool, default True)
    - Add `assistant_language_patterns` (list[str], default list per Req 16.5)
    - Raise `ValueError` at import for each out-of-range or invalid value
    - Update `config.yaml` with default values for all new keys
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 21.1, 21.2, 21.3, 9.3, 20.6, 19.9_

  - [x] 1.2 Write property test for config fail-fast validation (Property not in top-19 set — unit test only)  <!-- queued -->
    - Test each invalid config value produces the correct `ValueError` message
    - Test valid boundary values are accepted without error
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 21.1, 21.2, 21.3_

- [x]  `falcon/memory.py` retrieval with weighted scoring and persona isolation
  - [x] 2.1 Update `MemoryType`, `_ALL_TYPES` constants, and `add_memory` validation in `falcon/memory.py`
    - Add `"persona"` and `"procedural"` to `MemoryType` literal and `_ALL_TYPES`
    - Update `add_memory` to accept and validate all six types; raise `ValueError` for unknown types
    - _Requirements: 9.1, 9.2_

  - [x] 2.2 Write property test for invalid memory type rejection
    - **Property 6: Invalid memory type raises `ValueError`**
    - **Validates: Requirements 9.1**
    - Use `@given(st.text())` filtered to exclude valid type names; assert `ValueError` raised and no MongoDB document persisted (use mongomock)

  - [x] 2.3 Update `RetrievalResult` dataclass in `falcon/memory.py`
    - Add `by_type: dict[str, list[dict]]` field
    - Add `total_found: int` field
    - Ensure `entries`, `reasoning`, `by_type`, `total_found` are all present
    - _Requirements: 9.6, 9.7_

  - [x] 2.4 Implement weighted scoring algorithm in `retrieve_for_generation`
    - Rewrite `retrieve_for_generation` signature: add `top_k_per_type`, `recency_weight`, `relevance_weight` params
    - For each active type: fetch all non-archive entries for `identity_id`, compute `recency_rank_score` = 1/(rank+1) normalised to [0,1]
    - Compute `overlap_score`: pinned → 1.0; tag overlap → tag_count/len(tags); keyword match → ratio; else 0.0
    - Assign `match_reason`: `"pinned"` > `"tag-match"` > `"keyword-match"` > `"recency"` (priority order)
    - `final_score = (recency_rank_score * recency_weight) + (overlap_score * relevance_weight)`
    - Sort descending, take top `top_k_per_type` per type; populate `score` and `match_reason` on each entry
    - Populate `reasoning` list: one string per non-persona entry: `"<type>/<id>: score=<score>, reason=<match_reason>"`
    - Persona: always prepend to `entries` if exists; not scored; not in `reasoning`
    - Archive: never returned
    - All queries carry `{"identity_id": identity_id}` filter
    - _Requirements: 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 20.1, 20.3, 20.4, 20.8_

  - [x] 2.5 Write property test for identity isolation in retrieval
    - **Property 7: Retrieval never crosses identity boundary**
    - **Validates: Requirements 9.8, 12.2**
    - Generate two distinct identity IDs and memory stores; assert no cross-identity entries in result

  - [x] 2.6 Write property test for `top_k_per_type` limit enforcement
    - **Property 8: `top_k_per_type` limit is enforced per type**
    - **Validates: Requirements 9.4**
    - Generate memory stores with > k entries per type; assert `len(entries_of_type) <= top_k_per_type`

  - [x] 2.7 Write property test for persona inclusion/absence
    - **Property 9: Persona always included, never scored, absent when missing**
    - **Validates: Requirements 9.5, 20.3, 20.4**
    - Two cases: identity with persona → always in entries; identity without → no persona entry in result

  - [x] 2.8 Write property test for reasoning cardinality
    - **Property 10: Reasoning cardinality matches non-persona entries**
    - **Validates: Requirements 9.7**
    - Assert `len(result.reasoning) == len([e for e in result.entries if e["memory_type"] != "persona"])`

  - [x] 2.9 Write property test for score and match_reason validity
    - **Property 11: Non-persona entries carry valid score and match_reason**
    - **Validates: Requirements 9.6**
    - Assert all non-persona entries have `score` in [0.0, 1.0] and `match_reason` in `{"pinned","tag-match","keyword-match","recency"}`

  - [x] 2.10 Implement `clear_working_memory` identity scope enforcement
    - Ensure `clear_working_memory(identity_id)` deletes only `working` entries for that identity
    - _Requirements: 9.2, 12.3_

  - [x] 2.11 Write property test for `clear_working_memory` identity scoping
    - **Property 12: `clear_working_memory` is identity-scoped**
    - **Validates: Requirements 12.3, 9.2**
    - Seed working memory for two identities; clear id_A; assert id_A has 0 working entries, id_B unchanged

- [x] 3. Checkpoint — memory layer complete
  - Ensure all memory tests pass, ask the user if questions arise.

- [x]  `build_annotated_payload` and history truncation in `falcon/engine.py`
  - [x] 4.1 Implement `build_annotated_payload` function in `falcon/engine.py`
    - Define `VALID_SOURCES = frozenset({"system-prompt","persona","memory","history","user-input","history-summary"})`
    - Implement payload ordering: persona → system-prompt → history-summary → memory → history → user-input
    - Annotate each element with correct `source` value per the mapping table in the design
    - Return `(annotated_payload, context_snapshot)` tuple
    - `context_snapshot` includes: `system_prompt`, `prompt_state`, `persona_block`, `memory_entries`, `history_included`, `history_dropped_turns`, `truncation_strategy`, `current_input`, `assembled_payload`, `annotated_payload`, `context_token_estimate`, `retrieval_timeout`, `retrieval_result`, `message_count`
    - When `system_prompt` is None/empty/whitespace → no system-prompt element in payload
    - When `system_prompt` is non-empty → first element (after any persona) has `role="system"`, `source="system-prompt"`, `content` byte-for-byte identical
    - _Requirements: 1.1, 1.2, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 20.2, 20.3, 20.4, 20.5_

  - [x] 4.2 Write property test for empty system prompt produces no system message
    - **Property 1: Empty system prompt produces no system message**
    - **Validates: Requirements 1.1**
    - `@given(st.one_of(st.none(), st.just(""), st.text(alphabet=" \t\n\r")))` → assert no element with `role=="system"` and `source=="system-prompt"`

  - [x] 4.3 Write property test for non-empty system prompt is byte-for-byte first system element
    - **Property 2: Non-empty system prompt is byte-for-byte first system element**
    - **Validates: Requirements 1.2**
    - `@given(st.text().filter(lambda s: s.strip()))` → assert first system element has `source=="system-prompt"` and `content==system_prompt`

  - [x] 4.4 Write property test for annotated payload structural completeness
    - **Property 4: Annotated payload structural completeness**
    - **Validates: Requirements 3.1, 3.2**
    - `@given(...)` with arbitrary inputs → every element has exactly keys `{"role","content","source"}` and `source` in `VALID_SOURCES`

  - [x] 4.5 Write property test for source annotation correctness
    - **Property 5: Source annotation correctness**
    - **Validates: Requirements 3.3, 3.4, 3.5, 3.6, 20.5**
    - Assert each element's `source` value matches its semantic origin (persona→"persona", system→"system-prompt", etc.)

  - [x] 4.6 Implement `"last-n-turns"` truncation strategy in `build_annotated_payload`
    - Keep the most recent `history_max_turns` turn-pairs; record `history_dropped_turns`
    - Raise `ValueError` for invalid `truncation_strategy` value
    - _Requirements: 21.1, 21.4, 21.7_

  - [x] 4.7 Write property test for `last-n-turns` truncation
    - **Property 17: `last-n-turns` truncation includes exactly the right number of turn-pairs**
    - **Validates: Requirements 21.4, 21.7**
    - `@given(history=..., max_turns=st.integers(1,100))` → assert `len(included_pairs) <= max_turns` and `dropped == max(0, total - max_turns)`

  - [x] 4.8 Implement `"token-budget"` truncation strategy in `build_annotated_payload`
    - Token estimate: `len(content) // 4`
    - Keep newest turns fitting within `history_token_budget`; if single pair exceeds budget, include it alone
    - _Requirements: 21.5, 21.7_

  - [x] 4.9 Write property test for `token-budget` truncation
    - **Property 18: `token-budget` truncation keeps total history tokens within budget**
    - **Validates: Requirements 21.5**
    - `@given(history=..., budget=st.integers(100,200000))` → assert total token estimate of included history ≤ budget (except single-pair exception)

  - [x] 4.10 Implement `"summarize-and-compress"` truncation strategy in `build_annotated_payload`
    - Keep most recent `history_max_turns` turns verbatim; prepend summary message with `source="history-summary"`
    - On summary LLM call failure: fall back to `"last-n-turns"` and record fallback in snapshot
    - _Requirements: 21.6, 21.7, 3.6_

  - [x] 4.11 Implement 500 ms retrieval timeout enforcement in `build_annotated_payload` / `stream_inference`
    - Wrap `retrieve_for_generation` call with 500 ms timeout (threading or `concurrent.futures`)
    - On timeout: log WARNING with elapsed time, set `retrieval_timeout=True` in snapshot, proceed with empty memory block
    - Ensure no synchronous MongoDB writes during streaming phase
    - _Requirements: 22.5, 22.6, 22.9_

  - [x] 4.12 Implement `[no output]` marker for empty model responses in `stream_inference`
    - When model response is empty or whitespace: yield `"[no output]"` as sole token; set `raw_output = "[no output]"`
    - _Requirements: 2.1, 2.2_

  - [x] 4.13 Write property test for empty model output marker
    - **Property 3: Empty model output yields `[no output]` marker**
    - **Validates: Requirements 2.1, 2.2**
    - `@given(st.one_of(st.just(""), st.text(alphabet=" \t\n\r")))` as mock model output → assert `raw_output == "[no output]"` and `"[no output]"` was yielded

- [x] 5. Checkpoint — engine layer complete
  - Ensure all engine tests pass, ask the user if questions arise.

- [x] 6. Implement `falcon/memory_extractor.py` background extraction agent
  - [x] 6.1 Create `falcon/memory_extractor.py` with per-identity queue management
    - Define `_extractor_queues: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))`
    - Implement `run(turn_snapshot: dict) -> None`: receives immutable dict with `identity_id`, `user_message`, `assistant_message`, `turn_index`, `timestamp`
    - Call configured LLM via OpenRouter with minimal extraction prompt requesting JSON output classifying facts into `semantic`/`episodic`/`procedural`/`working` types
    - Validate extracted entries before calling `Memory_Module.add_memory` with `source="auto"`
    - On LLM call failure: log ERROR with `identity_id` and turn index, exit silently
    - On malformed JSON: catch `json.JSONDecodeError`, log ERROR, persist zero entries
    - On MongoDB write failure: catch, log ERROR, do not re-raise
    - On any uncaught exception at thread boundary: catch, log ERROR, do not crash main thread
    - When queue depth ≥ 10 for identity: drop task, log WARNING with `identity_id` and turn index
    - Never write `persona` or `archive` entries; never hold reference to mutable Engine state
    - All persisted entries carry `source="auto"` and `identity_id` exactly matching `turn_snapshot["identity_id"]`
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 22.3_

  - [x] 6.2 Write property test for memory extractor entry constraints
    - **Property 16: Memory extractor never writes persona or archive entries**
    - **Validates: Requirements 19.3, 19.4, 19.5**
    - Mock LLM call to return controlled JSON; assert no persisted entry has `memory_type in {"persona","archive"}`; all have `source=="auto"` and correct `identity_id`

- [x]  `falcon/identity.py` forbidden character validation
  - [x] 7.1 Verify and extend `_validate_identity_id` in `falcon/identity.py`
    - Confirm `forbidden_chars = frozenset({"/", "\\", "..", "\x00"})` is enforced
    - Confirm `_validate_identity_id` raises `ValueError` for any identity_id containing a forbidden char
    - Confirm `load_history` calls `_validate_identity_id` before any MongoDB query
    - Ensure `load_history` returns only messages with `identity_id == id_A` (no cross-identity leakage)
    - _Requirements: 12.1, 12.5, 12.6_

  - [x] 7.2 Write property test for forbidden character validation
    - **Property 14: Forbidden character in `identity_id` raises `ValueError`**
    - **Validates: Requirements 12.6**
    - `@given(st.text().filter(lambda s: any(c in s for c in {"/","\\","..","\\x00"})))` → assert `ValueError` raised by `load_history`

  - [x] 7.3 Write property test for `load_history` identity scoping
    - **Property 13: `load_history` is identity-scoped**
    - **Validates: Requirements 12.1**
    - Seed messages for id_A and id_B in mock MongoDB; assert `load_history(id_A)` returns only id_A messages

- [x]  `falcon/audit.py` audit record field enforcement
  - [x] 8.1 Update `build_audit_record` signature and field validation in `falcon/audit.py`
    - Add all required parameters: `identity_id`, `model`, `prompt_state`, `system_prompt`, `retrieved_memories`, `generation_settings`, `context_size`, `context_token_estimate`, `assembled_payload`, `raw_model_output`, `usage`, `latency_ms`
    - Raise `ValueError` if any required field is missing
    - Confirm `read_audit_records` filters by `identity_id`, returns newest-first
    - _Requirements: 14.1, 14.2, 14.4_

  - [x] 8.2 Write property test for audit record field completeness
    - **Property 15: Audit record contains all required fields**
    - **Validates: Requirements 14.1**
    - `@given(...)` with randomly generated valid inputs → assert returned dict contains all 13 required keys

- [x] 9. Checkpoint — backend modules complete
  - Ensure all backend tests pass, ask the user if questions arise.

- [x]  additions in `app.py`
  - [x] 10.1 Add system prompt toggle, text area, and reset button to sidebar in `app.py`
    - Toggle labelled "System prompt" (ON/OFF); caption "OFF — no platform prompt injected" / "ON — prompt active"
    - Text area displaying and allowing live edit of system prompt text
    - "Reset to default" button restoring `config.default_system_prompt` value
    - When toggle is OFF: pass empty string as `system_prompt` to engine
    - Default prompt text from `config.yaml` as per Req 1.8
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [x] 10.2 Add model selector dropdown to sidebar in `app.py`
    - Dropdown from `config.available_models`; update only `selected_model` in session state on change
    - Display currently active model name at all times
    - Error if selected model not in `available_models`; do not initiate inference
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 13.1, 13.4_

  - [x] 10.3 Add truncation strategy dropdown and limit input to sidebar in `app.py`
    - Dropdown for `history_truncation_strategy` (three options)
    - Numeric input showing `history_max_turns` or `history_token_budget` depending on selected strategy
    - When strategy changes: update visible limit input to the relevant parameter
    - Token counter display (loaded on identity init; updated after inference; persisted to MongoDB `tokens` collection)
    - _Requirements: 21.9, 21.10, 15.1, 15.2, 15.3, 15.4_

  - [x] 10.4 Implement identity switcher with message counts in `app.py`
    - Display message count for each listed identity
    - On switch: reload history, tokens, and traces for new identity; do not carry over session state from previous identity
    - _Requirements: 12.4, 12.5_

- [x]  additions in `app.py`
  - [x] 11.1 Add per-message "⌥ context" button in Chat tab
    - Each assistant message row has a button that calls `_show_payload_dialog` with that turn's context snapshot
    - _Requirements: 3.7, 4.4_

  - [x] 11.2 Add assistant-language warning banner in Chat tab
    - When system-prompt toggle is OFF and model output contains any pattern from `config.assistant_language_patterns`: display banner for that turn
    - _Requirements: 17.5, 16.5_

- [x]  (color-coded source annotation, dropped-turn placeholder, export) in `app.py`
  - [x] 12.1 Extend `_render_context_view` with color-coded source annotation
    - Apply distinct visual styling per source category: `system-prompt`, `persona`, `user-input`, `history`, `memory`, `history-summary`
    - Persona block labelled "Persona" visually separated from memory and system prompt blocks
    - "Platform / injected" label for `system-prompt` and `memory` elements; "User" for `user-input`; "History" for `history`
    - _Requirements: 3.7, 3.8, 4.1, 4.2, 4.3, 4.4, 20.7_

  - [x] 12.2 Render dropped-turn placeholder in Context tab
    - When `history_dropped_turns > 0`: render `▸ [N turns truncated]` placeholder element visually distinct from included turns
    - _Requirements: 21.8_

  - [x] 12.3 Add "Export context snapshot" button to Context tab
    - Download full `context_snapshot` dict as JSON with `falcon_export_version: "1"` envelope; all ObjectIds as strings
    - _Requirements: 10.4, 10.5, 10.6, 10.7_

- [x]  full CRUD in `app.py`
  - [x] 13.1 Implement Memory tab section display and empty-state messages
    - Display all six sections: Persona, Semantic, Episodic, Procedural, Working, Archive
    - Show `content`, `tags`, `created_at`, `memory_type` per entry; empty-state message when section empty
    - Display "Automatic memory extraction is disabled" banner when `memory_extraction_enabled=False`
    - _Requirements: 18.1, 18.2, 19.10_

  - [x] 13.2 Implement inline Edit / Save / Cancel for memory entries
    - "Edit" button per entry; shows inline text inputs for `content` and `tags`; "Save" and "Cancel" buttons
    - On Save: call `Memory_Module` update; reflect change immediately; on failure display inline error and retain form
    - _Requirements: 18.3, 18.4, 18.11_

  - [x] 13.3 Implement Delete action with confirmation modal for memory entries
    - "Delete" button per entry; confirmation modal before MongoDB delete
    - On confirmed success: remove entry from displayed list immediately
    - On failure: display inline error
    - _Requirements: 18.5, 18.11_

  - [x] 13.4 Implement "Clear type" button with confirmation modal
    - "Clear type" button per section (excluding Persona)
    - Confirmation modal; on success display empty-state message; on failure display inline error
    - _Requirements: 18.6, 18.11_

  - [x] 13.5 Implement "Add entry" form per memory type section
    - Form with `content` (max 10,000 chars) and `tags` fields
    - Validation error if `content` empty; do not call `Memory_Module`
    - On success: persist with `source="manual"`, display in correct section within 2 seconds; on failure retain form contents and show error
    - _Requirements: 18.7, 18.8, 18.11_

  - [x] 13.6 Implement Persona section with individual field editing
    - Individual text input fields for `name`, `tone`, `communication_style`, `core_traits`
    - Empty fields and prompt to create when no persona record exists
    - On Save: update or create persona record; error message and retain values on failure
    - _Requirements: 18.9, 18.10, 18.11_

  - [x] 13.7 Implement "Test retrieval" input and display in Memory tab
    - Input field and button; on click call `Memory_Module.retrieve_for_generation` with query and active `identity_id`
    - Display full `RetrievalResult` including per-entry `score`, `match_reason`, `memory_type`; no inference triggered
    - _Requirements: 9.9_

  - [x] 13.8 Add "Export memory" button to Memory tab
    - Download all memory entries for active identity as JSON with `falcon_export_version: "1"` envelope; all ObjectIds as strings
    - _Requirements: 10.2, 10.5, 10.6, 10.7_

- [x]  and Logs tab export buttons in `app.py`
  - [x] 14.1 Add "Export audit log" button to Audit tab
    - Download all audit records for active identity as JSON with `falcon_export_version: "1"` envelope; all ObjectIds as strings
    - _Requirements: 10.3, 10.5, 10.6, 10.7_

  - [x] 14.2 Add "Export conversation" button to Logs tab
    - Download full conversation history for active identity as JSON with `falcon_export_version: "1"` envelope; all ObjectIds as strings
    - _Requirements: 10.1, 10.5, 10.6, 10.7_

- [x]  post-generation pipeline in `app.py` / `_handle_send`
  - [x] 15.1 Refactor `_handle_send` to dispatch all post-generation tasks as background threads
    - After stream exhaustion: launch audit write, token persistence, `Memory_Extractor.run`, and logger persistence as non-blocking background tasks within 100 ms
    - Return control to UI render loop before background tasks complete
    - Each background task catches all exceptions, logs at ERROR with `identity_id` and task name, never surfaces to UI
    - Token counter: load on identity init; update after inference; persist to MongoDB `tokens` collection
    - Pass immutable `turn_snapshot` dict to `Memory_Extractor.run`; do not pass mutable engine references
    - When `memory_extraction_enabled=False`: do not launch extractor task
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.7, 22.9, 19.1, 19.6, 19.7, 19.9, 15.2, 15.3_

- [x]  serialisation utility and write property test for export round-trip
  - [x] 16.1 Implement export JSON envelope serialisation helper
    - `make_export_envelope(identity_id, data) -> dict` producing `{"falcon_export_version":"1","exported_at":..., "identity_id":..., "data":...}`
    - Serialise all MongoDB ObjectId fields as strings (use `str(oid)`)
    - Used by all four export buttons (conversation, memory, audit, context)
    - _Requirements: 10.5, 10.6, 10.7_

  - [x] 16.2 Write property test for export JSON round-trip
    - **Property 19: Export JSON is a valid round-trip serialization**
    - **Validates: Requirements 10.5, 10.6, 10.7**
    - `@given(...)` with random Falcon data sets → call `make_export_envelope`; parse with `json.loads`; assert `falcon_export_version == "1"` and no ObjectId instances in output

- [x] 17. Checkpoint — full integration
  - Ensure all tests pass, ask the user if questions arise.

- [x] 
  - [x] 18.1 Write integration test: full `_handle_send` flow
    - Test message logged, audit written, tokens updated against real test-database-isolated MongoDB instance
    - _Requirements: 14.2, 15.2, 22.3_

  - [x] 18.2 Write integration test: `Memory_Extractor.run` persists entries correctly
    - Assert entries persisted with correct `identity_id` and `source="auto"`
    - _Requirements: 19.5, 19.7_

  - [x] 18.3 Write integration test: identity switch — no state bleed
    - Switch identity; assert no history, memory, or token state from previous identity bleeds into new session
    - _Requirements: 12.4_

- [x]  — all tests pass
  - Ensure all property-based, unit, and integration tests pass. Ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for full traceability
- Checkpoints ensure incremental validation at each layer boundary
- Property tests use Hypothesis with `@settings(max_examples=100)` per design specification
- Unit tests cover concrete scenarios; property tests verify universal invariants — neither replaces the other
- All MongoDB interactions in tests use mongomock (in-memory) unless the task explicitly states a real test database
- `build_annotated_payload` is the primary payload assembly path; `build_payload` retained for backward compatibility
- Token estimation throughout uses `len(content) // 4` (consistent with existing `context_token_estimate`)
- The `summarize-and-compress` strategy requires a live LLM call for summary generation; mock in unit tests, use real call in integration tests

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "2.3"] },
    { "id": 2, "tasks": ["2.2", "2.4", "7.1", "8.1"] },
    { "id": 3, "tasks": ["2.5", "2.6", "2.7", "2.8", "2.9", "2.10", "4.1", "7.2", "7.3", "8.2"] },
    { "id": 4, "tasks": ["2.11", "4.2", "4.3", "4.4", "4.5", "4.6", "4.8", "4.10", "4.11", "4.12", "6.1"] },
    { "id": 5, "tasks": ["4.7", "4.9", "4.13", "6.2", "16.1"] },
    { "id": 6, "tasks": ["10.1", "10.2", "10.3", "10.4", "11.1", "11.2", "12.1", "12.2", "12.3", "13.1", "14.1", "14.2", "15.1", "16.2"] },
    { "id": 7, "tasks": ["13.2", "13.3", "13.4", "13.5", "13.6", "13.7", "13.8"] },
    { "id": 8, "tasks": ["18.1", "18.2", "18.3"] }
  ]
}
```
