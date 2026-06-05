# Implementation Plan: Falcon V1

## Overview

Implement a neutral LLM communication interface using Python, Streamlit, LangChain, and Groq. The flat five-module architecture (config → engine → logger/identity → app) is built bottom-up so each layer is tested before the next is added.

## Tasks

- [ ] 1. Set up project structure and dependencies
  - Create `falcon/` package directory with `__init__.py`
  - Create `logs/` directory (or let logger auto-create it)
  - Create `requirements.txt` with pinned versions for `streamlit>=1.35.0`, `langchain>=0.2.0`, `langchain-groq>=0.1.5`, `langchain-core>=0.2.0`, `python-dotenv>=1.0.0`, `PyYAML>=6.0`, `hypothesis`
  - Create `.env.example` with `GROQ_API_KEY=your_key_here`
  - Create `config.yaml` with `default_model`, `available_models`, `default_system_prompt: ""`, `log_dir: "logs"`
  - _Requirements: 5.2, 5.3_

- [ ] 2. Implement `config.py`
  - [x] 2.1 Implement Config module
    - Load `.env` via `python-dotenv` and `config.yaml` via `PyYAML`
    - Raise `ValueError` with a human-readable message if `GROQ_API_KEY` is absent, empty, or whitespace-only
    - Raise `ValueError` if `default_model` or `log_dir` is missing, empty, or non-string
    - Expose flat namespace: `GROQ_API_KEY`, `default_model`, `available_models`, `default_system_prompt` (must be `""`), `log_dir`
    - _Requirements: 5.1, 5.2, 5.3_

  - [ ] 2.2 Write property test for Config completeness (Property 5)
    - **Property 5: Config Completeness**
    - **Validates: Requirements 5.1, 5.2, 5.3**
    - Use `hypothesis` to generate arbitrary env dicts; assert `ValueError` is raised whenever `GROQ_API_KEY` is absent/empty/whitespace
    - Assert that a successful load always exposes non-empty `default_model`, non-empty `log_dir`, and `default_system_prompt == ""`

  - [ ] 2.3 Write unit tests for `config.py`
    - Test missing key raises `ValueError` with actionable message
    - Test valid config loads all fields correctly from a temp YAML
    - Test `default_system_prompt` is always `""`
    - _Requirements: 5.1, 5.2, 5.3_

- [ ] 3. Implement `logger.py`
  - [x] 3.1 Implement Logger module
    - Implement `append_message(identity_id: str, role: str, content: str) -> None`
    - Validate `identity_id` for `/`, `\`, `..`, null bytes — raise `ValueError` if found
    - Validate `role` is `"user"` or `"assistant"` — raise `ValueError` otherwise, do not write
    - Auto-create `logs/` directory if absent
    - Read existing file (start with `[]` if absent); raise `json.JSONDecodeError` if file exists but is not valid JSON — do not overwrite
    - Append entry with exactly three fields: `timestamp` (ISO 8601 UTC), `role`, `content`; write full file back
    - _Requirements: 1.7, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 8.1, 8.2_

  - [ ] 3.2 Write property test for Log Integrity (Property 3)
    - **Property 3: Log Integrity**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    - Use `hypothesis` to generate arbitrary sequences of `(role, content)` pairs (role in `{"user", "assistant"}`)
    - After each `append_message`, assert the file parses as valid JSON
    - After `N` appends, assert `load_history` returns exactly `N` entries in insertion order
    - Assert no prior entry's `timestamp`, `role`, or `content` is mutated by a subsequent append

  - [ ] 3.3 Write unit tests for `logger.py`
    - Test two-message round-trip: append twice, assert count and field values
    - Test `role` validation raises `ValueError` and does not write
    - Test corrupted log raises `json.JSONDecodeError` and leaves file unchanged
    - Test `logs/` directory auto-creation
    - _Requirements: 3.1–3.8_

- [ ] 4. Implement `identity.py`
  - [x] 4.1 Implement Identity Manager module
    - Implement `list_identities() -> list[str]` — scan `logs/*.json`, return identity IDs derived from filenames
    - Implement `load_history(identity_id: str) -> list[dict]` — return `[]` for missing file; raise `json.JSONDecodeError` for unparseable file; return entries in chronological order; return a copy, not a reference
    - Implement `clear_identity(identity_id: str) -> None` — delete/empty log file; no-op if absent
    - Validate `identity_id` for `/`, `\`, `..`, null bytes — raise `ValueError` if found
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 8.1, 8.2_

  - [ ] 4.2 Write property test for Identity Isolation (Property 1)
    - **Property 1: Identity Isolation**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
    - Use `hypothesis` to generate pairs of distinct identity IDs `A` and `B` with arbitrary message sequences
    - Assert `append_message(A, ...)` followed by `load_history(B)` returns the same result as `load_history(B)` alone
    - Assert `clear_identity(A)` does not alter `logs/B.json`
    - Assert `list_identities()` contains each identity ID exactly once

  - [ ] 4.3 Write unit tests for `identity.py`
    - Test `load_history` returns `[]` for non-existent identity
    - Test `load_history` raises `json.JSONDecodeError` for corrupted file without modifying it
    - Test `clear_identity` removes only the target identity file
    - Test `list_identities` reflects only existing log files
    - _Requirements: 1.1–1.7, 8.1, 8.2_

- [ ] 5. Checkpoint — core persistence layer complete
  - Ensure all tests pass for `config.py`, `logger.py`, and `identity.py`
  - Ask the user if questions arise before continuing.

- [ ] 6. Implement `engine.py`
  - [ ] 6.1 Implement Engine module
    - Implement `build_payload(system_prompt: str, messages: list[dict]) -> list[dict]`
      - Return `[{"role": "system", "content": system_prompt}, ...messages]` when `system_prompt` is non-empty and non-whitespace-only
      - Return `[...messages]` with no system entry when `system_prompt` is empty or whitespace-only
      - Do not modify, prefix, suffix, or normalize any `content` values
    - Implement `run_inference(model_name: str, system_prompt: str, messages: list[dict], groq_api_key: str) -> dict`
      - Convert `{role, content}` dicts to `SystemMessage`, `HumanMessage`, `AIMessage` objects
      - Instantiate `ChatGroq` with the given `model_name` and `groq_api_key` — no caching
      - Call `llm.invoke()` — no chains, no agents, no tools
      - Return `{"response": str, "raw_payload": list[dict]}` where `raw_payload` is the exact payload built by `build_payload`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3_

  - [ ] 6.2 Write property test for Payload Transparency (Property 2)
    - **Property 2: Payload Transparency**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
    - Use `hypothesis` to generate arbitrary `system_prompt` strings and `messages` lists
    - Assert: if `system_prompt` is non-empty, `raw_payload[0] == {"role": "system", "content": system_prompt}`
    - Assert: if `system_prompt` is empty or whitespace, no `{"role": "system", ...}` entry appears
    - Assert: all user/assistant messages appear in `raw_payload` in the same order with identical `content` values
    - Assert: `len(raw_payload) == len(messages) + (1 if system_prompt.strip() else 0)`

  - [ ] 6.3 Write property test for Neutrality (Property 4)
    - **Property 4: Neutrality**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.7**
    - Use `hypothesis` to generate arbitrary message contents (including whitespace-heavy strings)
    - Assert `build_payload("", messages)` returns exactly `len(messages)` entries with no system entry
    - Assert `build_payload(whitespace_only_prompt, messages)` returns exactly `len(messages)` entries with no system entry
    - Assert every `HumanMessage` and `AIMessage` content in the payload equals the input string with zero modification

  - [ ] 6.4 Write unit tests for `engine.py`
    - Mock `ChatGroq.invoke()` to test response/payload return shape without live API calls
    - Test `build_payload` with and without system prompt
    - Test `run_inference` returns correct dict structure
    - _Requirements: 2.1–2.5, 4.1–4.3_

- [ ] 7. Checkpoint — inference layer complete
  - Ensure all engine tests pass.
  - Ask the user if questions arise before continuing.

- [ ] 8. Implement `app.py` — Streamlit UI
  - [ ] 8.1 Implement session state initialization and identity management
    - Initialize session state on first load: `identity_id="default"`, `history=[]`, `system_prompt=""`, `selected_model=Config.default_model`, `last_payload=None`, `last_response=None`
    - On app start, call `load_history("default")` and populate `history` — zero-loss restart continuity
    - Render identity selector input in main area; on change, call `load_history(new_id)` and replace session history; if `json.JSONDecodeError`, display `st.error()` with file path and keep current identity
    - _Requirements: 1.9, 1.10, 4.5, 7.1, 7.2, 7.3, 9.1, 9.2, 9.3_

  - [ ] 8.2 Implement Send flow
    - Render chat history from session state
    - Render message input with no placeholder text; render Send button with label "Send" (no coaching copy)
    - On Send with non-empty message:
      1. Call `Logger.append_message(identity_id, "user", content)` first
      2. Build `messages` from full session history including the just-logged user message
      3. Call `Engine.run_inference(model, system_prompt, messages, api_key)`
      4. On success: call `Logger.append_message(identity_id, "assistant", response)`, update `history`, store `last_payload`/`last_response`, re-render
      5. On exception: display `st.error()` in main area, do not log assistant entry
    - Pass user input to engine with no modification
    - _Requirements: 1.9, 1.11, 4.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ] 8.3 Implement sidebar and auxiliary controls
    - Render model selector from `Config.available_models`; on change, update `selected_model` for next send
    - Render system prompt editor with no placeholder text (field displayed empty when `system_prompt == ""`); pass current value to engine without modification
    - Render raw payload panel using `st.json(last_payload)` after a successful send
    - Render raw response panel using `st.code(last_response)` after a successful send
    - Render "Clear Conversation" button with confirmation; on confirm, call `clear_identity(identity_id)` and reset `history` to `[]`
    - Render "Edit Logs" button that displays the file path and instructs user to edit manually then reload
    - UI labels are functional only — no coaching, steering, tooltips, or examples anywhere
    - _Requirements: 4.6, 4.8, 4.9, 6.3, 6.4, 6.7, 7.4_

  - [ ] 8.4 Write property test for Context Continuity (Property 6)
    - **Property 6: Context Continuity**
    - **Validates: Requirements 1.9, 1.10, 1.11**
    - Use `hypothesis` to generate sequences of messages for an identity
    - Write `N` messages to a temp log via `Logger.append_message`; simulate app restart by calling `load_history`; assert returned history has exactly `N` entries in insertion order
    - Assert no entry is dropped, truncated, or reordered across the simulated restart

  - [ ] 8.5 Write integration tests for full send flow
    - Without Streamlit: test that `Logger.append_message` → `Identity.load_history` → `Engine.run_inference` (mocked Groq) → `Logger.append_message` produces the correct log state and `raw_payload` shape
    - Test context continuity: write 10 messages, load history, assert engine receives all 10 plus the new user message (11 entries in `raw_payload` with system prompt absent)
    - Test identity isolation: write messages for `test_A` and `test_B`, assert neither load returns the other's data
    - _Requirements: 1.1, 1.9, 1.10, 1.11, 6.1, 6.2_

- [ ] 9. Final checkpoint — Ensure all tests pass
  - Run the full test suite; ensure all non-optional and all property tests pass.
  - Ask the user if questions arise before declaring the implementation complete.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use `hypothesis` and validate universal behavioral invariants
- Unit/integration tests validate specific examples and edge cases
- Checkpoints ensure incremental validation before moving to the next layer
- The design uses Python; all code examples and tests should be Python

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "3.1"] },
    { "id": 1, "tasks": ["2.2", "2.3", "3.2", "3.3", "4.1"] },
    { "id": 2, "tasks": ["4.2", "4.3", "6.1"] },
    { "id": 3, "tasks": ["6.2", "6.3", "6.4", "8.1"] },
    { "id": 4, "tasks": ["8.2", "8.3"] },
    { "id": 5, "tasks": ["8.4", "8.5"] }
  ]
}
```
