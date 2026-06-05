# Requirements Document

## Introduction

Falcon V1 is a neutral LLM communication interface — infrastructure, not an assistant. It provides a clean, transparent channel between a user and a language model via a Streamlit UI backed by Groq/LangChain inference. Every layer of each interaction is visible and auditable. There is no personality, no hidden prompt injection, and no default framing. The system is intentionally minimal: no database, no agents, no RAG, no streaming. This document captures the behavioral requirements derived from the approved design.

Two principles are foundational and intertwined:

1. **Context continuity**: An identity's complete conversation history is sent with every inference call. The model always has full accumulated context. Conversations resume transparently across app restarts with zero loss of history.

2. **Neutrality**: The system actively suppresses default assistant behavior — coaching, steering, helpfulness framing, personality overlays — at every layer: prompt, engine, and UI. No layer adds to what the user provides.

## Glossary

- **App**: The `app.py` Streamlit application — the single entry point and orchestrator of all modules
- **Config**: The `config.py` module — loads and validates runtime configuration from `.env` and `config.yaml`
- **Engine**: The `engine.py` module — thin LangChain/ChatGroq wrapper that builds payloads and calls the Groq API
- **Identity**: A named context (e.g., `"default"`, `"user_1"`) that scopes a conversation to its own isolated log file
- **Identity_Manager**: The `identity.py` module — manages per-identity log file access (list, load, clear)
- **Logger**: The `logger.py` module — appends messages to the JSON log file for a given identity
- **Session**: The active Streamlit session state, holding in-memory copies of history, system prompt, selected model, and last payload/response
- **Log_File**: A JSON array stored at `logs/{identity_id}.json` containing the full conversation history for one identity
- **Payload**: The exact list of `{role, content}` dicts sent to the Groq API, as built by `Engine.build_payload()`
- **System_Prompt**: A user-supplied string passed as the first message to the model; empty string means no system message is sent
- **GROQ_API_KEY**: The Groq API secret loaded from `.env`; required for any inference call

---

## Requirements

### Requirement 1: Identity Isolation and Context Continuity

**User Story:** As a user, I want each named identity to maintain a completely separate, persistent conversation history that is fully sent with every inference call, so that switching identities never exposes another identity's data, and every conversation resumes with complete context regardless of when it was last active.

#### Acceptance Criteria

1. WHEN `Logger.append_message(A, role, content)` is called, THE `Identity_Manager` SHALL return the same result for `load_history(B)` as it would have returned before the append, for any identity ID `B` that is distinct from `A`
2. WHEN `Identity_Manager.clear_identity(A)` is called, THE `Identity_Manager` SHALL leave the log file at `logs/B.json` unchanged for any identity ID `B` that is distinct from `A`
3. THE `Identity_Manager` SHALL return each identity ID exactly once in the result of `list_identities()`, regardless of how many messages that identity has logged
4. WHEN `Identity_Manager.load_history(identity_id)` is called for an identity that has no log file, THE `Identity_Manager` SHALL return an empty list
5. WHEN `Identity_Manager.load_history(identity_id)` is called, THE `Identity_Manager` SHALL return messages in strict chronological order, defined as the sequence in which `append_message` calls completed for that identity
6. WHEN `Identity_Manager.load_history(identity_id)` is called for an identity whose log file exists but is not parseable as valid JSON, THE `Identity_Manager` SHALL raise a `json.JSONDecodeError` without modifying the file
7. WHEN `Logger.append_message(identity_id, role, content)` or `Identity_Manager.load_history(identity_id)` is called with an `identity_id` that contains `/`, `\`, `..`, or null bytes, THE system SHALL raise a `ValueError` identifying the disallowed character(s)
8. THE `Identity_Manager` SHALL include an identity ID in the result of `list_identities()` if and only if `Logger.append_message` has been called at least once for that identity and its log file has not been subsequently cleared
9. WHEN the `App` sends a message for `identity_id`, THE `App` SHALL pass the complete history returned by `load_history(identity_id)` — all prior turns, in chronological order — as the `messages` argument to `Engine.run_inference()`; no prior turn SHALL be omitted, truncated, or reordered by the `App`
10. WHEN the `App` starts and an identity's log file already exists from a previous session, THE `App` SHALL load that identity's full history and resume the conversation exactly where it left off, with no loss of prior turns
11. WHEN the `App` loads an identity with a large history (hundreds of messages), THE `App` SHALL pass the full history to `Engine.run_inference()` without truncation; token-limit enforcement is the responsibility of the Groq API, not the `App`

---

### Requirement 2: Payload Transparency

**User Story:** As a user, I want to see the exact message payload sent to the model on every inference call, so that I can verify no hidden context or injected instructions are present.

#### Acceptance Criteria

1. WHEN `Engine.run_inference()` is called with a non-empty `system_prompt`, THE `Engine` SHALL return a `raw_payload` whose first entry is `{"role": "system", "content": system_prompt}` with the `content` value equal to `system_prompt` without any modification
2. WHEN `Engine.run_inference()` is called with an empty string or absent `system_prompt`, THE `Engine` SHALL return a `raw_payload` that contains no entry with `role == "system"`
3. WHEN `Engine.run_inference()` is called with a `messages` list, THE `Engine` SHALL return a `raw_payload` in which every entry with `role == "user"` or `role == "assistant"` appears in the same order and with `content` values identical to the corresponding entries in `messages`
4. WHEN `Engine.run_inference()` is called, THE `Engine` SHALL return a `raw_payload` whose total length equals `len(messages) + 1` when `system_prompt` is non-empty and `len(messages)` when `system_prompt` is empty or absent; if `messages` is an empty list and `system_prompt` is non-empty, the length SHALL be 1; if both are empty, the length SHALL be 0
5. WHEN `Engine.build_payload(system_prompt, messages)` is called with an empty `system_prompt`, THE `Engine` SHALL return a list with `len(messages)` entries and no entry with `role == "system"`

---

### Requirement 3: Log Integrity

**User Story:** As a user, I want the conversation log files to be valid, append-only JSON that I can open and edit in any text editor, so that my conversation history is durable, auditable, and recoverable.

#### Acceptance Criteria

1. WHEN `Logger.append_message(identity_id, role, content)` is called, THE `Logger` SHALL ensure `logs/{identity_id}.json` is parseable as valid JSON after the write completes
2. WHEN `Logger.append_message()` is called, THE `Logger` SHALL preserve all previously written entries in `logs/{identity_id}.json` without modifying their `timestamp`, `role`, or `content` fields
3. WHEN `Logger.append_message(identity_id, role, content)` is called `N` times for the same `identity_id`, THE `Logger` SHALL write exactly `N` entries such that `load_history(identity_id)` subsequently returns exactly `N` entries
4. THE `Logger` SHALL write each log entry with exactly three fields: `timestamp`, `role`, and `content` — no additional fields
5. THE `Logger` SHALL write the `timestamp` field of each log entry as an ISO 8601 UTC string (e.g., `"2025-06-05T14:22:01Z"`)
6. WHEN `Logger.append_message()` is called with a `role` value other than `"user"` or `"assistant"`, THE `Logger` SHALL raise a `ValueError` and not write any entry to the log file
7. WHEN `Logger.append_message()` is called and the `logs/` directory does not exist, THE `Logger` SHALL create the directory before writing
8. WHEN `Logger.append_message()` is called and the existing log file at `logs/{identity_id}.json` is not parseable as valid JSON, THE `Logger` SHALL raise a `json.JSONDecodeError` and not overwrite the existing file

---

### Requirement 4: Neutrality — Suppression of Default Assistant Behavior

**User Story:** As a user, I want the system to pass my inputs and prompts to the model exactly as I typed them, with no hidden injections, reformatting, coaching, or personality overlays at any layer, so that I have full and verifiable control over what the model receives and the model's behavior is shaped only by my explicit inputs.

#### Acceptance Criteria

1. WHEN `Engine.run_inference()` is called with `system_prompt == ""` or a whitespace-only `system_prompt`, THE `Engine` SHALL not include any `SystemMessage` in the LangChain message list passed to `ChatGroq.invoke()`
2. WHEN `Engine.run_inference()` is called with a non-empty `system_prompt`, THE `Engine` SHALL include exactly one `SystemMessage` in the LangChain message list with `content` equal to `system_prompt` without any modification, prefix, suffix, or whitespace normalization
3. WHEN `Engine.run_inference()` is called, THE `Engine` SHALL pass the `content` field of every `HumanMessage` and `AIMessage` to `ChatGroq.invoke()` with no modification, prefix, suffix, or whitespace normalization
4. WHEN the `App` sends a user message, THE `App` SHALL pass the user's input text to `Engine.run_inference()` as the `content` of the last user message without modification
5. WHEN the `App` initializes, THE `App` SHALL set the system prompt field to `Config.default_system_prompt`, which SHALL be an empty string
6. WHEN the user edits the system prompt field and clicks Send, THE `App` SHALL pass the current system prompt field value to `Engine.run_inference()` as `system_prompt` without modification
7. THE codebase SHALL contain no hardcoded string that injects assistant framing, helpfulness instructions, personality descriptors, or coaching language into any message sent to the model — at any layer (engine, app, config)
8. THE UI SHALL contain no text, placeholder, tooltip, or label that coaches, steers, or suggests how the user should phrase their inputs or what the model will do
9. WHEN `system_prompt` is empty and the `App` renders the system prompt field, THE `App` SHALL display the field as empty — no placeholder text, example content, or default suggestion SHALL be visible in the field

---

### Requirement 5: Configuration Completeness and Startup Validation

**User Story:** As a developer deploying Falcon, I want the system to fail immediately with a clear error when required configuration is missing, so that I can diagnose and fix configuration problems before the application enters an unusable state.

#### Acceptance Criteria

1. IF `GROQ_API_KEY` is absent from the environment (not set, empty string, or whitespace-only) at the time `Config` is imported or initialized, THEN THE `Config` SHALL raise a `ValueError` with a human-readable message that identifies the missing key and instructs the user to add it, before the application accepts any inference or UI request
2. WHEN `Config` loads successfully, THE `Config` SHALL expose `default_model` as a non-empty string, `log_dir` as a non-empty string, `default_system_prompt` as an empty string, and `available_models` as a list (which may be empty)
3. IF any required configuration field (`default_model` or `log_dir`) is absent, an empty string, or a non-string type in `config.yaml`, THEN THE `Config` SHALL raise a `ValueError` and not expose any partially-loaded configuration values

---

### Requirement 6: Chat Send Flow

**User Story:** As a user, I want to type a message and receive a model response in the chat interface, with both the user message and assistant response persisted to the log and the full payload visible in the sidebar, so that I can have a conversation while maintaining complete transparency.

#### Acceptance Criteria

1. WHEN the user clicks Send with a non-empty message, THE `App` SHALL call `Logger.append_message(identity_id, "user", content)` before calling `Engine.run_inference()`
2. WHEN `Engine.run_inference()` returns a response, THE `App` SHALL call `Logger.append_message(identity_id, "assistant", response)` and update the in-memory session history before re-rendering
3. WHEN a send completes without exception, THE `App` SHALL display the `raw_payload` returned by `Engine.run_inference()` in the sidebar payload panel using `st.json()`
4. WHEN a send completes without exception, THE `App` SHALL display the raw response text returned by `Engine.run_inference()` in the sidebar response panel using `st.code()`
5. IF `Engine.run_inference()` raises an exception, THEN THE `App` SHALL display an `st.error()` message in the main area and SHALL NOT log an assistant response entry for that call
6. IF `Engine.run_inference()` completes without an exception, THEN THE `App` SHALL NOT display any `st.error()` message for that call
7. WHEN the user changes the model selector, THE `App` SHALL use the newly selected model name on the next `Engine.run_inference()` call

---

### Requirement 7: Identity Switching

**User Story:** As a user, I want to switch between named identities in the UI, so that I can maintain separate, isolated conversations without restarting the application.

#### Acceptance Criteria

1. WHEN the user changes the `identity_id` input and no send operation is in progress, THE `App` SHALL load the history for the new identity and replace the session history with the returned messages; IF the new identity has no prior messages, THE `App` SHALL display an empty chat history
2. WHEN the user changes the `identity_id` input and a send operation is in progress, THE `App` SHALL not switch identity until the send completes
3. IF loading the new identity's history raises an exception, THEN THE `App` SHALL keep the current identity active and display an `st.error()` message identifying the failure
4. WHEN the user clicks "Clear Conversation" and confirms the action, THE `App` SHALL delete the log file for the current identity and reset the in-memory session history to an empty list

---

### Requirement 8: Path Traversal Prevention

**User Story:** As a system operator, I want identity IDs to be sanitized before use as filenames, so that malicious or malformed identity IDs cannot read or write files outside the `logs/` directory.

#### Acceptance Criteria

1. WHEN any function in `identity.py` or `logger.py` is called with an `identity_id` that contains `/`, `\`, `..`, or null bytes, THE function SHALL raise a `ValueError` before constructing any file path
2. IF an `identity_id` is rejected due to invalid characters, THEN THE system SHALL raise a `ValueError` whose message indicates which disallowed character(s) were detected

---

### Requirement 9: Error Handling for Corrupted Logs

**User Story:** As a user, I want to be informed when a log file is corrupted rather than having the application crash silently, so that I can take corrective action.

#### Acceptance Criteria

1. IF a log file at `logs/{identity_id}.json` exists but is not parseable as valid JSON when the `App` attempts to load it, THEN THE `App` SHALL display an `st.error()` message that includes the full file path
2. IF a `json.JSONDecodeError` occurs while loading a log file, THEN THE `App` SHALL display an empty chat history for that identity rather than crashing or displaying a partial history
3. IF a `json.JSONDecodeError` occurs while loading a log file, THEN THE `App` SHALL NOT overwrite or delete the corrupted file; the file SHALL remain on disk in its current state until the user takes explicit action
