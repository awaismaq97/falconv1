# Requirements Document

## Introduction

Falcon is a transparent context assembly and inference system. It is not a chatbot, assistant,
agent, or coach. Falcon is an inference substrate whose primary goal is the elimination of hidden
behavioral layers and automatic assistant-fallback behavior.

The platform is designed to answer a single research question: *which model introduces the least
unsolicited behavior when run under controlled, fully transparent, identical conditions?* To support
that goal, every component that affects generation — prompts, memory, retrieval results, routing,
context assembly, generation parameters — must be visible and inspectable. Nothing affecting
generation may be hidden.

Falcon runs locally (or against a hosted API), uses MongoDB for persistence, exposes a
Streamlit UI, and is managed entirely inside the `falcon` conda environment. All development,
testing, and runtime operations use that environment.

---

## Glossary

- **Falcon**: The transparent inference system described in this document.
- **Context_Assembler**: The component that combines system prompt, conversation history, retrieved
  memory, and documents into the final generation context before each inference call.
- **Inference_Engine**: The component that sends the assembled payload to the model and returns the
  streamed token output (`falcon/engine.py`).
- **Memory_Store**: The persistent, user-controlled store of episodic, semantic, working, identity,
  and archive memory entries (`falcon/memory.py`).
- **Audit_Log**: The append-only collection of inference event records stored in MongoDB
  (`falcon/audit.py`).
- **Identity**: An isolated conversation context with its own message history, memory, token
  counters, audit records, and trace snapshots.
- **Raw_Context_Viewer**: The UI component that displays, before each inference call, every
  component entering generation — system prompt, conversation history, retrieved memory, documents,
  and assembled payload.
- **Payload**: The exact `[{role, content}, …]` list sent to the model API.
- **Prompt_State**: The observable state of the system prompt — one of `present`, `empty`, `null`,
  or `deleted`.
- **Generation_Controls**: The five parameters that govern model output: temperature, top_p,
  repetition_penalty, max_tokens, stop_tokens.
- **Logger**: The component that records user and assistant messages to the `messages` MongoDB
  collection (`falcon/logger.py`).
- **Config_Loader**: The component that loads and validates `.env` and `config.yaml` at startup
  (`falcon/config.py`).
- **Streamlit_UI**: The five-tab web interface (`app.py`) rendered by Streamlit.
- **OpenRouter_API**: The OpenAI-compatible inference API used for model access.
- **Conda_Environment**: The `falcon` conda environment (Python 3.11) that must be active for all
  development, testing, and runtime operations.

---

## Requirements

---

### Requirement 1: No Default Assistant Fallback

**User Story:** As a researcher, I want Falcon to send no platform instruction when no system
prompt is active, so that the model receives only the conversation messages and any assistant-mode
behavior that appears is attributable to the model alone, not to the platform.

#### Acceptance Criteria

1. WHEN the system prompt is empty or whitespace, THE Inference_Engine SHALL build a payload
   that contains no system-role message.
2. WHEN the system prompt is empty or whitespace, THE Inference_Engine SHALL NOT prepend any
   default assistant persona, chatbot instruction, coaching framing, or platform description to
   the payload.
3. WHEN the system prompt is empty or whitespace, THE Prompt_State SHALL be set to `empty`.
4. WHEN the system prompt is a non-empty, non-whitespace string, THE Inference_Engine SHALL
   prepend exactly one system-role message containing that string, unmodified.
5. THE Context_Assembler SHALL set Prompt_State to `present` when a non-empty system prompt is
   active and `empty` when no system prompt is active.
6. FOR ALL payloads built from an empty system prompt, the assembled payload SHALL contain zero
   messages with role `system`.

---

### Requirement 2: Always Generate Output

**User Story:** As a researcher, I want Falcon to always return visible output for any valid user
input, so that silent failures never corrupt a session or produce an unrecorded inference event.

#### Acceptance Criteria

1. WHEN the model returns an empty string or whitespace-only response, THE Inference_Engine SHALL
   emit the literal marker `[no output]` as the response text.
2. WHEN a valid user message is submitted, THE Inference_Engine SHALL produce a non-empty string
   response within the configured `max_tokens` and API timeout constraints.
3. IF the OpenRouter API call raises an exception, THEN THE Streamlit_UI SHALL display an error
   message and SHALL NOT silently suppress the failure.
4. THE Inference_Engine SHALL never return `None` as a response value.
5. WHILE the Inference_Engine is streaming tokens, THE Streamlit_UI SHALL render output
   token-by-token without blocking the process until the stream is exhausted.

---

### Requirement 3: Raw Context Viewer

**User Story:** As a researcher, I want to see the exact assembled context before each inference
call, so that I can verify nothing hidden entered generation and trust the test results.

#### Acceptance Criteria

1. WHEN a message is sent, THE Raw_Context_Viewer SHALL display, for that turn: the Prompt_State
   indicator, the system prompt text (or a `None` notice), the current user input, the conversation
   history, retrieved memory entries with counts per type, any attached documents, and the
   assembled payload as a `[{role, content}, …]` list.
2. THE Raw_Context_Viewer SHALL display the Prompt_State as `present` (green) or `empty` (red)
   so the researcher can see at a glance whether a system prompt entered generation.
3. THE Context_Assembler SHALL make the assembled context available to the Raw_Context_Viewer
   before each inference call, not after.
4. THE Raw_Context_Viewer SHALL be accessible both in the dedicated Context tab and via a
   per-message `⌥ context` button in the Chat tab.
5. THE Raw_Context_Viewer SHALL display the assembled payload in the exact `[{role, content}, …]`
   format sent to the OpenRouter API.
6. IF memory entries were retrieved for a turn, THEN THE Raw_Context_Viewer SHALL display each
   entry's type, content, and pinned status.
7. THE Raw_Context_Viewer SHALL update its contents after every message send within the same
   Streamlit session.

---

### Requirement 4: Context Assembly Pipeline

**User Story:** As a researcher, I want every response to be generated from a fully visible,
labeled set of components, so that I can reconstruct exactly what entered generation for any turn.

#### Acceptance Criteria

1. THE Context_Assembler SHALL assemble the generation context from exactly these components in
   order: system prompt (if present) → conversation history → current user input.
2. THE Context_Assembler SHALL make retrieved memory available to the Raw_Context_Viewer as a
   labeled, separate section — not silently merged into the system prompt without disclosure.
3. WHEN memory entries are injected into the effective system prompt, THE Context_Assembler SHALL
   label them with their memory type (e.g., `[EPISODIC MEMORY]`) so the researcher can distinguish
   platform-added content from user-authored content.
4. THE Context_Assembler SHALL expose a `build_context_view` function that returns a structured
   dict containing: `system_prompt`, `prompt_state`, `current_input`, `conversation_history`,
   `retrieved_memory`, `documents`, `assembled_payload`, and `message_count`.
5. THE Context_Assembler SHALL expose a `build_payload` function that returns the exact
   `[{role, content}, …]` list and nothing else.
6. FOR ALL valid inputs, the output of `build_payload` and `build_context_view` SHALL be
   deterministic given the same inputs — no hidden state, no randomness in assembly.

---

### Requirement 5: Memory Architecture

**User Story:** As a researcher, I want a user-controlled persistent memory system with five typed
stores, so that I can maintain continuity across sessions and inspect exactly what memory entered
each generation.

#### Acceptance Criteria

1. THE Memory_Store SHALL support exactly five memory types: `episodic`, `semantic`, `working`,
   `identity`, and `archive`.
2. WHEN a memory entry is added, THE Memory_Store SHALL record: `identity_id`, `memory_type`,
   `content`, `tags`, `created_at`, `updated_at`, `pinned`, and `source`.
3. WHEN memory retrieval is performed for generation, THE Memory_Store SHALL return a
   `RetrievalResult` containing: the retrieved entries, a human-readable reasoning string per
   type, entries grouped by type, and a total count.
4. WHEN memory retrieval is performed, THE Memory_Store SHALL include pinned entries first,
   then most-recent entries, up to the configured limit per type.
5. THE Memory_Store SHALL scope all memory entries by `identity_id` — entries from Identity A
   SHALL NOT appear in retrieval results for Identity B.
6. WHEN a working memory entry exists for an identity, THE Memory_Store SHALL provide a
   `clear_working_memory` operation that deletes all working-type entries for that identity.
7. THE Memory_Store SHALL persist all entries in MongoDB and SHALL survive Falcon restarts.
8. FOR ALL memory entries, the round-trip `add_memory → get_memories` SHALL return an entry
   whose `content`, `memory_type`, `tags`, and `pinned` fields equal the values supplied to
   `add_memory`.
9. WHEN the user views the Memory tab after a message send, THE Streamlit_UI SHALL display what
   was retrieved, how many entries per type, and the retrieval reasoning so the researcher
   always knows what memory entered generation.

---

### Requirement 6: Identity Isolation

**User Story:** As a researcher, I want each identity to be a completely isolated context, so that
experiments run under Identity A cannot contaminate experiments run under Identity B.

#### Acceptance Criteria

1. THE Identity SHALL maintain separate, non-overlapping stores for: message history, memory
   (all five types), token usage counters, audit trail records, and trace snapshots.
2. WHEN the active identity is switched, THE Streamlit_UI SHALL immediately load the selected
   identity's history, memory, and token state and SHALL flush all session state belonging to
   the previous identity.
3. IF a new identity name is created, THEN THE Streamlit_UI SHALL create the identity context
   with empty history and empty memory stores.
4. WHEN an identity is deleted, THE Streamlit_UI SHALL delete all associated messages, traces,
   memory entries, token counters, and audit records for that identity.
5. THE Identity with `identity_id` equal to `default` SHALL NOT be deletable through the UI.
6. THE Identity SHALL validate `identity_id` values and SHALL raise a `ValueError` for values
   containing `/`, `\`, `..`, or null bytes.
7. FOR ALL pairs of distinct identities A and B, memory retrieval for A SHALL return zero
   entries whose `identity_id` equals B.

---

### Requirement 7: Inference Engine and Model Replaceability

**User Story:** As a researcher, I want to swap the active model without changing memory,
identity structure, context assembly, or the UI, so that I can run identical inputs through
multiple models and compare their raw outputs side by side.

#### Acceptance Criteria

1. THE Inference_Engine SHALL accept the model name as an explicit parameter; changing the model
   SHALL NOT require modifications to memory, identity, context assembly, Logger, Audit_Log, or
   Streamlit_UI.
2. THE Inference_Engine SHALL accept all five Generation_Controls as explicit parameters:
   temperature, top_p, repetition_penalty, max_tokens, stop_tokens.
3. WHEN a streaming inference call completes, THE Inference_Engine SHALL expose `.usage` (dict
   with `prompt_tokens`, `completion_tokens`, `total_tokens`) and `.raw_output` (unfiltered
   model text before any post-processing).
4. WHEN the model returns output containing `<think>…</think>` blocks, THE Inference_Engine
   SHALL strip those blocks from the displayed response but SHALL preserve the unstripped text
   in `.raw_output` for the audit trail.
5. THE Config_Loader SHALL expose an `available_models` list sourced from `config.yaml`;
   adding or removing a model from that list SHALL be sufficient to change the model selector
   options in the Streamlit_UI.
6. WHEN the model selector changes within a session, THE Inference_Engine SHALL use the newly
   selected model on the next inference call without requiring a page reload.

---

### Requirement 8: Generation Controls

**User Story:** As a researcher, I want all five generation parameters to be visible and adjustable
in the UI, so that I can reproduce any experimental condition and the parameters are always
recorded in the audit trail.

#### Acceptance Criteria

1. THE Streamlit_UI SHALL expose live controls for: temperature (0.0–2.0), top_p (0.0–1.0),
   repetition_penalty (1.0–2.0), max_tokens (64–32768), and stop_tokens (comma-separated strings).
2. WHEN generation controls are changed in the sidebar, THE Inference_Engine SHALL use the
   updated values on the next inference call without a page reload.
3. THE Config_Loader SHALL read default values for all five Generation_Controls from `config.yaml`
   and SHALL expose them as typed module-level constants.
4. WHEN an inference call is made, THE Audit_Log SHALL record the exact values of all five
   Generation_Controls used for that call.
5. THE Streamlit_UI SHALL display a compact summary of the current generation settings below
   the controls so the researcher can confirm active parameters at a glance.

---

### Requirement 9: Inference Audit Trail

**User Story:** As a researcher, I want a complete, append-only log of every inference event, so
that I can reconstruct any session, detect drift, identify contamination, and compare model
behavior under identical conditions.

#### Acceptance Criteria

1. WHEN an inference call completes, THE Audit_Log SHALL insert exactly one record containing:
   UTC timestamp, model name, identity, Prompt_State, system prompt text (or null), retrieved
   memory entries, Generation_Controls, assembled payload, raw model output, token usage, and
   latency in milliseconds.
2. THE Audit_Log SHALL be append-only; existing records SHALL NOT be modified or deleted through
   normal operation.
3. WHEN `read_audit_records` is called with an `identity_id`, THE Audit_Log SHALL return only
   records whose `identity_id` matches, sorted newest-first.
4. WHEN `read_all_audit_records` is called, THE Audit_Log SHALL return records across all
   identities sorted newest-first.
5. IF writing an audit record fails, THEN THE Audit_Log failure SHALL NOT interrupt or abort
   the main inference flow — the error SHALL be silently absorbed.
6. THE Audit_Log SHALL be stored in the MongoDB `audit_log` collection and SHALL be indexed on
   `identity_id` and `recorded_at`.
7. THE Streamlit_UI SHALL render the Audit tab with all logged fields per event, scoped to the
   active identity or across all identities.
8. FOR ALL inference events, the `assembled_payload` field in the Audit_Log record SHALL be
   identical to the payload actually sent to the OpenRouter API for that event.

---

### Requirement 10: Transparency of Platform-Added Content

**User Story:** As a researcher, I want any content added by the platform (not by the user, memory,
or identity file) to be visually distinguishable, so that I can identify platform influence in the
payload and test models against a truly clean context.

#### Acceptance Criteria

1. WHEN memory entries are injected into the effective system prompt, THE Context_Assembler SHALL
   label each block with its memory type in uppercase brackets (e.g., `[EPISODIC MEMORY]`) so the
   researcher can distinguish platform-appended content from user-authored content.
2. THE Raw_Context_Viewer SHALL display the Prompt_State indicator (`present` or `empty`) as a
   distinctly colored badge before the system prompt section.
3. THE Streamlit_UI SHALL indicate in the Raw_Context_Viewer when the system prompt is `None`
   with the exact caption: "None — empty instruction context. No assistant persona injected."
4. THE Inference_Engine SHALL expose `.raw_output` containing the unfiltered model text; the only
   transformation applied to model output before display SHALL be stripping `<think>…</think>`
   blocks.
5. WHEN a researcher wishes to test a clean payload across multiple models, THE Streamlit_UI
   SHALL allow the system prompt to be disabled via a checkbox, setting Prompt_State to `empty`
   with zero platform instructions sent to the model.

---

### Requirement 11: Persistent Message Logging

**User Story:** As a researcher, I want all conversation messages to be persisted in MongoDB, so
that sessions survive restarts and I can inspect, edit, or export the raw message data.

#### Acceptance Criteria

1. WHEN a user message is submitted, THE Logger SHALL append a record to the MongoDB `messages`
   collection with fields: `identity_id`, `timestamp` (UTC ISO 8601), `role` (`user`), `content`.
2. WHEN an assistant response is received, THE Logger SHALL append a record to the `messages`
   collection with fields: `identity_id`, `timestamp` (UTC ISO 8601), `role` (`assistant`),
   `content`.
3. WHEN `load_history` is called for an `identity_id`, THE Logger SHALL return all messages for
   that identity in chronological insertion order.
4. THE Streamlit_UI SHALL provide a Logs tab that renders conversation message pairs in a
   structured view with timestamps, editable content fields, inline save, and per-pair delete
   with confirmation.
5. THE Streamlit_UI SHALL provide a raw JSON view of the message log that is hand-editable and
   saveable directly to MongoDB.
6. FOR ALL messages logged then retrieved via `load_history`, the `role` and `content` fields
   SHALL equal the values originally passed to the Logger (round-trip property).

---

### Requirement 12: Configuration and Startup Validation

**User Story:** As a researcher, I want Falcon to fail immediately at startup with a clear error
message if any required configuration is missing, so that misconfiguration is never the hidden
cause of unexpected behavior during a session.

#### Acceptance Criteria

1. WHEN `falcon/config.py` is imported, THE Config_Loader SHALL read `OPENROUTER_API_KEY` from
   the `.env` file; IF the key is absent or empty, THEN THE Config_Loader SHALL raise a
   `ValueError` with a descriptive message before any other module initializes.
2. WHEN `falcon/config.py` is imported, THE Config_Loader SHALL read `MONGODB_URI` from the
   `.env` file; IF the URI is absent or empty, THEN THE Config_Loader SHALL raise a `ValueError`
   with a descriptive message.
3. WHEN `falcon/config.py` is imported, THE Config_Loader SHALL load `config.yaml`; IF the file
   is absent or malformed, THEN THE Config_Loader SHALL raise a `ValueError` with a descriptive
   message.
4. THE Config_Loader SHALL validate that `default_model` and `log_dir` are non-empty strings in
   `config.yaml`; IF either is absent or empty, THEN THE Config_Loader SHALL raise a `ValueError`.
5. WHEN the Streamlit_UI starts and a `ValueError` is raised during config import, THE
   Streamlit_UI SHALL display the error message and call `st.stop()` to halt further rendering.
6. THE Conda_Environment named `falcon` (Python 3.11) SHALL be used for all runtime, testing,
   and development operations; the README and all operational documentation SHALL reference
   `conda activate falcon` as the required activation step.

---

### Requirement 13: Portability and No Lock-In

**User Story:** As a researcher, I want all core components to be portable across machines, so
that I can move the system to new hardware without losing data, prompts, identity contexts, or
audit history.

#### Acceptance Criteria

1. THE Audit_Log, Memory_Store, message history, token counters, and trace snapshots SHALL all
   reside in MongoDB; a `mongodump` or equivalent export SHALL be sufficient to move all
   persistent data to a new machine.
2. THE Falcon system SHALL not require Docker, compiled migrations, or build steps; deploying to
   new hardware SHALL require only: cloning the repository, creating the `falcon` conda
   environment, installing `requirements.txt`, and setting `.env`.
3. THE model configuration SHALL be stored in `config.yaml`; changing the model list or default
   model SHALL require only editing `config.yaml` and restarting the Streamlit session.
4. THE Streamlit_UI, Inference_Engine, Context_Assembler, Memory_Store, Audit_Log, and Logger
   SHALL each be replaceable or deployable independently of the others without changing shared
   data schemas.
5. THE system SHALL support any OpenRouter-compatible model string in `available_models` without
   code changes.

---

### Requirement 14: Evaluation and Model Comparison Support

**User Story:** As a researcher, I want to run identical inputs through multiple models under
identical conditions with a complete audit trail per run, so that I can measure which model
introduces the least unsolicited behavior.

#### Acceptance Criteria

1. WHEN the model selector is changed between runs, THE Inference_Engine SHALL preserve the same
   identity, memory, conversation history, system prompt, and Generation_Controls so that the
   only variable is the model.
2. THE Audit_Log SHALL record the model name on every inference event so that records from
   different models can be compared side by side for the same identity and prompt conditions.
3. WHEN a researcher wishes to start a fresh comparison run, THE Streamlit_UI SHALL provide a
   "Clear conversation" action that wipes messages, traces, tokens, and audit records for the
   active identity (with confirmation) so that prior AI-style history does not contaminate the
   result.
4. THE Raw_Context_Viewer SHALL be available before and after every inference call so the
   researcher can verify the exact payload before submitting to the next model.
5. THE Audit_Log SHALL record `prompt_state` on every event so the researcher can confirm that
   a system prompt was absent (or present) consistently across all model comparison runs.

---

### Requirement 15: Think-Block Filtering (Parser / Serializer)

**User Story:** As a researcher, I want chain-of-thought `<think>…</think>` blocks from reasoning
models to be stripped from the displayed response but preserved in the audit trail, so that the UI
is clean while the raw model output is fully reconstructible.

#### Acceptance Criteria

1. WHEN the model output contains one or more `<think>…</think>` blocks, THE Inference_Engine
   SHALL strip those blocks from the token stream before rendering to the Streamlit_UI.
2. WHEN the model output contains `<think>…</think>` blocks, THE Inference_Engine SHALL preserve
   the full, unstripped text in `.raw_output` for storage in the Audit_Log.
3. WHEN the model output contains no `<think>…</think>` blocks, THE Inference_Engine SHALL pass
   the output through unmodified; `.raw_output` SHALL equal the displayed response.
4. FOR ALL model outputs, the Inference_Engine `<think>` filter SHALL be idempotent — applying
   the filter twice to an already-filtered output SHALL produce the same result as applying it
   once.
5. FOR ALL model outputs containing properly nested `<think>…</think>` blocks, stripping SHALL
   produce output that contains no `<think>` or `</think>` tags.
6. THE Inference_Engine SHALL handle `<think>` blocks that arrive split across multiple streaming
   chunks without dropping or duplicating tokens outside the think block.
