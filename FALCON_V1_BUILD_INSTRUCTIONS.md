# FALCON V1 — COMPLETE BUILD INSTRUCTIONS FOR CLAUDE

> These instructions are written for Claude to build Falcon V1 from scratch.
> Read every section fully before writing any code. Do not improvise scope. Do not add features not listed here.
> When in doubt: do less, keep it cleaner.

---

## 0. WHAT FALCON IS (AND IS NOT)

Falcon is **infrastructure**, not an assistant.

- It is NOT a chatbot. It is NOT a personality. It is NOT an agent.
- It IS a clean communication room: input goes in, output comes out, everything is visible, nothing is hidden.
- The primary design value is **neutrality**. Strip all assistant steering, coaching, default helpfulness framing, and personality overlays from every layer of the system — prompt, UI, and code.
- If you ever face a choice between adding a feature and keeping the system simpler: **choose simpler, always.**

---

## 1. STACK DECISION

| Layer | Choice | Reason |
|---|---|---|
| LLM Backend | **Groq API** (via `langchain-groq`) | User has Groq API key; no local Ollama available for now |
| Inference abstraction | **LangChain / LangGraph** | Portable — swap Groq for Ollama/vLLM later by changing one config line |
| UI | **Streamlit** | Simple, Python-native, fast to iterate |
| Memory/Logging | **Local JSON files** | Flat, human-readable, no DB dependency, easy to inspect/edit |
| Identity isolation | **`identity_id` field in JSON logs** | Strict per-identity file or filtered reads — zero cross-contamination |
| Config | **`.env` file + `config.yaml`** | Portable to any machine, no hardcoded secrets |

**Portability note:** The entire system must run with `python app.py` or `streamlit run app.py` on any machine. Moving to a DGX Spark or dedicated local hardware later should require only changing the `.env` LLM endpoint — nothing else.

---

## 2. REPOSITORY STRUCTURE

Build exactly this structure. No extra folders.

```
falcon-v1/
├── app.py                  # Streamlit entry point
├── falcon/
│   ├── __init__.py
│   ├── engine.py           # LangChain/Groq connection engine
│   ├── identity.py         # Identity management and isolation
│   ├── logger.py           # JSON logging system
│   └── config.py           # Config loader
├── logs/                   # Auto-created at runtime. JSON log files live here.
├── config.yaml             # Model list, defaults, system prompt default
├── .env.example            # Template: GROQ_API_KEY=your_key_here
├── .env                    # Actual key — gitignored
├── requirements.txt
├── .gitignore
└── RUNTIME_MANUAL.md       # Full usage and deployment guide
```

---

## 3. FILE-BY-FILE SPECIFICATIONS

### 3.1 `config.yaml`

```yaml
default_model: "llama3-70b-8192"

available_models:
  - "llama3-70b-8192"
  - "llama3-8b-8192"
  - "mixtral-8x7b-32768"
  - "gemma2-9b-it"

default_system_prompt: ""

log_dir: "logs"
```

- `default_system_prompt` is intentionally empty. Falcon starts with no system prompt. The user sets it.
- Model list should be editable in `config.yaml` without touching code.

---

### 3.2 `.env.example`

```
GROQ_API_KEY=your_groq_api_key_here
```

Instruct users to copy this to `.env` and fill in their key. `.env` must be in `.gitignore`.

---

### 3.3 `falcon/config.py`

- Load `.env` using `python-dotenv`.
- Load `config.yaml` using `PyYAML`.
- Expose: `GROQ_API_KEY`, `available_models`, `default_model`, `default_system_prompt`, `log_dir`.
- Raise a clear error if `GROQ_API_KEY` is missing.

---

### 3.4 `falcon/engine.py` — The Connection Engine

This is the core. Build it with LangChain and `langchain-groq`.

**Responsibilities:**
- Accept: `model_name`, `system_prompt` (string, may be empty), `messages` (list of `{role, content}` dicts), `identity_id`.
- Build the exact JSON payload that will be sent to the model. **Return this payload as a Python dict** so the UI can display it raw.
- Call the Groq API via LangChain's `ChatGroq`.
- Return: `{"response": <string>, "raw_payload": <dict>}`.

**LangChain usage:**
```python
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def build_payload(system_prompt, messages):
    # Construct the exact list that will be sent
    payload_messages = []
    if system_prompt:
        payload_messages.append({"role": "system", "content": system_prompt})
    payload_messages.extend(messages)
    return payload_messages

def run_inference(model_name, system_prompt, messages, groq_api_key):
    lc_messages = []
    if system_prompt:
        lc_messages.append(SystemMessage(content=system_prompt))
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    llm = ChatGroq(model=model_name, api_key=groq_api_key)
    response = llm.invoke(lc_messages)
    return response.content
```

**Important:** Do NOT use LangGraph chains, agents, tools, or memory abstractions in V1. Use only the raw `ChatGroq.invoke()` call. LangGraph can be added in V2. Keep the engine as thin as possible.

---

### 3.5 `falcon/identity.py` — Identity Isolation

- An `identity_id` is a simple string (e.g. `"user_1"`, `"test_A"`).
- Each identity has its own log file: `logs/{identity_id}.json`.
- **Zero cross-contamination:** the engine and logger must never mix data across identities. No shared state, no global conversation list.
- Provide these functions:
  - `list_identities()` → list all `identity_id`s that have log files.
  - `load_history(identity_id)` → return list of `{role, content, timestamp}` dicts for that identity, in strict chronological order.
  - `clear_identity(identity_id)` → delete or empty that identity's log file.

---

### 3.6 `falcon/logger.py` — JSON Logging

Log file format per identity (`logs/{identity_id}.json`):

```json
[
  {
    "timestamp": "2025-06-05T14:22:01Z",
    "role": "user",
    "content": "Hello."
  },
  {
    "timestamp": "2025-06-05T14:22:03Z",
    "role": "assistant",
    "content": "Hello."
  }
]
```

**Rules:**
- Capture only: `timestamp`, `role`, `content`. Nothing else.
- Append only. Never overwrite existing entries during a session.
- On each message, write immediately to disk (do not buffer).
- Provide: `append_message(identity_id, role, content)`.
- The log directory must be auto-created if it doesn't exist.
- Logs must be human-readable and hand-editable with any text editor.

---

### 3.7 `app.py` — Streamlit UI

Build the Streamlit interface. Layout: **two-column** (main chat area + right sidebar transparency panel) OR use `st.sidebar` for the control panel. Keep it clean and functional — no decorative elements, no branding, no color themes beyond Streamlit defaults.

#### Left / Main Area:

1. **Identity selector** — text input for `identity_id`. Default: `"default"`. Changing this switches context entirely.

2. **Chat history display** — show messages from the loaded identity log in order: user messages and assistant responses. Display role label and content only.

3. **Message input** — single `st.text_area` for user input. A "Send" button submits it.

4. **Edit history button** — a button labeled "Edit Logs" that opens the raw JSON log file path and tells the user to edit it manually, then reload. (Do not build an inline editor in V1 — keep it simple.)

5. **Clear conversation button** — calls `clear_identity(identity_id)` and resets the session.

#### Sidebar / Transparency Panel:

The sidebar must contain ALL of the following, in this order:

1. **Model selector** — `st.selectbox` populated from `config.yaml`. Changing it takes effect on the next message.

2. **System prompt editor** — `st.text_area`, starts empty, labeled "System Prompt". User can type, paste, or clear. This is sent directly to the model with zero modification.

3. **Raw Payload Panel** — labeled "Outbound Payload (sent to model)". After each message, display the exact JSON payload as `st.json(raw_payload)`. This must update in real time after each send.

4. **Raw Response Panel** — labeled "Raw Response". Display the model's raw text response as `st.code(response)`.

5. **Identity info** — display current `identity_id` and message count.

#### Session state:
- Use `st.session_state` to hold: current identity_id, conversation history (loaded from JSON), system prompt, selected model, last raw payload, last raw response.
- On identity switch: reload history from that identity's JSON file immediately.

---

## 4. WHAT TO EXPLICITLY NOT BUILD

Do not build any of the following, even if they seem helpful:

- No RAG, no vector store, no embeddings.
- No LangGraph agents, tools, or multi-step chains.
- No memory summarization or compression.
- No personality, tone injection, or default assistant framing in the system prompt.
- No user authentication.
- No database (SQLite, Postgres, etc.) — JSON files only.
- streaming responses.
- No Docker, no deployment scripts beyond pip install.
- No frontend beyond stock Streamlit.

---

## 5. NEUTRALITY RULES

These rules apply to every layer of the system:

- The default system prompt is **empty string**. Do not pre-fill it with anything.
- Do not inject any hidden system prompt. What the user types in the system prompt box is the complete and entire system prompt. If it's empty, send no system message.
- Do not add any "helpful assistant" framing, preamble, or postamble anywhere in the code.
- The engine must pass the user's message to the model exactly as typed. No preprocessing, no reformatting, no added context.
- The UI must not coach, suggest, or guide the user beyond basic labels.

---

## 6. REQUIREMENTS FILE

```
streamlit>=1.35.0
langchain>=0.2.0
langchain-groq>=0.1.5
langchain-core>=0.2.0
python-dotenv>=1.0.0
PyYAML>=6.0
```

No other dependencies. Do not add anything not in this list without noting it explicitly.

---

## 7. RUNTIME MANUAL (`RUNTIME_MANUAL.md`)

Write a complete manual covering:

1. **Prerequisites** — Python 3.10+, pip, a Groq API key.
2. **Installation** — `git clone`, `pip install -r requirements.txt`, copy `.env.example` → `.env`, add API key.
3. **Running** — `streamlit run app.py`.
4. **Using the UI** — what each panel does, how to switch identities, how to change models, how to set system prompt, how to read the payload panel.
5. **Log files** — where they are, how to read/edit them manually.
6. **Switching to Ollama/vLLM later** — explain that changing `engine.py` to use `ChatOllama` instead of `ChatGroq` is the only required change. Show the 3-line diff.
7. **Deploying to new hardware** — copy the repo, install Python, pip install, set `.env`. That's it.

---

## 8. BUILD ORDER

Build in this exact order. Test each step before moving to the next.

1. `config.yaml` + `.env.example` + `falcon/config.py` — verify config loads cleanly.
2. `falcon/logger.py` — write a test that appends two messages and reads them back.
3. `falcon/identity.py` — verify isolation: two identities, confirm no data bleeds between them.
4. `falcon/engine.py` — verify a raw Groq call returns a response and a correct raw payload dict.
5. `app.py` — build the Streamlit UI, wire all modules together.
6. End-to-end test: open UI, set identity, type a message, confirm it appears in log, confirm raw payload is visible in sidebar, confirm switching identity clears context.
7. Write `RUNTIME_MANUAL.md`.
8. Final check: delete `.env`, run app, confirm it fails with a clear error message (not a traceback).

---

## 9. HANDOVER CHECKLIST

Before considering V1 complete, confirm all of the following:

- [ ] `streamlit run app.py` works on a fresh machine after `pip install -r requirements.txt`.
- [ ] Changing identity in the UI loads a different (isolated) conversation.
- [ ] The raw JSON payload is visible in the sidebar after every message.
- [ ] The system prompt can be edited or cleared mid-conversation without restarting.
- [ ] Changing the model dropdown takes effect on the next message.
- [ ] Log files are written to `logs/` and are human-readable JSON.
- [ ] No cross-identity data contamination (verify by checking log files directly).
- [ ] `RUNTIME_MANUAL.md` covers all steps from zero to running.
- [ ] `.env` is in `.gitignore`. No API key is hardcoded anywhere.
- [ ] The codebase has no TODO comments, no dead code, no placeholder functions.

---

*End of Falcon V1 Build Instructions.*
