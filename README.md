# 🦅 Falcon V1

A neutral LLM communication interface — infrastructure, not an assistant.

Falcon is a clean, transparent channel between you and a language model. Input goes in, output comes out, everything is visible, nothing is hidden. There is no personality, no hidden prompt injection, and no default "helpful assistant" framing at any layer.

---

## What it is (and isn't)

| ✅ It IS | ❌ It is NOT |
|---|---|
| A transparent inference interface | A chatbot or assistant |
| A per-identity conversation logger | An agent or RAG pipeline |
| A payload inspector | A streaming proxy with memory compression |
| Portable, file-based infrastructure | A multi-user or cloud service |

---

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| LLM Backend | [Groq API](https://console.groq.com) via `langchain-groq` |
| LLM Abstraction | LangChain + LangChain Core |
| UI | Streamlit ≥ 1.35 |
| Storage | Local JSON files (no database) |
| Config | `.env` + `config.yaml` |
| Testing | pytest + Hypothesis |

---

## Project Structure

```
falconv1/
├── app.py                   # Streamlit entry point — full UI
├── config.yaml              # Model list, log dir, system prompt default
├── .env                     # Your Groq API key (gitignored)
├── .env.example             # Template for .env
├── requirements.txt
│
├── falcon/                  # Core package
│   ├── __init__.py
│   ├── config.py            # Config loader — validates on import
│   ├── engine.py            # Groq/LangChain inference + streaming
│   ├── identity.py          # Per-identity log file management
│   └── logger.py            # Append-only JSON logger
│
├── logs/                    # Auto-created at runtime
│   ├── {id}.json            # Conversation log per identity
│   ├── {id}.traces.json     # Per-turn inference trace snapshots
│   └── {id}.tokens.json     # Cumulative token usage (survives reloads)
│
└── tests/                   # Full test suite
    ├── test_config.py
    ├── test_engine.py
    ├── test_identity.py
    ├── test_logger.py
    ├── test_integration.py
    └── test_property_*.py   # Hypothesis property-based tests
```

---

## Installation

**Prerequisites:** Python 3.10+, pip, a [Groq API key](https://console.groq.com/keys).

```bash
# 1. Clone
git clone <repo-url>
cd falconv1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
copy .env.example .env      # Windows
# cp .env.example .env      # Linux/macOS

# Edit .env and set your key:
# GROQ_API_KEY=your_key_here

# 4. Run
streamlit run app.py
```

The app fails immediately with a clear error if `GROQ_API_KEY` is missing — no silent failures.

---

## Configuration

**`config.yaml`** — edit freely, no code changes needed:

```yaml
default_model: "llama-3.1-8b-instant"

available_models:
  - "llama-3.3-70b-versatile"
  - "openai/gpt-oss-20b"
  - "qwen/qwen3-32b"
  - "meta-llama/llama-4-scout-17b-16e-instruct"

default_system_prompt: "You are a neutral text-processing interface. "
    "Respond only to what is explicitly asked. "
    "Do not add explanations, caveats, suggestions, offers of further help, "
    "affirmations, apologies, or any framing language. "
    "Do not refer to yourself as an AI, assistant, or language model. "
    "Do not begin responses with filler phrases such as 'Certainly', 'Of course', "
    "'Sure', 'Great', 'Absolutely', or similar. "
    "Output only the direct answer or result. "
    "If the input is ambiguous, respond with the most literal interpretation. "
    "Use the minimum number of words necessary to be complete and accurate."

log_dir: "logs"
```

Add or remove models from `available_models` to populate the model selector in the UI.

---

## Using the UI

The interface has three tabs: **Chat**, **Logs**, and **Trace**.

### Chat tab

- Type a message and press Enter (or click the send button) to get a response.
- Responses stream in real-time token by token.
- Use the **Clear conversation** button (with confirmation) to wipe the current identity's log.

### Logs tab

Two sub-views:

- **Raw JSON** — editable textarea showing the full log file. Save writes directly to disk.
- **Structured** — expandable message pairs. Each pair can be individually edited, deleted, or inspected for its trace. Changes sync back to disk and the active session.

### Trace tab (sidebar)

Every inference call generates a step-by-step trace:

1. Config snapshot (model, temperature, identity)
2. User message logged to disk
3. Full history loaded
4. Payload built
5. LangChain messages sent to Groq
6. Response received + latency
7. Token usage (per-call and cumulative)
8. Final log state

---

## Identities

An **identity** is a named conversation context, isolated to its own log file (`logs/{name}.json`).

- Use the **Identity** selector in the sidebar to switch between conversations.
- Create a new identity by typing a name and clicking **＋ Create**.
- Delete any non-default identity with the **🗑 Delete** button (confirmation required).
- Switching identity immediately loads that identity's full history — there is no cross-contamination.

**Security:** Identity IDs are validated to reject path traversal characters (`/`, `\`, `..`, null bytes) before any file path is constructed.

---

## Data Flow

Each message send follows this exact sequence:

```
User input
  → Logger.append_message(identity, "user", content)
  → Identity.load_history(identity)        ← full history, no truncation
  → Engine.build_payload(system_prompt, messages)   ← zero injection
  → ChatGroq.stream()                      ← Groq API
  → st.write_stream()                      ← token-by-token render
  → Logger.append_message(identity, "assistant", response)
  → Trace snapshot written to disk
  → Session state updated → st.rerun()
```

The full conversation history is sent on every call. Token-limit enforcement is the Groq API's responsibility — Falcon never truncates.

---

## Log Files

All logs are plain JSON arrays, human-readable and hand-editable in any text editor:

```json
[
  {
    "timestamp": "2025-06-05T14:22:01Z",
    "role": "user",
    "content": "What is entropy?"
  },
  {
    "timestamp": "2025-06-05T14:22:03Z",
    "role": "assistant",
    "content": "A measure of disorder or uncertainty in a system."
  }
]
```

Each entry has exactly three fields: `timestamp` (ISO 8601 UTC), `role`, `content`.

---

## Neutrality

The following rules are enforced at every layer of the system:

- The default system prompt is set in falcon/config.py file.
- The engine adds **zero** hidden context — `build_payload()` is the authoritative, inspectable record of what goes to Groq.
- The `<think>...</think>` filter strips chain-of-thought blocks from reasoning models (e.g. Qwen3) before they reach the UI — the model's reasoning is not echoed.

---

## Switching to a Different LLM Provider

Only `falcon/engine.py` needs to change. To use Ollama instead of Groq:

```python
# Before (engine.py)
from langchain_groq import ChatGroq
llm = ChatGroq(model=model_name, api_key=groq_api_key, ...)

# After (engine.py)
from langchain_ollama import ChatOllama
llm = ChatOllama(model=model_name, ...)
```

Update `config.yaml` with the new model names. No other changes needed.

---

## Testing

```bash
# Run all tests
pytest

# Run a specific module
pytest tests/test_engine.py -v

# Run property-based tests only
pytest tests/test_property_logger.py tests/test_property_identity.py -v

# Reproduce a specific Hypothesis run
pytest --hypothesis-seed=0
```

Tests use `tmp_path` and `monkeypatch` to redirect log directories — no real files are written during test runs.

---

## Deploying to New Hardware

Copy the repo, install Python, run pip install, set `.env`. That's it.

```bash
pip install -r requirements.txt
# set GROQ_API_KEY in .env
streamlit run app.py
```

No Docker, no migrations, no build steps.

---
