"""
app.py — Falcon V1 Streamlit UI

Top-level tabs: Chat | Logs | Trace

Trace for every send is persisted to logs/{identity_id}.traces.json.
In the Logs tab, each message pair expander has a nested Trace section
showing every step of the inference call for that turn.
"""

import json
import os
import time
from datetime import datetime, timezone

import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    import falcon.config as Config
except ValueError as exc:
    st.error(str(exc))
    st.stop()

import falcon.engine as Engine
import falcon.identity as Identity
import falcon.logger as Logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3] + "Z"


def _log_path(identity_id: str) -> str:
    return os.path.join(Config.log_dir, f"{identity_id}.json")


def _trace_path(identity_id: str) -> str:
    return os.path.join(Config.log_dir, f"{identity_id}.traces.json")


def _tokens_path(identity_id: str) -> str:
    return os.path.join(Config.log_dir, f"{identity_id}.tokens.json")


def _load_persisted_tokens(identity_id: str) -> dict:
    """Read the durable token counter file for this identity."""
    path = _tokens_path(identity_id)
    if not os.path.exists(path):
        return {"prompt": 0, "completion": 0, "total": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "prompt":     data.get("prompt", 0),
            "completion": data.get("completion", 0),
            "total":      data.get("total", 0),
        }
    except Exception:
        return {"prompt": 0, "completion": 0, "total": 0}


def _persist_tokens(identity_id: str, tokens: dict) -> None:
    """Write the current token totals to disk so they survive page reloads."""
    os.makedirs(Config.log_dir, exist_ok=True)
    with open(_tokens_path(identity_id), "w", encoding="utf-8") as f:
        json.dump(tokens, f)


def _read_log_raw(identity_id: str) -> str:
    path = _log_path(identity_id)
    if not os.path.exists(path):
        return "[]"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_log_entries(identity_id: str):
    try:
        return json.loads(_read_log_raw(identity_id))
    except Exception:
        return None


def _read_traces(identity_id: str) -> list[dict]:
    path = _trace_path(identity_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except Exception:
        return []


def _append_trace(identity_id: str, snapshot: dict) -> None:
    """Append one trace snapshot to the traces file."""
    traces = _read_traces(identity_id)
    traces.append(snapshot)
    os.makedirs(Config.log_dir, exist_ok=True)
    with open(_trace_path(identity_id), "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)


def _delete_trace_for_timestamp(identity_id: str, user_ts: str) -> None:
    """Remove trace snapshot matching the given user message timestamp."""
    traces = [t for t in _read_traces(identity_id) if t.get("user_timestamp") != user_ts]
    os.makedirs(Config.log_dir, exist_ok=True)
    with open(_trace_path(identity_id), "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)


def _lc_message_repr(lc_messages: list) -> list[dict]:
    return [{"type": type(m).__name__, "content": m.content} for m in lc_messages]


def _pair_count(history: list) -> int:
    pairs, i = 0, 0
    while i < len(history):
        if (i + 1 < len(history)
                and history[i].get("role") == "user"
                and history[i + 1].get("role") == "assistant"):
            i += 2
        else:
            i += 1
        pairs += 1
    return pairs


def _build_pairs(entries: list[dict]) -> list[tuple[int, list[int], bool]]:
    pairs, i, pnum = [], 0, 1
    while i < len(entries):
        if (i + 1 < len(entries)
                and entries[i].get("role") == "user"
                and entries[i + 1].get("role") == "assistant"):
            pairs.append((pnum, [i, i + 1], True))
            i += 2
        else:
            pairs.append((pnum, [i], False))
            i += 1
        pnum += 1
    return pairs


def _save_entries(identity_id: str, entries: list[dict]) -> None:
    path = _log_path(identity_id)
    os.makedirs(Config.log_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def _render_trace_steps(steps: list[dict]) -> None:
    """Render a list of trace step dicts inline (no placeholder needed)."""
    sections = []
    for entry in steps:
        t, stage, data = entry["t"], entry["stage"], entry["data"]
        status  = entry.get("status", "info")
        elapsed = entry.get("elapsed_ms")
        icon    = {"info": "◦", "success": "✓", "error": "✗", "warn": "⚠"}.get(status, "◦")
        estr    = f" `+{elapsed}ms`" if elapsed is not None else ""
        if isinstance(data, (dict, list)):
            sections.append(
                f"**`{t}`**{estr} {icon} **{stage}**\n"
                f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
            )
        else:
            sections.append(f"**`{t}`**{estr} {icon} **{stage}**\n```\n{data}\n```")
    st.markdown("\n\n---\n\n".join(sections))


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "identity_id": "default",
        "history": [],
        "selected_model": Config.default_model,
        "last_payload": None,
        "last_response": None,
        "trace_log": [],
        "session_tokens": {"prompt": 0, "completion": 0, "total": 0},
        "_loaded_initial_history": False,
        "_confirm_clear": False,
        "_confirm_pair": None,
        "_delete_confirmed": False,
        "_confirm_delete_identity": False,
        "_view_trace_ts": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def _load_identity_history(identity_id: str) -> None:
    try:
        st.session_state.history = Identity.load_history(identity_id)
    except Exception as exc:
        st.error(f"Failed to load '{_log_path(identity_id)}': {exc}")


def _restore_token_totals(identity_id: str) -> None:
    """Restore session token totals from the durable token counter file."""
    st.session_state.session_tokens = _load_persisted_tokens(identity_id)


def _on_identity_change() -> None:
    pass  # replaced by sidebar identity selector



# ---------------------------------------------------------------------------
# Live trace renderer (for Trace tab)
# ---------------------------------------------------------------------------

def _render_trace(placeholder, trace: list[dict], live: bool = False) -> None:
    if not trace:
        placeholder.empty()
        return
    sections = []
    for entry in trace:
        t, stage, data = entry["t"], entry["stage"], entry["data"]
        status  = entry.get("status", "info")
        elapsed = entry.get("elapsed_ms")
        icon    = {"info": "◦", "success": "✓", "error": "✗", "warn": "⚠"}.get(status, "◦")
        estr    = f" `+{elapsed}ms`" if elapsed is not None else ""
        if isinstance(data, (dict, list)):
            sections.append(f"**`{t}`**{estr} {icon} **{stage}**\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```")
        else:
            sections.append(f"**`{t}`**{estr} {icon} **{stage}**\n```\n{data}\n```")
    if live:
        sections.append("_running…_")
    placeholder.markdown("\n\n---\n\n".join(sections))


# ---------------------------------------------------------------------------
# Send flow
# ---------------------------------------------------------------------------

def _handle_send(user_input: str) -> None:
    identity_id   = st.session_state.identity_id
    model         = st.session_state.selected_model
    system_prompt = Config.default_system_prompt
    trace: list[dict] = []
    t0 = time.monotonic()

    def _push(stage: str, data, status: str = "info"):
        trace.append({"t": _ts(), "stage": stage, "data": data,
                      "status": status, "elapsed_ms": round((time.monotonic() - t0) * 1000)})

    _push("config", {"model": model, "temperature": 0, "top_p": 1, "stop_sequences": None,
                     "log_dir": Config.log_dir, "log_file": _log_path(identity_id),
                     "identity": identity_id, "system_prompt": system_prompt})

    try:
        Logger.append_message(identity_id, "user", user_input)
    except Exception as exc:
        _push("ERROR — log user message", str(exc), status="error")
        st.error(f"Failed to log user message: {exc}")
        st.session_state.trace_log = trace
        return

    _push("user → logged to disk", {"file": _log_path(identity_id),
                                     "entry": {"role": "user", "content": user_input}})

    messages = Identity.load_history(identity_id)
    _push("history loaded from disk", {"file": _log_path(identity_id),
                                        "entry_count": len(messages), "entries": messages})

    raw_payload = Engine.build_payload(system_prompt, messages)
    _push("payload built", {"message_count": len(raw_payload), "payload": raw_payload})

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    lc_preview = []
    for e in raw_payload:
        if e["role"] == "system":      lc_preview.append(SystemMessage(content=e["content"]))
        elif e["role"] == "user":      lc_preview.append(HumanMessage(content=e["content"]))
        elif e["role"] == "assistant": lc_preview.append(AIMessage(content=e["content"]))
    _push("langchain messages → llm.invoke()", _lc_message_repr(lc_preview))

    _push("→ Groq API call (streaming)", {"model": model, "temperature": 0, "top_p": 1,
                                           "stop_sequences": None, "messages_count": len(raw_payload)})

    # Stream tokens into a chat bubble in real time
    api_t0 = time.monotonic()
    response_text = ""
    stream_gen = Engine.stream_inference(
        model_name=model,
        system_prompt=system_prompt,
        messages=messages,
        groq_api_key=Config.GROQ_API_KEY,
    )
    try:
        # Consume the stream silently first so we capture the full response and usage
        chunks = []
        for chunk in stream_gen:
            chunks.append(chunk)
        response_text = "".join(chunks)
    except Exception as exc:
        _push("ERROR — Groq API", str(exc), status="error")
        st.error(f"Inference failed: {exc}")
        st.session_state.history = Identity.load_history(identity_id)
        st.session_state.trace_log = trace
        return
    api_latency_ms = round((time.monotonic() - api_t0) * 1000)

    # Only render the assistant bubble if the model produced visible content
    if response_text.strip():
        with st.chat_message("assistant"):
            st.markdown(response_text)

    _push("← response complete", {"latency_ms": api_latency_ms, "content": response_text})

    # Token usage — read from the usage_metadata that Groq sends on the final chunk
    usage = stream_gen.usage  # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    if usage:
        st.session_state.session_tokens["prompt"]     += usage.get("prompt_tokens", 0)
        st.session_state.session_tokens["completion"] += usage.get("completion_tokens", 0)
        st.session_state.session_tokens["total"]      += usage.get("total_tokens", 0)
        # Persist so that page reloads restore the true cumulative total
        # (independent of message deletions)
        _persist_tokens(identity_id, st.session_state.session_tokens)

    _push("token usage", {
        "this_call": usage,
        "session_cumulative": {
            "prompt_tokens":     st.session_state.session_tokens["prompt"],
            "completion_tokens": st.session_state.session_tokens["completion"],
            "total_tokens":      st.session_state.session_tokens["total"],
        },
    })

    try:
        Logger.append_message(identity_id, "assistant", response_text)
    except Exception as exc:
        _push("ERROR — log assistant response", str(exc), status="error")
        st.error(f"Failed to log assistant response: {exc}")

    _push("assistant → logged to disk", {"file": _log_path(identity_id),
                                          "entry": {"role": "assistant", "content": response_text}})

    final_history = Identity.load_history(identity_id)
    try:
        log_contents = json.loads(_read_log_raw(identity_id))
    except Exception:
        log_contents = "parse error"
    _push("log file — final state", {"file": _log_path(identity_id),
                                      "total_entries": len(final_history),
                                      "contents": log_contents}, status="success")

    # Persist trace snapshot — keyed by user message timestamp (stable across deletions)
    # The user message is the second-to-last entry (last is assistant)
    user_ts = ""
    if len(final_history) >= 2:
        user_ts = final_history[-2].get("timestamp", "")
    elif len(final_history) == 1:
        user_ts = final_history[-1].get("timestamp", "")

    snapshot = {
        "user_timestamp": user_ts,
        "send_timestamp": _ts(),
        "user": user_input,
        "steps": trace,
    }
    _append_trace(identity_id, snapshot)

    st.session_state.last_payload  = raw_payload
    st.session_state.last_response = response_text
    st.session_state.history       = final_history
    st.session_state.trace_log     = trace


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------

def _handle_clear() -> None:
    identity_id = st.session_state.identity_id
    try:
        # Write empty array — keeps the identity file so it stays in the dropdown
        os.makedirs(Config.log_dir, exist_ok=True)
        with open(_log_path(identity_id), "w", encoding="utf-8") as f:
            f.write("[]")
    except Exception as exc:
        st.error(f"Failed to clear conversation: {exc}")
        return
    # Remove traces file (no messages = no traces)
    tp = _trace_path(identity_id)
    if os.path.exists(tp):
        os.remove(tp)
    # Remove tokens file so reload starts fresh
    tkp = _tokens_path(identity_id)
    if os.path.exists(tkp):
        os.remove(tkp)
    st.session_state.history        = []
    st.session_state.last_payload   = None
    st.session_state.last_response  = None
    st.session_state.trace_log      = []
    st.session_state.session_tokens = {"prompt": 0, "completion": 0, "total": 0}


# ---------------------------------------------------------------------------
# Tab: Chat
# ---------------------------------------------------------------------------

def _render_chat_tab(user_input: str | None) -> None:
    history = st.session_state.history

    # ── Empty state ───────────────────────────────────────────────────────────
    if not history and not user_input:
        st.markdown("""
        <div class="empty-chat">
            <div class="icon">🦅</div>
            <div class="title">Start a conversation</div>
            <div class="sub">Type a message below to begin</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Render existing history ───────────────────────────────────────────
        for entry in history:
            # Skip assistant entries with no visible content
            if entry.get("role") == "assistant" and not entry.get("content", "").strip():
                continue
            with st.chat_message(entry.get("role", "user")):
                st.markdown(entry.get("content", ""))

    # ── Handle new message ────────────────────────────────────────────────────
    if user_input and user_input.strip():
        with st.chat_message("user"):
            st.markdown(user_input)
        _handle_send(user_input)
        st.rerun()

    # ── Footer controls ───────────────────────────────────────────────────────
    if history:
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        col_clear, col_spacer = st.columns([1, 5])
        with col_clear:
            st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
            if st.button("Clear conversation", key="_clear_btn", use_container_width=True):
                st.session_state._confirm_clear = True
            st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state._confirm_clear:
        st.markdown("""
        <div class="confirm-bar">
            ⚠️ This will permanently delete the conversation log and all traces.
        </div>
        """, unsafe_allow_html=True)
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("Confirm", type="primary", use_container_width=True, key="_clear_confirm"):
                _handle_clear()
                st.session_state._confirm_clear = False
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True, key="_clear_cancel"):
                st.session_state._confirm_clear = False
                st.rerun()


# ---------------------------------------------------------------------------
# Tab: Logs
# ---------------------------------------------------------------------------

def _render_logs_tab() -> None:
    identity_id = st.session_state.identity_id
    path        = _log_path(identity_id)

    st.caption(f"**Log file:** `{path}`")
    st.caption(f"**Trace file:** `{_trace_path(identity_id)}`")

    entries = _read_log_entries(identity_id)
    traces  = _read_traces(identity_id)
    # Build a user_timestamp→steps lookup — stable key that survives entry deletions
    trace_by_ts: dict[str, list[dict]] = {
        t["user_timestamp"]: t["steps"]
        for t in traces
        if "user_timestamp" in t
    }

    st.caption(f"**Entries on disk:** {len(entries) if entries is not None else 'parse error'} "
               f"| **Trace snapshots:** {len(traces)}")
    st.divider()

    raw_tab, structured_tab = st.tabs(["Raw JSON", "Structured"])

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    with raw_tab:
        current_raw = _read_log_raw(identity_id)
        edited = st.text_area("JSON", value=current_raw, height=550,
                              label_visibility="collapsed")
        s_col, _ = st.columns([1, 3])
        with s_col:
            if st.button("Save", key="_logs_raw_save", use_container_width=True):
                _save_raw(identity_id, edited)

    # ── Structured ────────────────────────────────────────────────────────────
    with structured_tab:
        if entries is None:
            st.error("File is not valid JSON. Fix it in the Raw JSON tab.")
            return
        if len(entries) == 0:
            st.info("Log is empty.")
            return

        pairs = _build_pairs(entries)
        total = len(pairs)
        st.caption(f"**{total} message{'s' if total != 1 else ''}** "
                   f"({len(entries)} entries on disk)")

        # ── Delete confirmation dialog ─────────────────────────────────────
        # _confirm_pair holds the pair number pending confirmation.
        # _delete_confirmed is set True by the dialog; main loop does the actual work.
        if st.session_state.get("_delete_confirmed"):
            cp      = st.session_state._confirm_pair
            cp_data = next(((idxs, ip) for pn, idxs, ip in pairs if pn == cp), None)
            if cp_data:
                cp_idxs, _ = cp_data
                remaining = [e for i, e in enumerate(entries) if i not in cp_idxs]
                _save_entries(identity_id, remaining)
                user_ts = entries[cp_idxs[0]].get("timestamp", "")
                _delete_trace_for_timestamp(identity_id, user_ts)
                st.session_state.history = Identity.load_history(identity_id)
                for k in list(st.session_state.keys()):
                    if k.startswith("_lc_") or k.startswith("_lr_"):
                        del st.session_state[k]
            st.session_state._confirm_pair    = None
            st.session_state._delete_confirmed = False
            st.rerun()

        if st.session_state._confirm_pair is not None:
            cp      = st.session_state._confirm_pair
            cp_data = next(((idxs, ip) for pn, idxs, ip in pairs if pn == cp), None)
            if cp_data:
                cp_idxs, cp_is_pair = cp_data
                _delete_confirm_dialog(cp, cp_idxs, entries, cp_is_pair)

        updated: dict[int, dict] = {}

        for p_num, raw_idxs, is_pair in pairs:
            ts_label   = entries[raw_idxs[0]].get("timestamp", "")
            user_ts    = entries[raw_idxs[0]].get("timestamp", "")
            turn_trace = trace_by_ts.get(user_ts)

            user_q     = entries[raw_idxs[0]].get("content", "")
            # Collapse whitespace and trim to keep the label concise
            user_q_preview = " ".join(user_q.split())
            if len(user_q_preview) > 60:
                user_q_preview = user_q_preview[:60].rstrip() + "…"

            with st.expander(f"#{p_num}  —  {ts_label}  —  {user_q_preview}", expanded=False):
                if is_pair:
                    u_idx, a_idx = raw_idxs
                    u_e, a_e     = entries[u_idx], entries[a_idx]
                    u_key = f"_lc_{u_idx}_{u_e.get('timestamp','').replace(':','').replace('.','')}"
                    a_key = f"_lc_{a_idx}_{a_e.get('timestamp','').replace(':','').replace('.','')}"

                    st.markdown(f"**Q** `{u_e.get('timestamp','')}`")
                    new_u = st.text_area("Q", value=u_e.get("content",""),
                                         height=90, key=u_key,
                                         label_visibility="collapsed")
                    st.divider()
                    st.markdown(f"**A** `{a_e.get('timestamp','')}`")
                    new_a = st.text_area("A", value=a_e.get("content",""),
                                         height=90, key=a_key,
                                         label_visibility="collapsed")
                else:
                    r_idx = raw_idxs[0]
                    e     = entries[r_idx]
                    role  = e.get("role", "user")
                    ts    = e.get("timestamp", "")
                    r_key = f"_lc_{r_idx}_{ts.replace(':','').replace('.','')}"
                    nr = st.selectbox("Role", ["user", "assistant"],
                                      index=0 if role == "user" else 1,
                                      key=f"_lr_{r_key}")
                    nc = st.text_area("Content", value=e.get("content",""),
                                      height=100, key=r_key,
                                      label_visibility="collapsed")

                st.divider()
                save_col, del_col, trace_col = st.columns([1, 1, 1])
                with save_col:
                    if st.button("Save", key=f"_save_{p_num}", use_container_width=True):
                        # Build the updated full entries list, patching only this pair
                        new_entries = list(entries)
                        if is_pair:
                            new_entries[u_idx] = {"timestamp": u_e.get("timestamp",""),
                                                  "role": "user", "content": new_u}
                            new_entries[a_idx] = {"timestamp": a_e.get("timestamp",""),
                                                  "role": "assistant", "content": new_a}
                        else:
                            new_entries[r_idx] = {"timestamp": ts, "role": nr, "content": nc}
                        _save_entries(identity_id, new_entries)
                        if identity_id == st.session_state.identity_id:
                            st.session_state.history = Identity.load_history(identity_id)
                        # Purge stale widget keys for this entry
                        for k in list(st.session_state.keys()):
                            if k.startswith("_lc_") or k.startswith("_lr_"):
                                del st.session_state[k]
                        st.rerun()
                with del_col:
                    if st.button("Delete", key=f"_del_{p_num}",
                                 type="secondary", use_container_width=True):
                        st.session_state._confirm_pair = p_num
                        st.rerun()
                with trace_col:
                    if turn_trace:
                        if st.button("Trace", key=f"_trace_{p_num}",
                                     use_container_width=True):
                            st.session_state._view_trace_ts = user_ts
                            st.rerun()
                    else:
                        st.button("Trace", key=f"_trace_{p_num}",
                                  use_container_width=True, disabled=True)


        # Trace dialog — read-only, no write + rerun issue
        if st.session_state.get("_view_trace_ts") is not None:
            vts      = st.session_state._view_trace_ts
            vt_steps = trace_by_ts.get(vts)
            if vt_steps:
                _show_trace_dialog(vts, vt_steps)
            st.session_state._view_trace_ts = None


def _save_raw(identity_id: str, raw_text: str) -> None:
    path = _log_path(identity_id)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON — not saved: {exc}")
        return
    if not isinstance(parsed, list):
        st.error("Top-level value must be a JSON array — not saved.")
        return
    for idx, e in enumerate(parsed):
        if not isinstance(e, dict):
            st.error(f"Entry {idx} is not an object — not saved."); return
        if e.get("role") not in ("user", "assistant"):
            st.error(f"Entry {idx} invalid role {e.get('role')!r} — not saved."); return
        if "content" not in e:
            st.error(f"Entry {idx} missing 'content' — not saved."); return
        if "timestamp" not in e:
            st.error(f"Entry {idx} missing 'timestamp' — not saved."); return
    _save_entries(identity_id, parsed)
    if identity_id == st.session_state.identity_id:
        st.session_state.history = Identity.load_history(identity_id)
    st.success(f"Saved {len(parsed)} entries to `{path}`")
    st.rerun()


# ---------------------------------------------------------------------------
# Trace viewer dialog
# ---------------------------------------------------------------------------

@st.dialog("Trace", width="large")
def _show_trace_dialog(user_ts: str, steps: list[dict]) -> None:
    st.caption(f"User message timestamp: `{user_ts}`")
    st.divider()
    _render_trace_steps(steps)


# ---------------------------------------------------------------------------
# Delete message confirmation dialog
# — dialog only sets flags; actual delete happens in the main render loop
# ---------------------------------------------------------------------------

@st.dialog("Delete message", width="small")
def _delete_confirm_dialog(p_num: int, raw_idxs: list[int],
                            entries: list[dict], is_pair: bool) -> None:
    q_prev = entries[raw_idxs[0]].get("content", "")
    st.warning(
        f"Delete message **#{p_num}**?"
        + (" This removes both Q and A." if is_pair else "")
    )
    st.caption(f"Q: {q_prev[:120]}{'…' if len(q_prev) > 120 else ''}")
    if is_pair and len(raw_idxs) > 1:
        a_prev = entries[raw_idxs[1]].get("content", "")
        st.caption(f"A: {a_prev[:120]}{'…' if len(a_prev) > 120 else ''}")
    st.divider()
    yes_c, no_c = st.columns(2)
    with yes_c:
        if st.button("Delete", type="primary", use_container_width=True,
                     key="_dlg_del_yes"):
            # Set flag — main render loop will do the actual delete + rerun
            st.session_state._delete_confirmed = True
            st.rerun()
    with no_c:
        if st.button("Cancel", use_container_width=True, key="_dlg_del_no"):
            st.session_state._confirm_pair     = None
            st.session_state._delete_confirmed = False
            st.rerun()


# ---------------------------------------------------------------------------
# Delete identity dialog
# ---------------------------------------------------------------------------

@st.dialog("Delete identity", width="small")
def _confirm_delete_identity_dialog(identity_id: str) -> None:
    st.warning(
        f"Permanently delete identity **'{identity_id}'**?\n\n"
        "This removes the log file and all traces for this identity. "
        "This cannot be undone."
    )
    yes_c, no_c = st.columns(2)
    with yes_c:
        if st.button("Delete", type="primary", use_container_width=True,
                     key="_del_identity_yes"):
            # Delete log file
            try:
                Identity.clear_identity(identity_id)
            except Exception:
                pass
            # Delete traces file
            tp = _trace_path(identity_id)
            if os.path.exists(tp):
                os.remove(tp)
            # Delete tokens file
            tkp = _tokens_path(identity_id)
            if os.path.exists(tkp):
                os.remove(tkp)
            # Switch to default
            st.session_state.identity_id     = "default"
            st.session_state.history         = Identity.load_history("default")
            st.session_state.last_payload    = None
            st.session_state.last_response   = None
            st.session_state.trace_log       = []
            st.session_state._confirm_pair   = None
            st.session_state.session_tokens  = {"prompt": 0, "completion": 0, "total": 0}
            st.session_state._confirm_delete_identity = False
            _restore_token_totals("default")
            st.rerun()
    with no_c:
        if st.button("Cancel", use_container_width=True, key="_del_identity_no"):
            st.session_state._confirm_delete_identity = False
            st.rerun()


# ---------------------------------------------------------------------------
# Delete message confirmation dialog
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Falcon", layout="wide", page_icon="🦅")
    _init_session_state()

    # ── Global CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ── Base ── */
    [data-testid="stAppViewContainer"] {
        background: #0f1117;
    }
    [data-testid="stSidebar"] {
        background: #161b27;
        border-right: 1px solid #1e2535;
    }

    /* ── Hide default Streamlit decorations ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }

    /* ── App header bar ── */
    .falcon-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 18px 4px 10px 4px;
        border-bottom: 1px solid #1e2535;
        margin-bottom: 4px;
    }
    .falcon-header h1 {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .falcon-header .model-badge {
        font-size: 0.72rem;
        color: #64748b;
        background: #1e2535;
        padding: 2px 8px;
        border-radius: 999px;
        font-family: monospace;
        margin-left: auto;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 4px 0 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1;
        line-height: 1.65;
        font-size: 0.95rem;
    }
    /* User bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #1a2235 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        margin: 6px 0 !important;
    }
    /* Assistant bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: transparent !important;
        border-radius: 12px !important;
        padding: 8px 4px !important;
        margin: 4px 0 !important;
    }

    /* ── Chat input — fixed to bottom of viewport ── */
    [data-testid="stBottom"] {
        background: #0f1117;
        border-top: 1px solid #1e2535;
        padding: 10px 0 6px 0;
    }
    [data-testid="stChatInput"] textarea {
        background: #161b27 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 0.93rem !important;
        padding: 12px 16px !important;
        transition: border-color 0.2s;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    }

    /* ── Clear button (subtle) ── */
    .clear-btn button {
        background: transparent !important;
        border: 1px solid #2d3748 !important;
        color: #64748b !important;
        border-radius: 8px !important;
        font-size: 0.78rem !important;
        padding: 4px 14px !important;
        transition: all 0.18s;
    }
    .clear-btn button:hover {
        border-color: #ef4444 !important;
        color: #ef4444 !important;
        background: rgba(239,68,68,0.06) !important;
    }

    /* ── Confirm clear warning ── */
    .confirm-bar {
        background: #1e1a0f;
        border: 1px solid #78350f;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid #1e2535;
        gap: 0;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent !important;
        color: #64748b !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 8px 20px !important;
        border-bottom: 2px solid transparent !important;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        color: #e2e8f0 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* ── Sidebar labels ── */
    .sidebar-section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569;
        margin: 14px 0 6px 0;
    }
    .token-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3px 0;
        font-size: 0.8rem;
        color: #64748b;
    }
    .token-row span.val {
        font-family: monospace;
        color: #94a3b8;
    }

    /* ── Selectbox / text input ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stTextInput"] input {
        background: #1e2535 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-size: 0.85rem !important;
    }

    /* ── Expanders (Logs tab) ── */
    [data-testid="stExpander"] {
        background: #161b27 !important;
        border: 1px solid #1e2535 !important;
        border-radius: 10px !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stExpander"] summary {
        color: #94a3b8 !important;
        font-size: 0.83rem !important;
        font-family: monospace !important;
    }

    /* ── Scrollable chat area ── */
    .chat-scroll-area {
        overflow-y: auto;
        padding-right: 4px;
    }

    /* ── Empty state ── */
    .empty-chat {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 20px;
        color: #334155;
        text-align: center;
    }
    .empty-chat .icon { font-size: 2.8rem; margin-bottom: 12px; }
    .empty-chat .title { font-size: 1.1rem; font-weight: 600; color: #475569; margin-bottom: 6px; }
    .empty-chat .sub { font-size: 0.83rem; color: #334155; }
    </style>
    """, unsafe_allow_html=True)

    if not st.session_state._loaded_initial_history:
        _load_identity_history(st.session_state.identity_id)
        _restore_token_totals(st.session_state.identity_id)
        st.session_state._loaded_initial_history = True

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="falcon-header"><h1>🦅 Falcon</h1></div>', unsafe_allow_html=True)

        # Identity
        st.markdown('<div class="sidebar-section-label">Identity</div>', unsafe_allow_html=True)

        existing_ids = [i for i in Identity.list_identities() if "." not in i]
        all_ids = sorted(set(existing_ids) | {st.session_state.identity_id, "default"})
        current_idx = all_ids.index(st.session_state.identity_id) if st.session_state.identity_id in all_ids else 0

        chosen = st.selectbox(
            "Identity", options=all_ids, index=current_idx,
            label_visibility="collapsed",
        )
        if chosen != st.session_state.identity_id:
            st.session_state.identity_id   = chosen
            st.session_state.history       = Identity.load_history(chosen)
            st.session_state.last_payload  = None
            st.session_state.last_response = None
            st.session_state.trace_log     = []
            st.session_state._confirm_pair = None
            _restore_token_totals(chosen)
            st.rerun()

        with st.form("_new_identity_form", clear_on_submit=True):
            new_name = st.text_input(
                "New identity",
                placeholder="New identity name…",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("＋ Create", use_container_width=True)

        if submitted:
            new_id = new_name.strip()
            if not new_id:
                st.error("Enter an identity name.")
            elif "." in new_id:
                st.error("Name cannot contain a dot.")
            elif any(c in new_id for c in ("/", "\\", "\x00")) or ".." in new_id:
                st.error("Invalid characters in name.")
            elif new_id in all_ids:
                st.session_state.identity_id   = new_id
                st.session_state.history       = Identity.load_history(new_id)
                st.session_state.last_payload  = None
                st.session_state.last_response = None
                st.session_state.trace_log     = []
                st.session_state._confirm_pair = None
                _restore_token_totals(new_id)
                st.rerun()
            else:
                st.session_state.identity_id    = new_id
                st.session_state.history        = []
                st.session_state.last_payload   = None
                st.session_state.last_response  = None
                st.session_state.trace_log      = []
                st.session_state._confirm_pair  = None
                st.session_state.session_tokens = {"prompt": 0, "completion": 0, "total": 0}
                os.makedirs(Config.log_dir, exist_ok=True)
                log_file = _log_path(new_id)
                if not os.path.exists(log_file):
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write("[]")
                st.rerun()

        can_delete = st.session_state.identity_id != "default"
        if st.button(
            f"🗑 Delete '{st.session_state.identity_id}'",
            key="_delete_identity_btn",
            type="secondary",
            use_container_width=True,
            disabled=not can_delete,
        ):
            st.session_state._confirm_delete_identity = True

        if st.session_state.get("_confirm_delete_identity"):
            _confirm_delete_identity_dialog(st.session_state.identity_id)

        # Model
        st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
        if Config.available_models:
            sel = st.selectbox(
                "Model", options=Config.available_models,
                index=(Config.available_models.index(st.session_state.selected_model)
                       if st.session_state.selected_model in Config.available_models else 0),
                label_visibility="collapsed",
            )
            st.session_state.selected_model = sel
        else:
            st.caption(st.session_state.selected_model)

        # Session stats
        st.markdown('<div class="sidebar-section-label">Session</div>', unsafe_allow_html=True)
        tok = st.session_state.session_tokens
        msg_count = _pair_count(st.session_state.history)
        st.markdown(f"""
        <div class="token-row"><span>Messages</span><span class="val">{msg_count}</span></div>
        <div class="token-row"><span>Prompt tokens</span><span class="val">{tok.get('prompt', 0):,}</span></div>
        <div class="token-row"><span>Completion tokens</span><span class="val">{tok.get('completion', 0):,}</span></div>
        <div class="token-row" style="border-top:1px solid #1e2535;margin-top:4px;padding-top:6px">
            <span>Total</span><span class="val" style="color:#60a5fa">{tok.get('total', 0):,}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)
        st.caption(f"`{_log_path(st.session_state.identity_id)}`")

    # ── Chat input — must be at page level (not inside a tab) for bottom-sticky behaviour ──
    user_input = st.chat_input("Message Falcon…")

    # ── Main area ─────────────────────────────────────────────────────────────
    tab_chat, tab_logs = st.tabs(["Chat", "Logs"])

    with tab_chat:
        _render_chat_tab(user_input)

    with tab_logs:
        _render_logs_tab()


if __name__ == "__main__":
    main()
