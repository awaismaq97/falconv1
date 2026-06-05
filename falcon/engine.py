"""
Engine module for Falcon V1.

Provides two public functions:
  - build_payload(system_prompt, messages)  — build the exact {role, content} list
    that will be sent to the Groq API, with no hidden injections or modifications.
  - run_inference(model_name, system_prompt, messages, groq_api_key)  — call the
    Groq API via LangChain ChatGroq and return the response text alongside the
    exact payload that was sent.

Design principles enforced here:
  - Neutrality: no content is added to any message; no SystemMessage is injected
    when system_prompt is empty or whitespace-only.
  - Transparency: raw_payload in the return value is the exact list sent to the
    model, suitable for display in the sidebar via st.json().
  - Simplicity: no chains, no agents, no tools, no caching — one llm.invoke() call.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq


def build_payload(system_prompt: str, messages: list[dict]) -> list[dict]:
    """Build the exact list of {role, content} dicts to be sent to the model.

    Behaviour:
    - If `system_prompt` is non-empty and not whitespace-only, prepends one entry
      with role="system" and content=system_prompt, with no modification (REQ 2.1).
    - If `system_prompt` is empty or whitespace-only, no system entry is included
      in the returned list (REQ 2.2, REQ 2.5, REQ 4.1).
    - All entries from `messages` are included in their original order with their
      content values unchanged (REQ 2.3, REQ 4.3).
    - Total length equals len(messages) + (1 if system_prompt.strip() else 0) (REQ 2.4).

    Args:
        system_prompt: The system prompt string. May be empty.
        messages: List of {role, content} dicts representing conversation history.

    Returns:
        A list of {role, content} dicts representing the full payload.
    """
    payload: list[dict] = []

    if system_prompt and system_prompt.strip():
        # REQ 2.1, REQ 4.2 — include system entry only when non-empty / non-whitespace
        payload.append({"role": "system", "content": system_prompt})

    # REQ 2.3, REQ 4.3 — copy messages verbatim; do not normalise content
    for message in messages:
        payload.append({"role": message["role"], "content": message["content"]})

    return payload


def run_inference(
    model_name: str,
    system_prompt: str,
    messages: list[dict],
    groq_api_key: str,
) -> dict:
    """Call the Groq API and return the response and the exact outbound payload.

    Returns {"response": str, "raw_payload": list[dict], "response_metadata": dict,
             "usage": dict, "id": str|None}
    """
    raw_payload = build_payload(system_prompt, messages)

    lc_messages = []
    for entry in raw_payload:
        role, content = entry["role"], entry["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unknown role in message payload: {role!r}")

    llm = ChatGroq(
        model=model_name,
        api_key=groq_api_key,
        temperature=0,
        top_p=1,
        stop_sequences=None,
        max_tokens=1,
    )

    result = llm.invoke(lc_messages)

    usage = {}
    if hasattr(result, "response_metadata") and result.response_metadata:
        usage = result.response_metadata.get("token_usage") or result.response_metadata.get("usage") or {}

    return {
        "response": result.content,
        "raw_payload": raw_payload,
        "response_metadata": getattr(result, "response_metadata", {}),
        "usage": usage,
        "id": getattr(result, "id", None),
    }


class _StreamResult:
    """Wraps stream_inference so callers can read token usage after the stream ends.

    Usage::

        gen = stream_inference(...)
        text = st.write_stream(gen)        # exhausts the generator
        usage = gen.usage                  # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    """

    def __init__(self, model_name, system_prompt, messages, groq_api_key):
        self.usage: dict = {}
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._messages = messages
        self._groq_api_key = groq_api_key
        self._gen = self._run()

    def _run(self):
        raw_payload = build_payload(self._system_prompt, self._messages)

        lc_messages = []
        for entry in raw_payload:
            role, content = entry["role"], entry["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown role in message payload: {role!r}")

        llm = ChatGroq(
            model=self._model_name,
            api_key=self._groq_api_key,
            temperature=0,
            top_p=1,
            stop_sequences=None,
            max_tokens=1,
        )

        # State machine for stripping <think>...</think> blocks mid-stream.
        # We buffer text while inside a tag and discard it when the closing
        # tag arrives.  Any buffered text that turns out NOT to be a tag
        # (e.g. a lone '<') is flushed normally.
        in_think = False
        buf = ""          # accumulates text while we're unsure / inside a tag

        for chunk in llm.stream(lc_messages):
            # Capture usage metadata carried on the final (or any) chunk
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                meta = chunk.usage_metadata
                self.usage = {
                    "prompt_tokens":     meta.get("input_tokens", 0),
                    "completion_tokens": meta.get("output_tokens", 0),
                    "total_tokens":      meta.get("total_tokens", 0),
                }

            if not chunk.content:
                continue

            buf += chunk.content

            # Process the buffer, yielding safe text and discarding think blocks
            out = ""
            while buf:
                if in_think:
                    end = buf.find("</think>")
                    if end == -1:
                        # Entire buffer is inside <think>; wait for more chunks
                        break
                    # Found closing tag — discard everything up to and including it
                    buf = buf[end + len("</think>"):]
                    in_think = False
                else:
                    start = buf.find("<think>")
                    if start == -1:
                        # No think tag anywhere — safe to emit everything
                        out += buf
                        buf = ""
                    else:
                        # Emit text before the tag, then enter think mode
                        out += buf[:start]
                        buf = buf[start + len("<think>"):]
                        in_think = True

            if out:
                yield out

    # Make the object itself iterable so st.write_stream / for-loops work directly
    def __iter__(self):
        return self._gen

    def __next__(self):
        return next(self._gen)


def stream_inference(
    model_name: str,
    system_prompt: str,
    messages: list[dict],
    groq_api_key: str,
) -> "_StreamResult":
    """Stream tokens from the Groq API.

    Returns a :class:`_StreamResult` which is iterable (pass it to
    ``st.write_stream`` or iterate normally).  After the iterator is exhausted,
    read ``.usage`` for token counts::

        gen = stream_inference(...)
        text = st.write_stream(gen)
        print(gen.usage)   # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}

    Yields:
        str: each text chunk from the model.
    """
    return _StreamResult(model_name, system_prompt, messages, groq_api_key)
