"""
Engine module for Falcon V1.

Provides two public functions:
  - build_payload(system_prompt, messages)  — build the exact {role, content} list
    that will be sent to the model, with no hidden injections or modifications.
  - stream_inference(model_name, system_prompt, messages, api_key)  — stream tokens
    from OpenRouter and return a _StreamResult whose .usage is readable after
    the stream is exhausted.

Uses the openai SDK pointed at OpenRouter's base URL (OpenAI-compatible).

Design principles:
  - Neutrality: no content is added; no system message injected when empty.
  - Transparency: raw_payload is the exact list sent, suitable for display.
  - Simplicity: no chains, no agents, no tools — one streaming call.
"""

import openai


_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_payload(system_prompt: str, messages: list[dict]) -> list[dict]:
    """Build the exact list of {role, content} dicts to be sent to the model.

    - If system_prompt is non-empty/non-whitespace, prepends a system entry.
    - All messages are included unchanged.
    """
    payload: list[dict] = []

    if system_prompt and system_prompt.strip():
        payload.append({"role": "system", "content": system_prompt})

    for message in messages:
        payload.append({"role": message["role"], "content": message["content"]})

    return payload


class _StreamResult:
    """Wraps a streaming OpenRouter call so callers can read token usage after
    the stream ends.

    Usage::

        gen = stream_inference(...)
        text = st.write_stream(gen)   # exhausts the generator
        usage = gen.usage             # {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
    """

    def __init__(self, model_name: str, system_prompt: str,
                 messages: list[dict], api_key: str):
        self.usage: dict = {}
        self._model_name   = model_name
        self._system_prompt = system_prompt
        self._messages     = messages
        self._api_key      = api_key
        self._gen          = self._run()

    def _run(self):
        raw_payload = build_payload(self._system_prompt, self._messages)

        client = openai.OpenAI(
            api_key=self._api_key,
            base_url=_OPENROUTER_BASE_URL,
            default_headers={
                "Authorization": f"Bearer {self._api_key}",
                "HTTP-Referer": "https://github.com/falcon",
                "X-Title": "Falcon",
            },
        )

        stream = client.chat.completions.create(
            model=self._model_name,
            messages=raw_payload,
            temperature=0,
            top_p=1,
            stream=True,
            stream_options={"include_usage": True},
        )

        # State machine for stripping <think>...</think> blocks mid-stream.
        in_think = False
        buf = ""

        for chunk in stream:
            # Capture usage from the final chunk (sent when stream_options include_usage=True)
            if chunk.usage:
                self.usage = {
                    "prompt_tokens":     chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens":      chunk.usage.total_tokens,
                }

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta or not delta.content:
                continue

            buf += delta.content

            # Yield text outside <think>...</think> blocks
            out = ""
            while buf:
                if in_think:
                    end = buf.find("</think>")
                    if end == -1:
                        break  # still inside think block, wait for more
                    buf = buf[end + len("</think>"):]
                    in_think = False
                else:
                    start = buf.find("<think>")
                    if start == -1:
                        out += buf
                        buf = ""
                    else:
                        out += buf[:start]
                        buf = buf[start + len("<think>"):]
                        in_think = True

            if out:
                yield out

    def __iter__(self):
        return self._gen

    def __next__(self):
        return next(self._gen)


def stream_inference(
    model_name: str,
    system_prompt: str,
    messages: list[dict],
    api_key: str,
) -> "_StreamResult":
    """Stream tokens from OpenRouter.

    Returns a _StreamResult iterable. After exhausting it, read .usage::

        gen = stream_inference(...)
        text = st.write_stream(gen)
        print(gen.usage)
    """
    return _StreamResult(model_name, system_prompt, messages, api_key)
