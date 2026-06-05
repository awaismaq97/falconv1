"""
Unit tests for falcon/engine.py.

Covers:
- build_payload with and without system prompt (REQ 2.1–2.5, 4.1–4.3)
- run_inference returns correct dict structure with mocked ChatGroq (REQ 2.1–2.5, 4.1–4.3)
- run_inference propagates ChatGroq exceptions to the caller
"""

from unittest.mock import MagicMock, patch

import pytest

from falcon.engine import build_payload, run_inference


# ---------------------------------------------------------------------------
# build_payload tests
# ---------------------------------------------------------------------------

class TestBuildPayload:
    """Tests for build_payload(system_prompt, messages) — REQ 2.1–2.5, 4.1–4.3."""

    # --- with non-empty system prompt ---

    def test_non_empty_prompt_is_prepended(self):
        """REQ 2.1: non-empty system_prompt → first entry is {"role": "system", ...}."""
        payload = build_payload("You are a pirate.", [])
        assert len(payload) == 1
        assert payload[0] == {"role": "system", "content": "You are a pirate."}

    def test_system_prompt_content_unmodified(self):
        """REQ 2.1, 4.2: system_prompt content is passed through without modification."""
        prompt = "  extra spaces and\nnewlines  "
        payload = build_payload(prompt, [])
        assert payload[0]["content"] == prompt

    def test_messages_follow_system_entry(self):
        """REQ 2.3: user/assistant messages follow the system entry in order."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        payload = build_payload("sys prompt", messages)
        assert len(payload) == 4
        assert payload[0] == {"role": "system", "content": "sys prompt"}
        assert payload[1] == {"role": "user", "content": "Hello"}
        assert payload[2] == {"role": "assistant", "content": "Hi there"}
        assert payload[3] == {"role": "user", "content": "How are you?"}

    def test_payload_length_with_system_prompt(self):
        """REQ 2.4: len(payload) == len(messages) + 1 when system_prompt non-empty."""
        messages = [{"role": "user", "content": "msg"}] * 5
        payload = build_payload("prompt", messages)
        assert len(payload) == 6

    # --- with empty or whitespace-only system prompt ---

    def test_empty_prompt_no_system_entry(self):
        """REQ 2.2, 2.5: empty string → no system entry."""
        messages = [{"role": "user", "content": "hi"}]
        payload = build_payload("", messages)
        assert len(payload) == 1
        assert payload[0] == {"role": "user", "content": "hi"}

    def test_whitespace_only_prompt_no_system_entry(self):
        """REQ 4.1: whitespace-only prompt → no system entry."""
        messages = [{"role": "user", "content": "hi"}]
        payload = build_payload("   \t\n  ", messages)
        assert all(e["role"] != "system" for e in payload)
        assert len(payload) == 1

    def test_empty_prompt_empty_messages_returns_empty_list(self):
        """REQ 2.4: empty prompt + empty messages → empty payload."""
        payload = build_payload("", [])
        assert payload == []

    def test_non_empty_prompt_empty_messages_returns_one_entry(self):
        """REQ 2.4: non-empty prompt + empty messages → payload length 1."""
        payload = build_payload("system only", [])
        assert len(payload) == 1
        assert payload[0]["role"] == "system"

    def test_payload_length_without_system_prompt(self):
        """REQ 2.4: len(payload) == len(messages) when system_prompt is empty."""
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(7)]
        payload = build_payload("", messages)
        assert len(payload) == 7

    # --- message content preservation ---

    def test_message_content_is_not_modified(self):
        """REQ 4.3: message content values are passed through with no modification."""
        messages = [
            {"role": "user", "content": "  leading and trailing spaces  "},
            {"role": "assistant", "content": "line1\nline2\nline3"},
            {"role": "user", "content": ""},
        ]
        payload = build_payload("", messages)
        for original, result in zip(messages, payload):
            assert result["content"] == original["content"]

    def test_original_messages_list_is_not_mutated(self):
        """build_payload must not mutate the input messages list."""
        messages = [{"role": "user", "content": "original"}]
        original_copy = [dict(m) for m in messages]
        build_payload("prompt", messages)
        assert messages == original_copy


# ---------------------------------------------------------------------------
# run_inference tests (mocked ChatGroq)
# ---------------------------------------------------------------------------

class TestRunInference:
    """Tests for run_inference() with mocked ChatGroq — REQ 2.1–2.5, 4.1–4.3."""

    def _make_mock_llm(self, response_text: str = "mocked response"):
        """Return a mock ChatGroq instance whose invoke() returns an AIMessage-like object."""
        mock_result = MagicMock()
        mock_result.content = response_text
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_result
        return mock_llm

    @patch("falcon.engine.ChatGroq")
    def test_returns_response_and_raw_payload(self, mock_chatgroq_cls):
        """run_inference returns dict with 'response' str and 'raw_payload' list."""
        mock_chatgroq_cls.return_value = self._make_mock_llm("hello from model")
        messages = [{"role": "user", "content": "hi"}]

        result = run_inference("test-model", "", messages, "fake-key")

        assert isinstance(result, dict)
        assert "response" in result
        assert "raw_payload" in result
        assert result["response"] == "hello from model"
        assert isinstance(result["raw_payload"], list)

    @patch("falcon.engine.ChatGroq")
    def test_raw_payload_matches_build_payload_output(self, mock_chatgroq_cls):
        """raw_payload returned by run_inference equals build_payload(system_prompt, messages)."""
        mock_chatgroq_cls.return_value = self._make_mock_llm()
        messages = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow up"},
        ]
        system_prompt = "Be concise."

        result = run_inference("model", system_prompt, messages, "key")

        expected_payload = build_payload(system_prompt, messages)
        assert result["raw_payload"] == expected_payload

    @patch("falcon.engine.ChatGroq")
    def test_no_system_entry_in_payload_for_empty_prompt(self, mock_chatgroq_cls):
        """REQ 4.1: no system entry in raw_payload when system_prompt is empty."""
        mock_chatgroq_cls.return_value = self._make_mock_llm()
        messages = [{"role": "user", "content": "test"}]

        result = run_inference("model", "", messages, "key")

        assert all(e["role"] != "system" for e in result["raw_payload"])

    @patch("falcon.engine.ChatGroq")
    def test_system_entry_present_for_non_empty_prompt(self, mock_chatgroq_cls):
        """REQ 2.1: system entry is first in raw_payload when system_prompt is non-empty."""
        mock_chatgroq_cls.return_value = self._make_mock_llm()
        messages = [{"role": "user", "content": "hello"}]
        system_prompt = "Custom system instruction."

        result = run_inference("model", system_prompt, messages, "key")

        assert result["raw_payload"][0] == {"role": "system", "content": system_prompt}

    @patch("falcon.engine.ChatGroq")
    def test_chatgroq_instantiated_with_correct_args(self, mock_chatgroq_cls):
        """ChatGroq is instantiated with the provided model_name, groq_api_key, and generation params."""
        mock_chatgroq_cls.return_value = self._make_mock_llm()

        run_inference("llama3-70b-8192", "", [], "my-secret-key")

        mock_chatgroq_cls.assert_called_once_with(
            model="llama3-70b-8192",
            api_key="my-secret-key",
            temperature=0,
            top_p=1,
            stop_sequences=None,
        )

    @patch("falcon.engine.ChatGroq")
    def test_invoke_called_once(self, mock_chatgroq_cls):
        """llm.invoke() is called exactly once per run_inference call."""
        mock_llm = self._make_mock_llm()
        mock_chatgroq_cls.return_value = mock_llm

        run_inference("model", "prompt", [{"role": "user", "content": "hi"}], "key")

        mock_llm.invoke.assert_called_once()

    @patch("falcon.engine.ChatGroq")
    def test_exception_from_invoke_is_propagated(self, mock_chatgroq_cls):
        """Exceptions from ChatGroq.invoke() must propagate to the caller."""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API failure")
        mock_chatgroq_cls.return_value = mock_llm

        with pytest.raises(RuntimeError, match="API failure"):
            run_inference("model", "", [{"role": "user", "content": "hi"}], "key")

    @patch("falcon.engine.ChatGroq")
    def test_response_is_string(self, mock_chatgroq_cls):
        """The 'response' field in the result is a string."""
        mock_chatgroq_cls.return_value = self._make_mock_llm("text output")

        result = run_inference("model", "", [{"role": "user", "content": "q"}], "key")

        assert isinstance(result["response"], str)
        assert result["response"] == "text output"

    @patch("falcon.engine.ChatGroq")
    def test_lc_messages_constructed_correctly(self, mock_chatgroq_cls):
        """
        LangChain message objects passed to invoke() match the payload roles.
        Verifies that SystemMessage, HumanMessage, AIMessage are used correctly.
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        mock_llm = self._make_mock_llm()
        mock_chatgroq_cls.return_value = mock_llm

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
            {"role": "user", "content": "again"},
        ]
        run_inference("model", "sys", messages, "key")

        call_args = mock_llm.invoke.call_args[0][0]
        assert isinstance(call_args[0], SystemMessage)
        assert call_args[0].content == "sys"
        assert isinstance(call_args[1], HumanMessage)
        assert call_args[1].content == "hello"
        assert isinstance(call_args[2], AIMessage)
        assert call_args[2].content == "world"
        assert isinstance(call_args[3], HumanMessage)
        assert call_args[3].content == "again"

    @patch("falcon.engine.ChatGroq")
    def test_empty_messages_empty_prompt_invokes_with_empty_list(self, mock_chatgroq_cls):
        """When both messages and system_prompt are empty, invoke is called with []."""
        mock_chatgroq_cls.return_value = self._make_mock_llm()

        result = run_inference("model", "", [], "key")

        mock_chatgroq_cls.return_value.invoke.assert_called_once_with([])
        assert result["raw_payload"] == []
