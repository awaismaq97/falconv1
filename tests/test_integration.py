"""
Integration tests for Falcon V1.

Task 8.4: Property test for Context Continuity (Property 6)
  Validates: Requirements 1.9, 1.10, 1.11

Task 8.5: Full send-flow integration tests
  Tests that Logger → Identity → Engine (mocked Groq) → Logger
  produces the correct log state and raw_payload shape.
  Validates: Requirements 1.1, 1.9, 1.10, 1.11, 6.1, 6.2
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import falcon.identity as identity_module
import falcon.logger as logger_module
from falcon.engine import build_payload, run_inference
from falcon.identity import load_history
from falcon.logger import append_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_log_dirs(tmp_dir: str):
    original_logger = logger_module._LOG_DIR
    original_identity = identity_module._LOG_DIR
    logger_module._LOG_DIR = tmp_dir
    identity_module._LOG_DIR = tmp_dir
    return original_logger, original_identity


def _restore_log_dirs(orig_logger: str, orig_identity: str):
    logger_module._LOG_DIR = orig_logger
    identity_module._LOG_DIR = orig_identity


def _make_mock_llm(response_text: str = "mocked response"):
    mock_result = MagicMock()
    mock_result.content = response_text
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_result
    return mock_llm


# ---------------------------------------------------------------------------
# Property 6: Context Continuity (task 8.4)
# Validates: Requirements 1.9, 1.10, 1.11
# ---------------------------------------------------------------------------

safe_identity_id = st.from_regex(r"[a-zA-Z][a-zA-Z0-9_]{0,19}", fullmatch=True)
valid_role = st.sampled_from(["user", "assistant"])
content_text = st.text(min_size=0, max_size=200)
message_pair = st.tuples(valid_role, content_text)
message_sequence = st.lists(message_pair, min_size=1, max_size=20)


@given(identity_id=safe_identity_id, messages=message_sequence)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
def test_context_continuity_after_restart(
    identity_id: str,
    messages: list[tuple[str, str]],
):
    """
    Property 6: Context Continuity

    **Validates: Requirements 1.9, 1.10, 1.11**

    Write N messages to a temp log via Logger.append_message; simulate an app
    restart by calling load_history; assert returned history has exactly N
    entries in insertion order.

    No entry is dropped, truncated, or reordered across the simulated restart.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_logger, orig_identity = _patch_log_dirs(tmp_dir)
        try:
            # Write all messages (simulating a live session)
            for role, content in messages:
                append_message(identity_id, role, content)

            # Simulate app restart: fresh load_history call
            loaded = load_history(identity_id)

            # REQ 1.10: exact count preserved
            assert len(loaded) == len(messages), (
                f"After restart: expected {len(messages)} entries, got {len(loaded)}"
            )

            # REQ 1.9, 1.11: no reordering, no truncation
            for idx, (exp_role, exp_content) in enumerate(messages):
                assert loaded[idx]["role"] == exp_role, (
                    f"Entry {idx} role mismatch after restart"
                )
                assert loaded[idx]["content"] == exp_content, (
                    f"Entry {idx} content mismatch after restart"
                )
        finally:
            _restore_log_dirs(orig_logger, orig_identity)


# ---------------------------------------------------------------------------
# Task 8.5: Integration tests for full send flow
# Validates: Requirements 1.1, 1.9, 1.10, 1.11, 6.1, 6.2
# ---------------------------------------------------------------------------

class TestFullSendFlow:
    """
    End-to-end flow without Streamlit:
      Logger.append_message → Identity.load_history → Engine.run_inference (mocked) → Logger.append_message
    """

    @patch("falcon.engine.ChatGroq")
    def test_send_flow_produces_correct_log_state(self, mock_chatgroq_cls, tmp_path, monkeypatch):
        """
        REQ 6.1, 6.2: After a full send flow, the log contains the user message
        followed by the assistant message.
        """
        monkeypatch.setattr(logger_module, "_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(identity_module, "_LOG_DIR", str(tmp_path))

        mock_chatgroq_cls.return_value = _make_mock_llm("assistant reply")

        identity_id = "test_flow"

        # Step 1: log user message
        append_message(identity_id, "user", "user question")

        # Step 2: load full history for engine call
        history = load_history(identity_id)

        # Step 3: run inference
        result = run_inference("model", "", history, "fake-key")
        response_text = result["response"]

        # Step 4: log assistant response
        append_message(identity_id, "assistant", response_text)

        # Verify final log state
        final_history = load_history(identity_id)
        assert len(final_history) == 2
        assert final_history[0]["role"] == "user"
        assert final_history[0]["content"] == "user question"
        assert final_history[1]["role"] == "assistant"
        assert final_history[1]["content"] == "assistant reply"

    @patch("falcon.engine.ChatGroq")
    def test_raw_payload_shape_in_send_flow(self, mock_chatgroq_cls, tmp_path, monkeypatch):
        """
        REQ 2.1–2.5: raw_payload from run_inference has the correct shape:
        system entry (if prompt non-empty) + all prior messages.
        """
        monkeypatch.setattr(logger_module, "_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(identity_module, "_LOG_DIR", str(tmp_path))

        mock_chatgroq_cls.return_value = _make_mock_llm("response")

        identity_id = "payload_test"
        system_prompt = "Be concise."

        append_message(identity_id, "user", "question one")
        history = load_history(identity_id)
        result = run_inference("model", system_prompt, history, "key")

        assert result["raw_payload"][0] == {"role": "system", "content": system_prompt}
        assert result["raw_payload"][1] == {"role": "user", "content": "question one"}
        assert len(result["raw_payload"]) == 2  # system + 1 message

    @patch("falcon.engine.ChatGroq")
    def test_context_continuity_10_messages(self, mock_chatgroq_cls, tmp_path, monkeypatch):
        """
        REQ 1.9, 1.11: With 10 prior messages and an empty system prompt,
        engine receives all 10 prior messages plus the new user message = 11 entries.
        """
        monkeypatch.setattr(logger_module, "_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(identity_module, "_LOG_DIR", str(tmp_path))

        mock_chatgroq_cls.return_value = _make_mock_llm("response")

        identity_id = "context_test"

        # Write 10 prior messages (5 exchanges)
        for i in range(5):
            append_message(identity_id, "user", f"user message {i}")
            append_message(identity_id, "assistant", f"assistant reply {i}")

        # Load full history (10 entries)
        history = load_history(identity_id)
        assert len(history) == 10

        # Add the new user message to history (simulating what app.py does)
        new_user_message = {"role": "user", "content": "the 11th message"}
        messages_to_engine = history + [new_user_message]

        result = run_inference("model", "", messages_to_engine, "key")

        # No system prompt → raw_payload length == 11
        assert len(result["raw_payload"]) == 11
        assert result["raw_payload"][-1]["role"] == "user"
        assert result["raw_payload"][-1]["content"] == "the 11th message"

    @patch("falcon.engine.ChatGroq")
    def test_identity_isolation_in_send_flow(self, mock_chatgroq_cls, tmp_path, monkeypatch):
        """
        REQ 1.1: Messages written for identity A do not appear in identity B's history.
        """
        monkeypatch.setattr(logger_module, "_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(identity_module, "_LOG_DIR", str(tmp_path))

        mock_chatgroq_cls.return_value = _make_mock_llm()

        # Write messages for identity A
        append_message("test_A", "user", "A says hello")
        append_message("test_A", "assistant", "A gets a reply")

        # Write message for identity B
        append_message("test_B", "user", "B says hi")

        history_a = load_history("test_A")
        history_b = load_history("test_B")

        assert len(history_a) == 2
        assert len(history_b) == 1
        assert all(e["content"] != "B says hi" for e in history_a), (
            "A's history must not contain B's messages"
        )
        assert all(e["content"] != "A says hello" for e in history_b), (
            "B's history must not contain A's messages"
        )

    @patch("falcon.engine.ChatGroq")
    def test_app_restart_continuity(self, mock_chatgroq_cls, tmp_path, monkeypatch):
        """
        REQ 1.10: After writing messages in a 'first session', loading history in a
        'second session' returns identical entries — zero loss.
        """
        monkeypatch.setattr(logger_module, "_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(identity_module, "_LOG_DIR", str(tmp_path))

        mock_chatgroq_cls.return_value = _make_mock_llm("model answer")

        identity_id = "restart_test"

        # First session: write some messages
        session1_messages = [
            ("user", "first user message"),
            ("assistant", "first assistant response"),
            ("user", "second user message"),
        ]
        for role, content in session1_messages:
            append_message(identity_id, role, content)

        # Simulate app restart: load from scratch
        loaded_after_restart = load_history(identity_id)

        assert len(loaded_after_restart) == len(session1_messages)
        for idx, (exp_role, exp_content) in enumerate(session1_messages):
            assert loaded_after_restart[idx]["role"] == exp_role
            assert loaded_after_restart[idx]["content"] == exp_content

    @patch("falcon.engine.ChatGroq")
    def test_failed_inference_does_not_log_assistant_entry(self, mock_chatgroq_cls, tmp_path, monkeypatch):
        """
        REQ 6.5: If run_inference raises an exception, no assistant entry is logged.
        (Mirrors what app.py must do — only log on success.)
        """
        monkeypatch.setattr(logger_module, "_LOG_DIR", str(tmp_path))
        monkeypatch.setattr(identity_module, "_LOG_DIR", str(tmp_path))

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("API error")
        mock_chatgroq_cls.return_value = mock_llm

        identity_id = "error_test"

        append_message(identity_id, "user", "a question")
        history = load_history(identity_id)

        # run_inference raises — caller (app.py equivalent) must NOT log assistant
        with pytest.raises(RuntimeError):
            run_inference("model", "", history, "key")

        # Only the user message should be in the log — no assistant entry added
        final_history = load_history(identity_id)
        assert len(final_history) == 1
        assert final_history[0]["role"] == "user"
