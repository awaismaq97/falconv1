"""
Unit tests for falcon/logger.py.

Covers:
- Two-message round-trip (REQ 3.1, 3.2, 3.3, 3.4, 3.5)
- role validation raises ValueError and does not write (REQ 3.6)
- Corrupted log raises json.JSONDecodeError and leaves file unchanged (REQ 3.8)
- logs/ directory auto-creation (REQ 3.7)
- Path-traversal identity_id raises ValueError (REQ 8.1, 8.2)
"""

import json
import re

import pytest

import falcon.logger as logger_module
from falcon.logger import append_message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ISO_8601_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _log_path(log_dir, identity_id: str):
    """Return the expected path for a given identity's log file."""
    import os
    return os.path.join(str(log_dir), f"{identity_id}.json")


# ---------------------------------------------------------------------------
# Fixture: redirect _LOG_DIR to a temporary directory
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def isolated_log_dir(tmp_path, monkeypatch):
    """Monkeypatch falcon.logger._LOG_DIR to a fresh temp directory."""
    monkeypatch.setattr(logger_module, "_LOG_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Test: logs/ directory auto-creation (REQ 3.7)
# ---------------------------------------------------------------------------

def test_logs_directory_autocreated(tmp_path, monkeypatch):
    """
    When the logs/ directory does not yet exist, append_message must create it.
    REQ 3.7: auto-create logs/ directory if absent.
    """
    import os
    new_dir = tmp_path / "new_logs_dir"
    assert not new_dir.exists(), "Pre-condition: directory must not exist before the call"

    monkeypatch.setattr(logger_module, "_LOG_DIR", str(new_dir))

    append_message("alice", "user", "hello")

    assert new_dir.exists() and new_dir.is_dir()
    log_file = new_dir / "alice.json"
    assert log_file.exists()


# ---------------------------------------------------------------------------
# Test: two-message round-trip (REQ 3.1, 3.2, 3.3, 3.4, 3.5)
# ---------------------------------------------------------------------------

def test_two_message_round_trip(isolated_log_dir):
    """
    Append two messages and verify count, field presence, field values, and
    that the file is valid JSON after each write.
    REQ 3.1–3.5.
    """
    log_file = isolated_log_dir / "conv.json"

    append_message("conv", "user", "first message")
    # File must be valid JSON after first write
    entries = json.loads(log_file.read_text(encoding="utf-8"))
    assert len(entries) == 1

    append_message("conv", "assistant", "second message")
    entries = json.loads(log_file.read_text(encoding="utf-8"))

    # REQ 3.3: exactly N entries for N appends
    assert len(entries) == 2

    # REQ 3.4: each entry has exactly three fields
    for entry in entries:
        assert set(entry.keys()) == {"timestamp", "role", "content"}

    # REQ 3.5: timestamp is ISO 8601 UTC
    for entry in entries:
        assert ISO_8601_UTC.match(entry["timestamp"]), (
            f"timestamp {entry['timestamp']!r} does not match ISO 8601 UTC pattern"
        )

    # Field values are preserved exactly
    assert entries[0]["role"] == "user"
    assert entries[0]["content"] == "first message"
    assert entries[1]["role"] == "assistant"
    assert entries[1]["content"] == "second message"


def test_prior_entries_not_mutated(isolated_log_dir):
    """
    After a second append, the first entry's timestamp, role, and content must
    be unchanged.
    REQ 3.2: prior entries are not mutated.
    """
    log_file = isolated_log_dir / "immut.json"

    append_message("immut", "user", "original")
    entries_after_first = json.loads(log_file.read_text(encoding="utf-8"))
    first_snapshot = dict(entries_after_first[0])

    append_message("immut", "assistant", "reply")
    entries_after_second = json.loads(log_file.read_text(encoding="utf-8"))

    # REQ 3.2
    assert entries_after_second[0] == first_snapshot


# ---------------------------------------------------------------------------
# Test: role validation (REQ 3.6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_role", ["system", "User", "ASSISTANT", "", "admin", "human"])
def test_invalid_role_raises_value_error(isolated_log_dir, bad_role):
    """
    An invalid role must raise ValueError before any I/O.
    REQ 3.6: invalid role → ValueError, no write.
    """
    log_file = isolated_log_dir / "role_test.json"
    assert not log_file.exists()

    with pytest.raises(ValueError):
        append_message("role_test", bad_role, "some content")

    # No file should have been created
    assert not log_file.exists()


def test_invalid_role_does_not_overwrite_existing_log(isolated_log_dir):
    """
    If the log file already exists, an invalid role must not modify it.
    REQ 3.6: no write on ValueError.
    """
    append_message("existing", "user", "prior entry")
    log_file = isolated_log_dir / "existing.json"
    original_content = log_file.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        append_message("existing", "invalid_role", "should not appear")

    assert log_file.read_text(encoding="utf-8") == original_content


# ---------------------------------------------------------------------------
# Test: corrupted log raises json.JSONDecodeError and leaves file unchanged (REQ 3.8)
# ---------------------------------------------------------------------------

def test_corrupted_log_raises_json_decode_error(isolated_log_dir):
    """
    If the log file exists but contains invalid JSON, append_message must raise
    json.JSONDecodeError without overwriting the file.
    REQ 3.8.
    """
    log_file = isolated_log_dir / "bad.json"
    corrupted_content = "this is not json at all {{{"
    log_file.write_text(corrupted_content, encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        append_message("bad", "user", "new entry")

    # File must be unchanged
    assert log_file.read_text(encoding="utf-8") == corrupted_content


def test_partially_corrupted_log_leaves_file_unchanged(isolated_log_dir):
    """
    A file that starts like JSON but is truncated must also leave the file alone.
    REQ 3.8.
    """
    log_file = isolated_log_dir / "partial.json"
    partial_content = '[{"timestamp": "2025-01-01T00:00:00Z", "role": "user"'
    log_file.write_text(partial_content, encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        append_message("partial", "assistant", "follow-up")

    assert log_file.read_text(encoding="utf-8") == partial_content


# ---------------------------------------------------------------------------
# Test: path-traversal identity_id validation (REQ 8.1, 8.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_id,expected_fragment", [
    ("../etc/passwd", "/"),
    ("..\\secrets", "\\"),
    ("foo/bar", "/"),
    ("foo\\bar", "\\"),
    ("foo\x00bar", "null byte"),
    ("..", ".."),
])
def test_path_traversal_identity_id_raises_value_error(isolated_log_dir, bad_id, expected_fragment):
    """
    identity_id values containing /, \\, .., or null bytes must raise ValueError
    before any file path is constructed.
    REQ 8.1, 8.2.
    """
    with pytest.raises(ValueError) as exc_info:
        append_message(bad_id, "user", "content")

    assert expected_fragment in str(exc_info.value), (
        f"Expected error message to mention {expected_fragment!r}, got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Test: valid roles are accepted (sanity / REQ 3.6 positive case)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valid_role", ["user", "assistant"])
def test_valid_roles_accepted(isolated_log_dir, valid_role):
    """Both 'user' and 'assistant' are valid roles and must not raise."""
    append_message("valid_role_test", valid_role, "hello")
    log_file = isolated_log_dir / "valid_role_test.json"
    entries = json.loads(log_file.read_text(encoding="utf-8"))
    assert len(entries) >= 1
    assert entries[-1]["role"] == valid_role
