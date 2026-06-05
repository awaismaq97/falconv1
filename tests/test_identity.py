"""
Unit tests for falcon/identity.py.

Covers:
- load_history returns [] for non-existent identity (REQ 1.4)
- load_history raises json.JSONDecodeError for corrupted file without modifying it (REQ 1.6)
- load_history returns entries in chronological order (REQ 1.5)
- load_history returns a copy, not a reference (design spec)
- clear_identity removes only the target identity file (REQ 1.2)
- clear_identity is a no-op for a non-existent identity (REQ 1.2)
- list_identities reflects only existing log files (REQ 1.3, 1.8)
- Path-traversal identity_id raises ValueError (REQ 8.1, 8.2)

Requirements: 1.1–1.7, 8.1, 8.2
"""

import json
import os

import pytest

import falcon.identity as identity_module
from falcon.identity import clear_identity, list_identities, load_history


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def isolated_log_dir(tmp_path, monkeypatch):
    """Monkeypatch falcon.identity._LOG_DIR to a fresh temp directory."""
    monkeypatch.setattr(identity_module, "_LOG_DIR", str(tmp_path))
    return tmp_path


def _write_log(log_dir, identity_id: str, entries: list) -> None:
    """Write a JSON log file directly for test setup."""
    log_file = os.path.join(str(log_dir), f"{identity_id}.json")
    with open(log_file, "w", encoding="utf-8") as fh:
        json.dump(entries, fh)


def _make_entry(role: str, content: str, timestamp: str = "2025-06-05T14:22:01Z") -> dict:
    """Return a minimal valid log entry dict."""
    return {"timestamp": timestamp, "role": role, "content": content}


# ---------------------------------------------------------------------------
# load_history: non-existent identity returns [] (REQ 1.4)
# ---------------------------------------------------------------------------

class TestLoadHistoryMissingFile:
    """REQ 1.4: load_history returns [] when no log file exists."""

    def test_returns_empty_list_for_unknown_identity(self, isolated_log_dir):
        """Non-existent identity → empty list, no exception."""
        result = load_history("nonexistent")
        assert result == []

    def test_returns_empty_list_is_a_list(self, isolated_log_dir):
        """The return value must be a list instance, not None or any other type."""
        result = load_history("ghost")
        assert isinstance(result, list)

    def test_returns_empty_list_when_log_dir_absent(self, tmp_path, monkeypatch):
        """If the entire logs/ directory is absent, load_history still returns []."""
        missing_dir = tmp_path / "no_such_dir"
        monkeypatch.setattr(identity_module, "_LOG_DIR", str(missing_dir))
        result = load_history("any_id")
        assert result == []


# ---------------------------------------------------------------------------
# load_history: corrupted file raises json.JSONDecodeError, file unchanged (REQ 1.6)
# ---------------------------------------------------------------------------

class TestLoadHistoryCorruptedFile:
    """REQ 1.6: corrupted log raises json.JSONDecodeError without modifying the file."""

    def test_corrupted_file_raises_json_decode_error(self, isolated_log_dir):
        """Plain garbage text → json.JSONDecodeError."""
        log_file = isolated_log_dir / "corrupt.json"
        log_file.write_text("this is not json {{{{", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_history("corrupt")

    def test_corrupted_file_is_not_modified(self, isolated_log_dir):
        """After a failed load, the corrupted file must remain exactly as it was."""
        log_file = isolated_log_dir / "corrupt2.json"
        original_content = "not json at all"
        log_file.write_text(original_content, encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_history("corrupt2")

        assert log_file.read_text(encoding="utf-8") == original_content

    def test_partially_valid_json_raises_json_decode_error(self, isolated_log_dir):
        """A truncated JSON array raises json.JSONDecodeError."""
        log_file = isolated_log_dir / "partial.json"
        log_file.write_text('[{"timestamp": "2025-01-01T00:00:00Z", "role": "user"',
                             encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_history("partial")

    def test_partially_valid_json_file_is_not_modified(self, isolated_log_dir):
        """A truncated JSON file must remain untouched after a failed load."""
        log_file = isolated_log_dir / "partial2.json"
        partial_content = '[{"role": "user", "content": "hi"'
        log_file.write_text(partial_content, encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_history("partial2")

        assert log_file.read_text(encoding="utf-8") == partial_content


# ---------------------------------------------------------------------------
# load_history: returns entries in chronological order (REQ 1.5)
# ---------------------------------------------------------------------------

class TestLoadHistoryOrder:
    """REQ 1.5: load_history returns entries in chronological (insertion) order."""

    def test_single_entry_returned_correctly(self, isolated_log_dir):
        """A single-entry log is returned as a one-element list."""
        entry = _make_entry("user", "hello")
        _write_log(isolated_log_dir, "single", [entry])
        result = load_history("single")
        assert result == [entry]

    def test_multiple_entries_returned_in_order(self, isolated_log_dir):
        """Multiple entries are returned in the order they were written."""
        entries = [
            _make_entry("user", "first", "2025-06-05T10:00:00Z"),
            _make_entry("assistant", "second", "2025-06-05T10:00:01Z"),
            _make_entry("user", "third", "2025-06-05T10:00:02Z"),
        ]
        _write_log(isolated_log_dir, "ordered", entries)
        result = load_history("ordered")
        assert result == entries

    def test_returns_correct_field_values(self, isolated_log_dir):
        """Each entry's timestamp, role, and content are preserved exactly."""
        entries = [
            _make_entry("user", "exact content here", "2025-01-15T08:30:00Z"),
            _make_entry("assistant", "response text", "2025-01-15T08:30:02Z"),
        ]
        _write_log(isolated_log_dir, "fields", entries)
        result = load_history("fields")
        for expected, actual in zip(entries, result):
            assert actual["timestamp"] == expected["timestamp"]
            assert actual["role"] == expected["role"]
            assert actual["content"] == expected["content"]


# ---------------------------------------------------------------------------
# load_history: returns a copy, not a reference (design spec)
# ---------------------------------------------------------------------------

class TestLoadHistoryReturnsCopy:
    """load_history must return a copy so callers cannot mutate the loaded data."""

    def test_mutating_result_does_not_affect_subsequent_load(self, isolated_log_dir):
        """Modifying the returned list must not affect a subsequent load call."""
        entry = _make_entry("user", "original")
        _write_log(isolated_log_dir, "copy_test", [entry])

        result1 = load_history("copy_test")
        result1.append({"timestamp": "fake", "role": "user", "content": "injected"})

        result2 = load_history("copy_test")
        assert len(result2) == 1
        assert result2[0]["content"] == "original"

    def test_returned_list_is_not_same_object_on_repeated_calls(self, isolated_log_dir):
        """Two successive calls return distinct list objects."""
        _write_log(isolated_log_dir, "id_test", [_make_entry("user", "msg")])
        r1 = load_history("id_test")
        r2 = load_history("id_test")
        assert r1 is not r2


# ---------------------------------------------------------------------------
# clear_identity: removes only the target identity file (REQ 1.2)
# ---------------------------------------------------------------------------

class TestClearIdentity:
    """REQ 1.2: clear_identity deletes the target file and only the target file."""

    def test_clear_removes_target_file(self, isolated_log_dir):
        """clear_identity deletes the named identity's log file."""
        _write_log(isolated_log_dir, "to_clear", [_make_entry("user", "bye")])
        log_file = isolated_log_dir / "to_clear.json"
        assert log_file.exists()

        clear_identity("to_clear")

        assert not log_file.exists()

    def test_clear_does_not_remove_other_identity_file(self, isolated_log_dir):
        """Clearing identity A must not affect identity B's file. REQ 1.1, 1.2."""
        _write_log(isolated_log_dir, "alpha", [_make_entry("user", "alpha message")])
        _write_log(isolated_log_dir, "beta", [_make_entry("user", "beta message")])

        alpha_file = isolated_log_dir / "alpha.json"
        beta_file = isolated_log_dir / "beta.json"

        clear_identity("alpha")

        assert not alpha_file.exists(), "alpha should be cleared"
        assert beta_file.exists(), "beta must remain untouched"

    def test_clear_leaves_other_identity_content_unchanged(self, isolated_log_dir):
        """After clearing A, B's content is byte-for-byte unchanged. REQ 1.1."""
        entries_b = [
            _make_entry("user", "beta first"),
            _make_entry("assistant", "beta reply"),
        ]
        _write_log(isolated_log_dir, "a_id", [_make_entry("user", "a's msg")])
        _write_log(isolated_log_dir, "b_id", entries_b)

        beta_file = isolated_log_dir / "b_id.json"
        original_content = beta_file.read_text(encoding="utf-8")

        clear_identity("a_id")

        assert beta_file.read_text(encoding="utf-8") == original_content

    def test_clear_noop_for_nonexistent_identity(self, isolated_log_dir):
        """clear_identity on a missing identity must not raise. REQ 1.2 (no-op)."""
        # Should not raise any exception
        clear_identity("does_not_exist")

    def test_clear_noop_when_log_dir_absent(self, tmp_path, monkeypatch):
        """clear_identity is a no-op when the entire logs/ dir is absent."""
        missing_dir = tmp_path / "no_dir"
        monkeypatch.setattr(identity_module, "_LOG_DIR", str(missing_dir))
        # Should not raise
        clear_identity("any_identity")

    def test_after_clear_load_history_returns_empty_list(self, isolated_log_dir):
        """After clearing, load_history for that identity returns []."""
        _write_log(isolated_log_dir, "cleared", [_make_entry("user", "data")])
        clear_identity("cleared")
        assert load_history("cleared") == []


# ---------------------------------------------------------------------------
# list_identities: reflects only existing log files (REQ 1.3, 1.8)
# ---------------------------------------------------------------------------

class TestListIdentities:
    """REQ 1.3, 1.8: list_identities returns exactly the IDs with existing log files."""

    def test_empty_logs_dir_returns_empty_list(self, isolated_log_dir):
        """An empty logs/ directory → empty list."""
        result = list_identities()
        assert result == []

    def test_absent_logs_dir_returns_empty_list(self, tmp_path, monkeypatch):
        """If the logs/ directory does not exist, list_identities returns []."""
        missing_dir = tmp_path / "no_logs"
        monkeypatch.setattr(identity_module, "_LOG_DIR", str(missing_dir))
        result = list_identities()
        assert result == []

    def test_single_identity_appears(self, isolated_log_dir):
        """One log file → that identity ID appears in the list."""
        _write_log(isolated_log_dir, "alice", [_make_entry("user", "hi")])
        result = list_identities()
        assert "alice" in result

    def test_multiple_identities_all_appear(self, isolated_log_dir):
        """Multiple log files → all identity IDs appear."""
        for name in ("alice", "bob", "carol"):
            _write_log(isolated_log_dir, name, [_make_entry("user", "msg")])
        result = list_identities()
        assert set(result) == {"alice", "bob", "carol"}

    def test_each_identity_appears_exactly_once(self, isolated_log_dir):
        """REQ 1.3: each ID appears exactly once regardless of message count."""
        _write_log(isolated_log_dir, "multi", [
            _make_entry("user", "first"),
            _make_entry("assistant", "second"),
            _make_entry("user", "third"),
        ])
        result = list_identities()
        assert result.count("multi") == 1

    def test_cleared_identity_no_longer_listed(self, isolated_log_dir):
        """After clear_identity, that ID must not appear in list_identities. REQ 1.8."""
        _write_log(isolated_log_dir, "temp", [_make_entry("user", "msg")])
        assert "temp" in list_identities()

        clear_identity("temp")

        assert "temp" not in list_identities()

    def test_non_json_files_not_included(self, isolated_log_dir):
        """Files without .json extension must not be listed as identity IDs."""
        # Write a non-JSON file alongside a valid log
        (isolated_log_dir / "readme.txt").write_text("not a log", encoding="utf-8")
        _write_log(isolated_log_dir, "valid", [_make_entry("user", "msg")])

        result = list_identities()
        assert "valid" in result
        assert "readme" not in result

    def test_list_contains_only_ids_from_json_filenames(self, isolated_log_dir):
        """Filename stem is used as identity ID — extension stripped correctly."""
        _write_log(isolated_log_dir, "user_42", [_make_entry("user", "hello")])
        result = list_identities()
        assert "user_42" in result
        assert "user_42.json" not in result


# ---------------------------------------------------------------------------
# Path-traversal identity_id validation (REQ 8.1, 8.2)
# ---------------------------------------------------------------------------

class TestPathTraversalValidation:
    """REQ 8.1, 8.2: forbidden characters in identity_id raise ValueError."""

    @pytest.mark.parametrize("bad_id,expected_fragment", [
        ("../etc/passwd", "/"),
        ("..\\secrets", "\\"),
        ("foo/bar", "/"),
        ("foo\\bar", "\\"),
        ("foo\x00bar", "null byte"),
        ("..", ".."),
    ])
    def test_load_history_rejects_bad_identity_id(self, isolated_log_dir, bad_id, expected_fragment):
        """load_history raises ValueError for identity_id with forbidden chars."""
        with pytest.raises(ValueError) as exc_info:
            load_history(bad_id)
        assert expected_fragment in str(exc_info.value), (
            f"Expected error message to mention {expected_fragment!r}, got: {exc_info.value}"
        )

    @pytest.mark.parametrize("bad_id,expected_fragment", [
        ("../etc/passwd", "/"),
        ("..\\secrets", "\\"),
        ("foo/bar", "/"),
        ("foo\\bar", "\\"),
        ("foo\x00bar", "null byte"),
        ("..", ".."),
    ])
    def test_clear_identity_rejects_bad_identity_id(self, isolated_log_dir, bad_id, expected_fragment):
        """clear_identity raises ValueError for identity_id with forbidden chars."""
        with pytest.raises(ValueError) as exc_info:
            clear_identity(bad_id)
        assert expected_fragment in str(exc_info.value), (
            f"Expected error message to mention {expected_fragment!r}, got: {exc_info.value}"
        )

    def test_load_history_raises_before_any_io(self, isolated_log_dir):
        """ValueError from load_history must be raised before touching the filesystem."""
        # A traversal ID that would resolve outside logs/ if allowed through
        bad_id = "../outside"
        log_file = isolated_log_dir / "outside.json"
        log_file.write_text('[{"timestamp":"2025-01-01T00:00:00Z","role":"user","content":"x"}]',
                             encoding="utf-8")

        with pytest.raises(ValueError):
            load_history(bad_id)

    def test_clear_identity_raises_before_any_io(self, isolated_log_dir):
        """ValueError from clear_identity must be raised before touching the filesystem."""
        bad_id = "../outside"
        # Even if a matching file somehow existed, it must not be deleted
        with pytest.raises(ValueError):
            clear_identity(bad_id)

    def test_valid_identity_ids_do_not_raise(self, isolated_log_dir):
        """Standard alphanumeric and underscore identity IDs should not raise."""
        for valid_id in ("default", "user_1", "alice", "session-42"):
            # Should not raise for any of these
            result = load_history(valid_id)
            assert result == []


# ---------------------------------------------------------------------------
# REQ 1.1: Cross-identity isolation — append + load other identity
# ---------------------------------------------------------------------------

class TestCrossIdentityIsolation:
    """REQ 1.1: operations on identity A must not affect identity B."""

    def test_load_history_b_unaffected_after_writing_to_a(self, isolated_log_dir):
        """
        Simulate: append to A (by writing A's file directly), then load B.
        B should be empty since we never wrote to it.
        """
        _write_log(isolated_log_dir, "identity_a", [_make_entry("user", "a message")])
        result_b = load_history("identity_b")
        assert result_b == []

    def test_load_history_b_content_unchanged_after_writing_a(self, isolated_log_dir):
        """B's history is unchanged when A gets new messages."""
        entries_b = [_make_entry("user", "b's only message")]
        _write_log(isolated_log_dir, "id_b", entries_b)

        # Write to A (simulate append)
        _write_log(isolated_log_dir, "id_a", [
            _make_entry("user", "a first"),
            _make_entry("assistant", "a second"),
        ])

        result_b = load_history("id_b")
        assert result_b == entries_b
