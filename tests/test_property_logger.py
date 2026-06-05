"""
Property-based tests for Log Integrity (Property 3).

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

Property 3: Log Integrity
  - After every call to append_message(identity_id, ...), logs/{identity_id}.json
    parses as valid JSON.
  - The number of entries in the log only ever increases within a session.
  - load_history(identity_id) after N calls to append_message returns exactly N
    entries in insertion order.
  - No existing entry's timestamp, role, or content is mutated by a subsequent
    append_message call.
"""

import copy
import json
import os
import tempfile
from datetime import datetime

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import falcon.logger as logger_module
import falcon.identity as identity_module
from falcon.logger import append_message
from falcon.identity import load_history


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Safe identity IDs: alphanumeric + underscore, non-empty, no forbidden chars
safe_identity_id = st.from_regex(r"[a-zA-Z][a-zA-Z0-9_]{0,19}", fullmatch=True)

valid_role = st.sampled_from(["user", "assistant"])

# Content: printable text including unicode, empty strings are fine
content_text = st.text(min_size=0, max_size=200)

# A single (role, content) pair
message_pair = st.tuples(valid_role, content_text)

# A non-empty sequence of message pairs (up to 20 to keep tests fast)
message_sequence = st.lists(message_pair, min_size=1, max_size=20)


# ---------------------------------------------------------------------------
# Property Test
# ---------------------------------------------------------------------------

@given(identity_id=safe_identity_id, messages=message_sequence)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_log_integrity(identity_id: str, messages: list[tuple[str, str]]):
    """
    Property 3: Log Integrity

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

    For any sequence of (role, content) pairs appended to the same identity:
    1. The log file parses as valid JSON after each append (REQ 3.1).
    2. Prior entries' timestamp, role, content are not mutated (REQ 3.2).
    3. After N appends, load_history returns exactly N entries in insertion
       order (REQ 3.3).
    4. Each entry has exactly 3 fields: timestamp, role, content (REQ 3.4).
    5. Timestamp is an ISO 8601 UTC string ending with 'Z' (REQ 3.5).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Redirect both logger and identity modules to the same isolated temp dir
        original_logger_log_dir = logger_module._LOG_DIR
        original_identity_log_dir = identity_module._LOG_DIR
        logger_module._LOG_DIR = tmp_dir
        identity_module._LOG_DIR = tmp_dir

        try:
            log_file = os.path.join(tmp_dir, f"{identity_id}.json")
            snapshots: list[list[dict]] = []  # deep copy after each append

            for i, (role, content) in enumerate(messages):
                append_message(identity_id, role, content)

                # --- REQ 3.1: file parses as valid JSON after every write ---
                assert os.path.exists(log_file), (
                    f"Log file does not exist after append #{i + 1}"
                )
                with open(log_file, "r", encoding="utf-8") as fh:
                    raw = fh.read()
                try:
                    current_entries = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise AssertionError(
                        f"Log file is not valid JSON after append #{i + 1}: {exc}"
                    ) from exc

                # --- REQ 3.2: prior entries are not mutated ---
                for snap_idx, snapshot in enumerate(snapshots):
                    for entry_idx, prior_entry in enumerate(snapshot):
                        current_entry = current_entries[entry_idx]
                        assert current_entry["timestamp"] == prior_entry["timestamp"], (
                            f"Entry {entry_idx} timestamp mutated after append "
                            f"#{i + 1} (was {prior_entry['timestamp']!r}, "
                            f"now {current_entry['timestamp']!r})"
                        )
                        assert current_entry["role"] == prior_entry["role"], (
                            f"Entry {entry_idx} role mutated after append #{i + 1}"
                        )
                        assert current_entry["content"] == prior_entry["content"], (
                            f"Entry {entry_idx} content mutated after append #{i + 1}"
                        )

                # Save a deep copy for future mutation checks
                snapshots.append(copy.deepcopy(current_entries))

                # --- REQ 3.3: entry count equals number of appends so far ---
                assert len(current_entries) == i + 1, (
                    f"Expected {i + 1} entries after {i + 1} appends, "
                    f"got {len(current_entries)}"
                )

                # --- REQ 3.3 (order): entries appear in insertion order ---
                for j, (exp_role, exp_content) in enumerate(messages[: i + 1]):
                    assert current_entries[j]["role"] == exp_role, (
                        f"Entry {j} role mismatch: expected {exp_role!r}, "
                        f"got {current_entries[j]['role']!r}"
                    )
                    assert current_entries[j]["content"] == exp_content, (
                        f"Entry {j} content mismatch"
                    )

                # --- REQ 3.4: each entry has exactly 3 fields ---
                for entry in current_entries:
                    assert set(entry.keys()) == {"timestamp", "role", "content"}, (
                        f"Entry has unexpected fields: {set(entry.keys())}"
                    )

                # --- REQ 3.5: timestamp is ISO 8601 UTC string ending with 'Z' ---
                for entry in current_entries:
                    ts = entry["timestamp"]
                    assert isinstance(ts, str), f"timestamp is not a string: {ts!r}"
                    assert ts.endswith("Z"), (
                        f"timestamp does not end with 'Z': {ts!r}"
                    )
                    # Validate ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
                    try:
                        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                    except ValueError as exc:
                        raise AssertionError(
                            f"timestamp is not valid ISO 8601 UTC: {ts!r}"
                        ) from exc

            # --- Final REQ 3.3 check: load_history returns exactly N entries ---
            # Uses falcon.identity.load_history with the same patched _LOG_DIR
            final_history = load_history(identity_id)
            assert len(final_history) == len(messages), (
                f"load_history returned {len(final_history)} entries, "
                f"expected {len(messages)}"
            )

            # Verify insertion order matches appended sequence
            for idx, (exp_role, exp_content) in enumerate(messages):
                assert final_history[idx]["role"] == exp_role, (
                    f"load_history entry {idx} role mismatch: "
                    f"expected {exp_role!r}, got {final_history[idx]['role']!r}"
                )
                assert final_history[idx]["content"] == exp_content, (
                    f"load_history entry {idx} content mismatch"
                )

        finally:
            # Restore original log directories so other tests are not affected
            logger_module._LOG_DIR = original_logger_log_dir
            identity_module._LOG_DIR = original_identity_log_dir
