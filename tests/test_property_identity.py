"""
Property-based tests for Identity Isolation (Property 1).

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

Property 1: Identity Isolation
  For any two distinct identity IDs A and B, no operation on identity A may
  affect the data returned for identity B.

  - append_message(A, ...) followed by load_history(B) returns the same
    result as load_history(B) alone (REQ 1.1).
  - clear_identity(A) does not alter logs/B.json (REQ 1.2).
  - list_identities() returns a set — each identity appears exactly once
    regardless of how many messages it has (REQ 1.3).
  - load_history returns [] for a non-existent identity (REQ 1.4).
  - load_history returns messages in strict chronological/insertion order (REQ 1.5).
"""

import json
import os
import tempfile

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import falcon.identity as identity_module
import falcon.logger as logger_module
from falcon.identity import clear_identity, list_identities, load_history
from falcon.logger import append_message


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Safe identity IDs: start with a letter, lowercase alphanumeric + underscore, ≤ 20 chars.
# Lowercase-only avoids Windows case-insensitive filesystem collisions (e.g. 'A' vs 'a'
# map to the same file on NTFS).
safe_identity_id = st.from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True)

# Pairs of DISTINCT identity IDs (A ≠ B).
distinct_identity_pair = st.tuples(safe_identity_id, safe_identity_id).filter(
    lambda pair: pair[0] != pair[1]
)

valid_role = st.sampled_from(["user", "assistant"])

# Content: printable text including unicode; empty strings are valid.
content_text = st.text(min_size=0, max_size=200)

# A single (role, content) pair.
message_pair = st.tuples(valid_role, content_text)

# A sequence of message pairs (up to 10 to keep tests fast).
message_sequence = st.lists(message_pair, min_size=1, max_size=10)

# A possibly-empty sequence (for identity B which may start with no messages).
optional_message_sequence = st.lists(message_pair, min_size=0, max_size=10)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _patch_log_dirs(tmp_dir: str):
    """Redirect both logger and identity modules to a shared temp directory."""
    original_logger = logger_module._LOG_DIR
    original_identity = identity_module._LOG_DIR
    logger_module._LOG_DIR = tmp_dir
    identity_module._LOG_DIR = tmp_dir
    return original_logger, original_identity


def _restore_log_dirs(original_logger: str, original_identity: str):
    """Restore logger and identity modules to their original log directories."""
    logger_module._LOG_DIR = original_logger
    identity_module._LOG_DIR = original_identity


# ---------------------------------------------------------------------------
# Property 1a — append_message(A) does not affect load_history(B)
#
# Validates: Requirements 1.1
# ---------------------------------------------------------------------------

@given(
    ids=distinct_identity_pair,
    messages_a=message_sequence,
    messages_b=optional_message_sequence,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_append_to_A_does_not_affect_B(
    ids: tuple[str, str],
    messages_a: list[tuple[str, str]],
    messages_b: list[tuple[str, str]],
):
    """
    Property 1a: append_message(A, ...) does not change load_history(B).

    **Validates: Requirements 1.1**

    For distinct identities A and B:
    1. Pre-populate B with some messages (may be empty).
    2. Snapshot load_history(B) BEFORE writing to A.
    3. Write messages_a to A.
    4. Assert load_history(B) is identical to the snapshot.
    """
    identity_a, identity_b = ids

    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_logger, orig_identity = _patch_log_dirs(tmp_dir)
        try:
            # Pre-populate identity B.
            for role, content in messages_b:
                append_message(identity_b, role, content)

            # Snapshot B before any writes to A.
            snapshot_b = load_history(identity_b)

            # Write to A.
            for role, content in messages_a:
                append_message(identity_a, role, content)

            # B's history must be unchanged.
            history_b_after = load_history(identity_b)

            assert history_b_after == snapshot_b, (
                f"load_history({identity_b!r}) changed after writing to "
                f"{identity_a!r}.\n"
                f"  Before: {snapshot_b}\n"
                f"  After:  {history_b_after}"
            )
        finally:
            _restore_log_dirs(orig_logger, orig_identity)


# ---------------------------------------------------------------------------
# Property 1b — clear_identity(A) does not alter logs/B.json
#
# Validates: Requirements 1.2
# ---------------------------------------------------------------------------

@given(
    ids=distinct_identity_pair,
    messages_a=message_sequence,
    messages_b=message_sequence,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_clear_A_does_not_affect_B(
    ids: tuple[str, str],
    messages_a: list[tuple[str, str]],
    messages_b: list[tuple[str, str]],
):
    """
    Property 1b: clear_identity(A) does not alter logs/B.json.

    **Validates: Requirements 1.2**

    For distinct identities A and B:
    1. Write messages to both A and B.
    2. Read and snapshot the raw bytes of B's log file.
    3. Clear identity A.
    4. Assert B's log file bytes are identical to the snapshot.
    5. Assert A's file no longer exists.
    """
    identity_a, identity_b = ids

    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_logger, orig_identity = _patch_log_dirs(tmp_dir)
        try:
            # Populate both identities.
            for role, content in messages_a:
                append_message(identity_a, role, content)
            for role, content in messages_b:
                append_message(identity_b, role, content)

            # Snapshot raw bytes of B's log file.
            path_b = os.path.join(tmp_dir, f"{identity_b}.json")
            assert os.path.exists(path_b), (
                f"Expected log file for {identity_b!r} to exist before clear"
            )
            with open(path_b, "rb") as fh:
                original_bytes_b = fh.read()

            # Clear A.
            clear_identity(identity_a)

            # A's file should be gone (or never recreated).
            path_a = os.path.join(tmp_dir, f"{identity_a}.json")
            assert not os.path.exists(path_a), (
                f"Expected {identity_a!r} log file to be deleted after clear_identity"
            )

            # B's file must be byte-for-byte identical.
            assert os.path.exists(path_b), (
                f"B's log file was deleted by clear_identity({identity_a!r})"
            )
            with open(path_b, "rb") as fh:
                current_bytes_b = fh.read()

            assert current_bytes_b == original_bytes_b, (
                f"logs/{identity_b}.json was modified by clear_identity({identity_a!r}).\n"
                f"  Original: {original_bytes_b!r}\n"
                f"  Current:  {current_bytes_b!r}"
            )
        finally:
            _restore_log_dirs(orig_logger, orig_identity)


# ---------------------------------------------------------------------------
# Property 1c — list_identities() contains each ID exactly once
#
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------

@given(
    ids=distinct_identity_pair,
    messages_a=message_sequence,
    messages_b=message_sequence,
    extra_appends=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_list_identities_no_duplicates(
    ids: tuple[str, str],
    messages_a: list[tuple[str, str]],
    messages_b: list[tuple[str, str]],
    extra_appends: int,
):
    """
    Property 1c: list_identities() contains each identity ID exactly once.

    **Validates: Requirements 1.3**

    Even if many messages are written to the same identity, it appears
    exactly once in list_identities().
    """
    identity_a, identity_b = ids

    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_logger, orig_identity = _patch_log_dirs(tmp_dir)
        try:
            # Write to A multiple times (including extra appends to test dedup).
            for role, content in messages_a:
                append_message(identity_a, role, content)
            for i in range(extra_appends):
                append_message(identity_a, "user", f"extra message {i}")

            # Write to B.
            for role, content in messages_b:
                append_message(identity_b, role, content)

            identities = list_identities()

            # A and B must both appear.
            assert identity_a in identities, (
                f"{identity_a!r} not found in list_identities(): {identities}"
            )
            assert identity_b in identities, (
                f"{identity_b!r} not found in list_identities(): {identities}"
            )

            # No duplicates — each ID appears exactly once.
            assert identities.count(identity_a) == 1, (
                f"{identity_a!r} appears {identities.count(identity_a)} times "
                f"in list_identities(), expected exactly 1"
            )
            assert identities.count(identity_b) == 1, (
                f"{identity_b!r} appears {identities.count(identity_b)} times "
                f"in list_identities(), expected exactly 1"
            )
        finally:
            _restore_log_dirs(orig_logger, orig_identity)


# ---------------------------------------------------------------------------
# Property 1d — load_history returns [] for a non-existent identity
#
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------

@given(identity_id=safe_identity_id)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_load_history_empty_for_nonexistent_identity(identity_id: str):
    """
    Property 1d: load_history returns [] when no log file exists.

    **Validates: Requirements 1.4**

    For any valid identity ID that has never been written to,
    load_history must return an empty list.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_logger, orig_identity = _patch_log_dirs(tmp_dir)
        try:
            result = load_history(identity_id)
            assert result == [], (
                f"load_history({identity_id!r}) returned {result!r} for a "
                f"non-existent identity; expected []"
            )
        finally:
            _restore_log_dirs(orig_logger, orig_identity)


# ---------------------------------------------------------------------------
# Property 1e — load_history returns messages in strict insertion order
#
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

@given(identity_id=safe_identity_id, messages=message_sequence)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_load_history_insertion_order(
    identity_id: str,
    messages: list[tuple[str, str]],
):
    """
    Property 1e: load_history returns messages in strict insertion order.

    **Validates: Requirements 1.5**

    After appending N messages in order, load_history must return exactly
    those N messages in the same sequence — no reordering, no gaps.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_logger, orig_identity = _patch_log_dirs(tmp_dir)
        try:
            for role, content in messages:
                append_message(identity_id, role, content)

            history = load_history(identity_id)

            assert len(history) == len(messages), (
                f"load_history returned {len(history)} entries; "
                f"expected {len(messages)}"
            )

            for idx, (exp_role, exp_content) in enumerate(messages):
                assert history[idx]["role"] == exp_role, (
                    f"Entry {idx}: expected role {exp_role!r}, "
                    f"got {history[idx]['role']!r}"
                )
                assert history[idx]["content"] == exp_content, (
                    f"Entry {idx}: content mismatch — "
                    f"expected {exp_content!r}, got {history[idx]['content']!r}"
                )
        finally:
            _restore_log_dirs(orig_logger, orig_identity)
