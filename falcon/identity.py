"""
Identity Manager module for Falcon V1.

Provides three operations over per-identity log files:
  - list_identities()  — enumerate all identities that have log files
  - load_history()     — load the message history for a given identity
  - clear_identity()   — delete the log file for a given identity

Log files live at logs/{identity_id}.json, using the same _LOG_DIR constant
as logger.py.  The two modules must agree on this path at all times.
"""

import glob
import json
import os

# Must match logger._LOG_DIR — both modules read from / write to this directory.
_LOG_DIR = "logs"

# Forbidden characters / sequences in identity_id values.
_FORBIDDEN_CHARS = ("/", "\\")
_FORBIDDEN_SEQUENCES = ("..",)
_FORBIDDEN_BYTES = ("\x00",)


def _validate_identity_id(identity_id: str) -> None:
    """Raise ValueError if identity_id contains path-traversal characters.

        - forward slash  /
        - backslash      \\
        - double-dot     ..
        - null byte      \\x00
    """
    bad: list[str] = []

    if "/" in identity_id:
        bad.append("/")
    if "\\" in identity_id:
        bad.append("\\")
    if ".." in identity_id:
        bad.append("..")
    if "\x00" in identity_id:
        bad.append("null byte")

    if bad:
        raise ValueError(
            f"identity_id contains disallowed character(s): {', '.join(bad)}"
        )


def list_identities() -> list[str]:
    """Return all identity IDs that currently have a log file in logs/.

    Each identity ID is derived from its filename:
        logs/alice.json  →  "alice"

    Returns an empty list if the logs/ directory does not exist or is empty.

     — each ID appears exactly once.
     — only IDs whose log file currently exists are included.
    """
    pattern = os.path.join(_LOG_DIR, "*.json")
    paths = glob.glob(pattern)
    identities: list[str] = []
    for path in paths:
        basename = os.path.basename(path)          # e.g. "alice.json"
        identity_id, _ = os.path.splitext(basename)  # e.g. "alice"
        identities.append(identity_id)
    return identities


def load_history(identity_id: str) -> list[dict]:
    """Return the message history for identity_id in chronological order.

    Behaviour:
    - Validates identity_id for path-traversal characters.
    - Returns an empty list [] if no log file exists for this identity.
    - Raises json.JSONDecodeError if the log file exists but is not valid JSON,
      without modifying the file.
    - Returns entries in the order they were appended — chronological order is
      preserved because logger.py always appends to the end.
    - Returns a shallow copy of the list so the caller cannot mutate the
      internal state (defensive copy per design spec).

    Args:
        identity_id: The identity whose history to load.

    Returns:
        A list of {timestamp, role, content} dicts in chronological order.

    Raises:
        ValueError: If identity_id contains forbidden characters.
        json.JSONDecodeError: If the log file exists but is not valid JSON.
    """
    _validate_identity_id(identity_id)

    log_path = os.path.join(_LOG_DIR, f"{identity_id}.json")

    if not os.path.exists(log_path):
        return []

    with open(log_path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    # Raises json.JSONDecodeError if the file is not valid JSON — we do not
    # catch this; the caller (app.py) is responsible for surfacing the error.
    entries: list[dict] = json.loads(raw)

    # Return a copy so the caller cannot inadvertently mutate logged history.
    return list(entries)


def clear_identity(identity_id: str) -> None:
    """Delete the log file for identity_id.

    Behaviour:
    - Validates identity_id for path-traversal characters.
    - Deletes logs/{identity_id}.json if it exists.
    - No-op if the file does not exist — does not raise.
    - Does NOT affect any other identity's log file.

    Args:
        identity_id: The identity whose log file should be deleted.

    Raises:
        ValueError: If identity_id contains forbidden characters.
    """
    _validate_identity_id(identity_id)

    log_path = os.path.join(_LOG_DIR, f"{identity_id}.json")

    if os.path.exists(log_path):
        os.remove(log_path)
