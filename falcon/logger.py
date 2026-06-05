"""
Logger module for Falcon V1.

Provides append_message() — the sole write path for conversation logs.
Log files are stored as JSON arrays at logs/{identity_id}.json (relative
to the current working directory, or wherever log_dir from config points).
"""

import json
import os
from datetime import datetime, timezone

# Directory used for log files.  Change this if you integrate with config.py.
_LOG_DIR = "logs"

# Characters that are forbidden in identity_id values.
_FORBIDDEN_CHARS = {"/", "\\"}
_FORBIDDEN_SEQUENCES = {".."}
_FORBIDDEN_BYTES = {"\x00"}


def _validate_identity_id(identity_id: str) -> None:
    """Raise ValueError if identity_id contains path-traversal characters.

    Checked characters / sequences per REQ 8.1 and REQ 8.2:
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


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string ending with 'Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_message(identity_id: str, role: str, content: str) -> None:
    """Append one message entry to logs/{identity_id}.json.

    Behaviour:
    - Validates identity_id for path-traversal characters (REQ 8.1, 8.2).
    - Validates role is "user" or "assistant"; raises ValueError otherwise (REQ 3.6).
    - Auto-creates the logs/ directory if it does not exist (REQ 3.7).
    - Reads the existing file, or starts with [] if absent (REQ 3.1).
    - Raises json.JSONDecodeError if the existing file is not valid JSON; does
      not overwrite (REQ 3.8).
    - Appends an entry with exactly three fields: timestamp, role, content (REQ 3.4, 3.5).
    - Writes the full updated array back to disk (REQ 3.1, 3.2, 3.3).

    Args:
        identity_id: Scoping key for the conversation; used as the log filename.
        role: Must be "user" or "assistant".
        content: Message text (may be an empty string).

    Raises:
        ValueError: If identity_id contains forbidden characters.
        ValueError: If role is not "user" or "assistant".
        json.JSONDecodeError: If the existing log file exists but is not valid JSON.
    """
    # REQ 8.1 / 8.2 — reject dangerous identity_id values before any path work
    _validate_identity_id(identity_id)

    # REQ 3.6 — validate role before any I/O
    if role not in ("user", "assistant"):
        raise ValueError(
            f"role must be 'user' or 'assistant', got: {role!r}"
        )

    # REQ 3.7 — ensure the logs directory exists
    log_dir = _LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{identity_id}.json")

    # REQ 3.8 — read existing entries; raise JSONDecodeError for corrupt files
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        # json.loads raises json.JSONDecodeError if the content is not valid JSON
        entries: list[dict] = json.loads(raw)
    else:
        entries = []

    # REQ 3.4 / 3.5 — build the new entry with exactly three fields
    entry = {
        "timestamp": _utc_now_iso(),
        "role": role,
        "content": content,
    }

    entries.append(entry)

    # REQ 3.1 / 3.2 / 3.3 — write the full array back so the file is valid JSON
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)
