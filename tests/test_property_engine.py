"""
Property-based tests for engine.py.

Property 2: Payload Transparency  (task 6.2)
  Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5

Property 4: Neutrality  (task 6.3)
  Validates: Requirements 4.1, 4.2, 4.3, 4.7
"""

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from falcon.engine import build_payload


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Any string (including empty, whitespace-only, unicode)
any_text = st.text(min_size=0, max_size=200)

# Non-empty, non-whitespace-only strings
non_empty_prompt = st.text(min_size=1, max_size=200).filter(lambda s: s.strip() != "")

# Whitespace-only strings
whitespace_prompt = st.text(
    alphabet=st.characters(whitelist_categories=("Zs",)),
    min_size=1,
    max_size=50,
)

valid_role = st.sampled_from(["user", "assistant"])

message_dict = st.fixed_dictionaries({
    "role": valid_role,
    "content": any_text,
})

message_list = st.lists(message_dict, min_size=0, max_size=20)


# ---------------------------------------------------------------------------
# Property 2: Payload Transparency
# Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
# ---------------------------------------------------------------------------

@given(system_prompt=non_empty_prompt, messages=message_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_non_empty_prompt_is_first_entry(system_prompt: str, messages: list[dict]):
    """
    REQ 2.1: When system_prompt is non-empty, raw_payload[0] == {"role": "system", "content": system_prompt}
    with content equal to system_prompt without any modification.
    """
    payload = build_payload(system_prompt, messages)
    assert len(payload) >= 1
    assert payload[0] == {"role": "system", "content": system_prompt}, (
        f"First entry must be the system message with unmodified content. "
        f"Got: {payload[0]!r}"
    )


@given(system_prompt=st.just("") | whitespace_prompt, messages=message_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_empty_or_whitespace_prompt_has_no_system_entry(system_prompt: str, messages: list[dict]):
    """
    REQ 2.2 / REQ 2.5: When system_prompt is empty or whitespace-only,
    no entry with role == "system" appears in the payload.
    """
    payload = build_payload(system_prompt, messages)
    system_entries = [e for e in payload if e.get("role") == "system"]
    assert system_entries == [], (
        f"Payload must not contain system entry for prompt={system_prompt!r}. "
        f"Found: {system_entries!r}"
    )


@given(system_prompt=any_text, messages=message_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_user_and_assistant_messages_preserved_in_order(system_prompt: str, messages: list[dict]):
    """
    REQ 2.3: All user/assistant messages appear in payload in the same order
    with identical content values.
    """
    payload = build_payload(system_prompt, messages)
    # Filter out the system entry (if any)
    non_system = [e for e in payload if e.get("role") != "system"]

    assert len(non_system) == len(messages), (
        f"Expected {len(messages)} non-system entries, got {len(non_system)}"
    )
    for idx, (expected, actual) in enumerate(zip(messages, non_system)):
        assert actual["role"] == expected["role"], (
            f"Entry {idx}: role mismatch expected={expected['role']!r} actual={actual['role']!r}"
        )
        assert actual["content"] == expected["content"], (
            f"Entry {idx}: content was modified — expected={expected['content']!r} actual={actual['content']!r}"
        )


@given(system_prompt=any_text, messages=message_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_payload_length_equals_messages_plus_system_flag(system_prompt: str, messages: list[dict]):
    """
    REQ 2.4: len(payload) == len(messages) + (1 if system_prompt.strip() else 0).
    """
    payload = build_payload(system_prompt, messages)
    expected_len = len(messages) + (1 if system_prompt.strip() else 0)
    assert len(payload) == expected_len, (
        f"payload length mismatch: expected {expected_len}, got {len(payload)} "
        f"(system_prompt={system_prompt!r}, messages count={len(messages)})"
    )


# ---------------------------------------------------------------------------
# Property 4: Neutrality
# Validates: Requirements 4.1, 4.2, 4.3, 4.7
# ---------------------------------------------------------------------------

@given(messages=message_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_empty_prompt_produces_no_system_entry(messages: list[dict]):
    """
    REQ 4.1: build_payload("", messages) must return exactly len(messages)
    entries with no system entry.
    """
    payload = build_payload("", messages)
    assert len(payload) == len(messages), (
        f"Expected {len(messages)} entries for empty prompt, got {len(payload)}"
    )
    assert all(e["role"] != "system" for e in payload), (
        "No system entry expected for empty prompt"
    )


@given(system_prompt=whitespace_prompt, messages=message_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_whitespace_only_prompt_produces_no_system_entry(system_prompt: str, messages: list[dict]):
    """
    REQ 4.1: build_payload(whitespace_only_prompt, messages) must return
    exactly len(messages) entries with no system entry.
    """
    payload = build_payload(system_prompt, messages)
    assert len(payload) == len(messages), (
        f"Expected {len(messages)} entries for whitespace-only prompt, got {len(payload)}"
    )
    assert all(e["role"] != "system" for e in payload), (
        f"No system entry expected for whitespace-only prompt={system_prompt!r}"
    )


@given(system_prompt=any_text, messages=message_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_message_content_is_not_modified(system_prompt: str, messages: list[dict]):
    """
    REQ 4.3: Every HumanMessage/AIMessage content in the payload equals the
    input string with zero modification — no prefix, suffix, or whitespace normalization.
    """
    payload = build_payload(system_prompt, messages)
    non_system = [e for e in payload if e["role"] != "system"]
    for idx, (original, entry) in enumerate(zip(messages, non_system)):
        assert entry["content"] == original["content"], (
            f"Entry {idx}: content was modified. "
            f"Original={original['content']!r}, Got={entry['content']!r}"
        )


@given(system_prompt=non_empty_prompt, messages=message_list)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_system_prompt_content_not_modified(system_prompt: str, messages: list[dict]):
    """
    REQ 4.2: When system_prompt is non-empty, the system entry's content
    must equal system_prompt with no modification, prefix, suffix, or
    whitespace normalization.
    """
    payload = build_payload(system_prompt, messages)
    system_entries = [e for e in payload if e["role"] == "system"]
    assert len(system_entries) == 1
    assert system_entries[0]["content"] == system_prompt, (
        f"System entry content was modified. "
        f"Input={system_prompt!r}, Got={system_entries[0]['content']!r}"
    )
