"""
Property-based tests for config.py — Property 5: Config Completeness.

Validates: Requirements 5.1, 5.2, 5.3

Because config.py executes validation at import time (module-level code),
each test must evict `falcon.config` from `sys.modules` and reload it with
a controlled environment and mocked YAML so the validation logic re-runs
under the desired conditions.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid YAML config dict used as the default for "happy-path" tests.
VALID_YAML_CFG = {
    "default_model": "llama3-70b-8192",
    "log_dir": "logs",
    "available_models": ["llama3-70b-8192"],
    "default_system_prompt": "",
}


def _reload_config(env_override: dict, yaml_cfg: dict):
    """
    Reload `falcon.config` with a fully controlled environment.

    Parameters
    ----------
    env_override : dict
        The entire os.environ mapping to use (replaces real env).
    yaml_cfg : dict | None
        The dict that `yaml.safe_load` will return.  Pass `None` to
        simulate a missing config.yaml (FileNotFoundError).

    Returns
    -------
    module
        The freshly-imported `falcon.config` module.

    Raises
    ------
    Whatever `falcon.config` raises on import (ValueError, etc.)
    """
    # Remove any previously cached version of the module.
    sys.modules.pop("falcon.config", None)

    with patch.dict("os.environ", env_override, clear=True), \
         patch("dotenv.load_dotenv", return_value=None), \
         patch("builtins.open", MagicMock()), \
         patch("yaml.safe_load", return_value=yaml_cfg):
        config = importlib.import_module("falcon.config")

    return config


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strings that are absent/empty/whitespace-only — must all trigger ValueError.
invalid_api_key_strategy = st.one_of(
    # Absent key is handled by the fixture; here we cover empty and whitespace.
    st.just(""),
    st.text(alphabet=st.characters(whitelist_categories=("Zs",)), min_size=1, max_size=20),
    st.just("   "),
    st.just("\t"),
    st.just("\n"),
    st.just(" \t \n "),
)

# Keys that are definitely valid (non-empty, non-whitespace, no null bytes).
valid_api_key_strategy = st.text(
    alphabet=st.characters(
        blacklist_categories=("Cs",),
        blacklist_characters=(" ", "\t", "\n", "\r", "\x00"),
    ),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip() != "")

# Non-empty strings for default_model and log_dir.
non_empty_str = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
    min_size=1,
    max_size=80,
).filter(lambda s: s.strip() != "")

# Invalid values for required YAML fields (absent, empty, whitespace, non-string).
invalid_field_strategy = st.one_of(
    st.just(None),          # missing key returns None
    st.just(""),            # empty string
    st.just("   "),         # whitespace-only
    st.integers(),          # wrong type
    st.lists(st.text()),    # wrong type (list)
)


# ---------------------------------------------------------------------------
# Property 5a — GROQ_API_KEY absent raises ValueError
#
# Validates: Requirement 5.1
# ---------------------------------------------------------------------------

def test_missing_api_key_raises():
    """GROQ_API_KEY absent from environment must raise ValueError."""
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        _reload_config({}, VALID_YAML_CFG)


@given(key=invalid_api_key_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
def test_invalid_api_key_raises_value_error(key):
    """
    **Validates: Requirements 5.1**

    For any empty or whitespace-only GROQ_API_KEY value, config.py must
    raise ValueError before exposing any configuration.
    """
    env = {"GROQ_API_KEY": key}
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        _reload_config(env, VALID_YAML_CFG)


# ---------------------------------------------------------------------------
# Property 5b — Successful load exposes correct flat namespace
#
# Validates: Requirement 5.2
# ---------------------------------------------------------------------------

@given(
    api_key=valid_api_key_strategy,
    model=non_empty_str,
    log_dir=non_empty_str,
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
def test_successful_load_exposes_valid_namespace(api_key, model, log_dir):
    """
    **Validates: Requirements 5.2**

    After a successful load:
    - default_model is a non-empty string
    - log_dir is a non-empty string
    - default_system_prompt is exactly ""
    - available_models is a list
    """
    yaml_cfg = {
        "default_model": model,
        "log_dir": log_dir,
        "available_models": ["some-model"],
        "default_system_prompt": "",
    }
    config = _reload_config({"GROQ_API_KEY": api_key}, yaml_cfg)

    assert isinstance(config.default_model, str) and config.default_model.strip() != "", \
        f"default_model must be a non-empty string, got {config.default_model!r}"
    assert isinstance(config.log_dir, str) and config.log_dir.strip() != "", \
        f"log_dir must be a non-empty string, got {config.log_dir!r}"
    assert isinstance(config.default_system_prompt, str) and config.default_system_prompt, \
        f"default_system_prompt must be a non-empty string, got {config.default_system_prompt!r}"
    assert isinstance(config.available_models, list), \
        f"available_models must be a list, got {type(config.available_models)}"


# ---------------------------------------------------------------------------
# Property 5c — Missing/invalid required YAML fields raise ValueError
#
# Validates: Requirement 5.3
# ---------------------------------------------------------------------------

@given(bad_value=invalid_field_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
def test_invalid_default_model_raises_value_error(bad_value):
    """
    **Validates: Requirements 5.3**

    If default_model is absent, empty, whitespace-only, or a non-string,
    config.py must raise ValueError.
    """
    yaml_cfg = {
        "default_model": bad_value,
        "log_dir": "logs",
        "available_models": [],
        "default_system_prompt": "",
    }
    with pytest.raises(ValueError):
        _reload_config({"GROQ_API_KEY": "valid-key"}, yaml_cfg)


@given(bad_value=invalid_field_strategy)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
def test_invalid_log_dir_raises_value_error(bad_value):
    """
    **Validates: Requirements 5.3**

    If log_dir is absent, empty, whitespace-only, or a non-string,
    config.py must raise ValueError.
    """
    yaml_cfg = {
        "default_model": "llama3-70b-8192",
        "log_dir": bad_value,
        "available_models": [],
        "default_system_prompt": "",
    }
    with pytest.raises(ValueError):
        _reload_config({"GROQ_API_KEY": "valid-key"}, yaml_cfg)


# ---------------------------------------------------------------------------
# Edge cases — ensure clean teardown doesn't leak state between tests
# ---------------------------------------------------------------------------

def test_module_not_cached_after_error():
    """
    falcon.config must NOT remain in sys.modules when import raises ValueError,
    so that subsequent tests get a fresh import attempt.
    """
    with pytest.raises(ValueError):
        _reload_config({}, VALID_YAML_CFG)  # no GROQ_API_KEY → should raise

    # After a failed reload, the module must NOT be cached.
    assert "falcon.config" not in sys.modules, (
        "falcon.config should not remain in sys.modules after a failed import"
    )
