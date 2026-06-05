"""
Unit tests for falcon/config.py

Requirements covered:
  -  Config raises ValueError with actionable message if GROQ_API_KEY
             is missing, empty, or whitespace-only.
  -  Config exposes default_model (non-empty str), log_dir (non-empty str),
             default_system_prompt (""), and available_models (list).
  -  Config raises ValueError if default_model or log_dir is missing,
             empty, or non-string in config.yaml.

Strategy:
  config.py runs validation at module level, so each test must:
    1. Patch os.environ to inject/remove GROQ_API_KEY.
    2. Patch the open() call (or yaml.safe_load) to supply a controlled YAML dict.
    3. Use importlib.reload(falcon.config) to re-execute module-level code.
    4. Always reload inside a try/finally to leave the module cache consistent.
"""

import importlib
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal YAML bytes that represent a valid, complete config.
_VALID_YAML_CONTENT = (
    "default_model: test-model\n"
    "log_dir: logs\n"
    "available_models:\n"
    "  - test-model\n"
    "  - other-model\n"
    "default_system_prompt: ''\n"
)

_VALID_CFG_DICT = {
    "default_model": "test-model",
    "log_dir": "logs",
    "available_models": ["test-model", "other-model"],
    "default_system_prompt": "",
}


def _reload_config(env_overrides: dict, cfg_dict: dict | None = None):
    """
    Reload falcon.config with the given environment and YAML mapping.

    Parameters
    ----------
    env_overrides : dict
        Key/value pairs to add (or replace) in os.environ before reload.
        Pass ``{"GROQ_API_KEY": ""}`` to simulate a missing key, or
        ``{"GROQ_API_KEY": "sk-test-key"}`` for a valid key.
    cfg_dict : dict | None
        The object that yaml.safe_load will return.  Defaults to a minimal
        valid configuration when not provided.

    Returns
    -------
    module
        The reloaded falcon.config module (only on success).

    Raises
    ------
    Any exception raised by config.py during import (e.g. ValueError).
    """
    if cfg_dict is None:
        cfg_dict = _VALID_CFG_DICT

    # Build a clean environment: start from a copy, apply overrides.
    clean_env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
    clean_env.update(env_overrides)

    with (
        patch.dict(os.environ, clean_env, clear=True),
        patch("builtins.open", mock_open(read_data=_VALID_YAML_CONTENT)),
        patch("yaml.safe_load", return_value=cfg_dict),
        # load_dotenv is a no-op here; we control os.environ directly.
        patch("dotenv.load_dotenv"),
    ):
        import falcon.config as config_module
        return importlib.reload(config_module)


# ---------------------------------------------------------------------------
# GROQ_API_KEY validation
# ---------------------------------------------------------------------------


class TestGroqApiKeyValidation:
    """ ValueError is raised when GROQ_API_KEY is absent or blank."""

    def test_missing_key_raises_value_error(self):
        """GROQ_API_KEY not present in environment → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(env_overrides={})  # no key at all

    def test_empty_key_raises_value_error(self):
        """GROQ_API_KEY set to empty string → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(env_overrides={"GROQ_API_KEY": ""})

    def test_whitespace_only_key_raises_value_error(self):
        """GROQ_API_KEY set to whitespace only → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(env_overrides={"GROQ_API_KEY": "   "})

    def test_whitespace_only_key_tabs_raises_value_error(self):
        """GROQ_API_KEY set to tab characters only → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(env_overrides={"GROQ_API_KEY": "\t\n"})

    def test_missing_key_error_message_is_actionable(self):
        """
        The ValueError message must name the missing key and tell the user
        how to fix it ("human-readable message that identifies the
        missing key and instructs the user to add it").
        """
        with pytest.raises(ValueError) as exc_info:
            _reload_config(env_overrides={})

        msg = str(exc_info.value).upper()
        # Message must at minimum reference the key name.
        assert "GROQ_API_KEY" in msg

    def test_empty_key_error_message_is_actionable(self):
        """Empty-string key also yields an actionable message."""
        with pytest.raises(ValueError) as exc_info:
            _reload_config(env_overrides={"GROQ_API_KEY": ""})

        msg = str(exc_info.value).upper()
        assert "GROQ_API_KEY" in msg

    def test_valid_key_does_not_raise(self):
        """A non-empty, non-whitespace key must not raise."""
        cfg = _reload_config(env_overrides={"GROQ_API_KEY": "sk-test-valid-key"})
        assert cfg.GROQ_API_KEY == "sk-test-valid-key"


# ---------------------------------------------------------------------------
#  — Exposed fields after a successful load
# ---------------------------------------------------------------------------


class TestValidConfigLoad:
    """ 5.2: Config exposes required fields with correct types and values."""

    @pytest.fixture(autouse=True)
    def loaded_config(self):
        """Reload config once per test with a valid env + YAML."""
        self.cfg = _reload_config(
            env_overrides={"GROQ_API_KEY": "sk-unit-test-key"},
            cfg_dict=_VALID_CFG_DICT,
        )

    def test_groq_api_key_is_exposed(self):
        """GROQ_API_KEY attribute matches the value from the environment."""
        assert self.cfg.GROQ_API_KEY == "sk-unit-test-key"

    def test_default_model_is_non_empty_string(self):
        """default_model must be a non-empty string."""
        assert isinstance(self.cfg.default_model, str)
        assert self.cfg.default_model  # truthy → non-empty

    def test_default_model_matches_yaml(self):
        """default_model value matches what the YAML provided."""
        assert self.cfg.default_model == "test-model"

    def test_log_dir_is_non_empty_string(self):
        """log_dir must be a non-empty string."""
        assert isinstance(self.cfg.log_dir, str)
        assert self.cfg.log_dir  # truthy → non-empty

    def test_log_dir_matches_yaml(self):
        """log_dir value matches what the YAML provided."""
        assert self.cfg.log_dir == "logs"

    def test_available_models_is_list(self):
        """available_models must be a list (may be empty)."""
        assert isinstance(self.cfg.available_models, list)

    def test_available_models_contains_expected_entries(self):
        """available_models contains the models from the YAML."""
        assert "test-model" in self.cfg.available_models
        assert "other-model" in self.cfg.available_models

    def test_default_system_prompt_is_non_empty_string(self):
        """default_system_prompt is a non-empty string (the neutrality prompt)."""
        assert isinstance(self.cfg.default_system_prompt, str)
        assert self.cfg.default_system_prompt  # non-empty


# ---------------------------------------------------------------------------
#  — default_system_prompt is always ""
# ---------------------------------------------------------------------------


class TestDefaultSystemPromptAlwaysSet:
    """
    default_system_prompt is a hardcoded neutrality instruction — always a
    non-empty string regardless of what config.yaml contains.
    """

    def test_default_system_prompt_is_non_empty_regardless_of_yaml(self):
        """Even if YAML ships a non-empty default_system_prompt, the hardcoded value is used."""
        cfg_with_filled_prompt = {
            **_VALID_CFG_DICT,
            "default_system_prompt": "You are a helpful assistant.",
        }
        cfg = _reload_config(
            env_overrides={"GROQ_API_KEY": "sk-test"},
            cfg_dict=cfg_with_filled_prompt,
        )
        assert isinstance(cfg.default_system_prompt, str)
        assert cfg.default_system_prompt  # non-empty

    def test_default_system_prompt_is_non_empty_with_minimal_yaml(self):
        """Minimal YAML (no default_system_prompt key) → hardcoded value is still set."""
        minimal_cfg = {"default_model": "some-model", "log_dir": "logs"}
        cfg = _reload_config(
            env_overrides={"GROQ_API_KEY": "sk-test"},
            cfg_dict=minimal_cfg,
        )
        assert isinstance(cfg.default_system_prompt, str)
        assert cfg.default_system_prompt  # non-empty


# ---------------------------------------------------------------------------
#  — config.yaml field validation
# ---------------------------------------------------------------------------


class TestYamlFieldValidation:
    """ ValueError is raised for invalid default_model or log_dir."""

    # --- default_model ---

    def test_missing_default_model_raises_value_error(self):
        """default_model absent from YAML → ValueError."""
        cfg = {k: v for k, v in _VALID_CFG_DICT.items() if k != "default_model"}
        with pytest.raises(ValueError):
            _reload_config(env_overrides={"GROQ_API_KEY": "sk-test"}, cfg_dict=cfg)

    def test_empty_default_model_raises_value_error(self):
        """default_model is empty string → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(
                env_overrides={"GROQ_API_KEY": "sk-test"},
                cfg_dict={**_VALID_CFG_DICT, "default_model": ""},
            )

    def test_whitespace_default_model_raises_value_error(self):
        """default_model is whitespace only → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(
                env_overrides={"GROQ_API_KEY": "sk-test"},
                cfg_dict={**_VALID_CFG_DICT, "default_model": "   "},
            )

    def test_non_string_default_model_raises_value_error(self):
        """default_model is an integer → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(
                env_overrides={"GROQ_API_KEY": "sk-test"},
                cfg_dict={**_VALID_CFG_DICT, "default_model": 42},
            )

    def test_none_default_model_raises_value_error(self):
        """default_model is None (unset YAML key) → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(
                env_overrides={"GROQ_API_KEY": "sk-test"},
                cfg_dict={**_VALID_CFG_DICT, "default_model": None},
            )

    # --- log_dir ---

    def test_missing_log_dir_raises_value_error(self):
        """log_dir absent from YAML → ValueError."""
        cfg = {k: v for k, v in _VALID_CFG_DICT.items() if k != "log_dir"}
        with pytest.raises(ValueError):
            _reload_config(env_overrides={"GROQ_API_KEY": "sk-test"}, cfg_dict=cfg)

    def test_empty_log_dir_raises_value_error(self):
        """log_dir is empty string → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(
                env_overrides={"GROQ_API_KEY": "sk-test"},
                cfg_dict={**_VALID_CFG_DICT, "log_dir": ""},
            )

    def test_whitespace_log_dir_raises_value_error(self):
        """log_dir is whitespace only → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(
                env_overrides={"GROQ_API_KEY": "sk-test"},
                cfg_dict={**_VALID_CFG_DICT, "log_dir": "   "},
            )

    def test_non_string_log_dir_raises_value_error(self):
        """log_dir is a list → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(
                env_overrides={"GROQ_API_KEY": "sk-test"},
                cfg_dict={**_VALID_CFG_DICT, "log_dir": ["logs"]},
            )

    def test_none_log_dir_raises_value_error(self):
        """log_dir is None (unset YAML key) → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(
                env_overrides={"GROQ_API_KEY": "sk-test"},
                cfg_dict={**_VALID_CFG_DICT, "log_dir": None},
            )

    # --- config.yaml missing entirely ---

    def test_missing_config_yaml_raises_value_error(self):
        """FileNotFoundError from open() is converted to ValueError."""
        clean_env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        clean_env["GROQ_API_KEY"] = "sk-test"

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch("builtins.open", side_effect=FileNotFoundError("no such file")),
            patch("dotenv.load_dotenv"),
        ):
            import falcon.config as config_module
            with pytest.raises(ValueError):
                importlib.reload(config_module)

    # --- top-level YAML is not a mapping ---

    def test_yaml_list_at_root_raises_value_error(self):
        """YAML whose top level is a list (not a dict) → ValueError."""
        with pytest.raises(ValueError):
            _reload_config(
                env_overrides={"GROQ_API_KEY": "sk-test"},
                cfg_dict=["not", "a", "dict"],  # type: ignore[arg-type]
            )
