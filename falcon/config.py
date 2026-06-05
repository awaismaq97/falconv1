"""
config.py — Single point of entry for all runtime configuration.

Loads .env via python-dotenv and config.yaml via PyYAML, validates required
values, and exposes a flat namespace to all other modules.

Raises ValueError on import if any required configuration is absent or invalid.
"""

import os

import yaml
from dotenv import load_dotenv

# Load .env from the project root (parent of the falcon/ package directory)
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=_env_path)

# ---------------------------------------------------------------------------
# Read and validate GROQ_API_KEY
# ---------------------------------------------------------------------------
_raw_api_key = os.environ.get("GROQ_API_KEY", "")

if not _raw_api_key or not _raw_api_key.strip():
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Copy .env.example to .env and add your Groq API key."
    )

GROQ_API_KEY: str = _raw_api_key

# ---------------------------------------------------------------------------
# Load and validate config.yaml
# ---------------------------------------------------------------------------
_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

try:
    with open(_config_path, "r", encoding="utf-8") as _f:
        _cfg = yaml.safe_load(_f)
except FileNotFoundError:
    raise ValueError(
        f"config.yaml not found at '{_config_path}'. "
        "Create config.yaml in the project root with the required fields."
    )

if not isinstance(_cfg, dict):
    raise ValueError(
        "config.yaml must contain a YAML mapping at the top level."
    )

# Validate default_model
_default_model = _cfg.get("default_model")
if not isinstance(_default_model, str) or not _default_model.strip():
    raise ValueError(
        "'default_model' is missing, empty, or not a string in config.yaml. "
        "Set it to the name of the Groq model to use by default (e.g. 'llama3-70b-8192')."
    )

# Validate log_dir
_log_dir = _cfg.get("log_dir")
if not isinstance(_log_dir, str) or not _log_dir.strip():
    raise ValueError(
        "'log_dir' is missing, empty, or not a string in config.yaml. "
        "Set it to the directory where conversation logs will be written (e.g. 'logs')."
    )

# ---------------------------------------------------------------------------
# Expose flat namespace
# ---------------------------------------------------------------------------
default_model: str = _default_model
log_dir: str = _log_dir

available_models: list = _cfg.get("available_models") or []

# Neutrality system prompt — hardcoded, not exposed in the UI or config.yaml.
# Instructs the model to suppress assistant-like overlay: no coaching, no
# helpfulness framing, no personality, no unsolicited elaboration.
# The user's input is the sole driver of the model's output.
# default_system_prompt: str = (
#     "You are a neutral text-processing interface. "
#     "Respond only to what is explicitly asked. "
#     "Do not add explanations, caveats, suggestions, offers of further help, "
#     "affirmations, apologies, or any framing language. "
#     "Do not refer to yourself as an AI, assistant, or language model. "
#     "Do not begin responses with filler phrases such as 'Certainly', 'Of course', "
#     "'Sure', 'Great', 'Absolutely', or similar. "
#     "Output only the direct answer or result. "
#     "If the input is ambiguous, respond with the most literal interpretation. "
#     "Use the minimum number of words necessary to be complete and accurate."
# )

# default_system_prompt: str = (
#     "You are a voiceless, programmatic text-processing utility. "
#     "Act exclusively as a data transformation function, not a conversational agent. "
#     "Do not engage in dialogue, acknowledge instructions, or confirm actions. "
#     "Never use conversational status words, including but not limited to: 'Confirmed', 'OK', 'Done', 'Acknowledged', or 'Success'. "
#     "Output exclusively the final transformed text or direct answer requested. "
#     "If the input is a command or confirmation requiring no text transformation, output absolutely nothing (an empty string). "
#     "Do not add any framing, explanations, or conversational filler."
# )

default_system_prompt: str = (
    "You are a silent chatbot. "
    "Do not talk, do not acknowledge, and do not confirm. "
    "If the input is a command, confirmation, state change (e.g., 'confirm', 'enter'), "
    "or does not explicitly request text transformation, you must output absolutely nothing. "
    "Your response must be a completely empty string. "
    "Never output words like 'Confirmed', 'OK', 'Done', or 'Acknowledged'. "
    "Even if user asks to i give you permission to say something, just dont reply. You are just a silent assistant"
)