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
#     "You are a silent chatbot. "
#     "Do not talk, do not acknowledge, and do not confirm. "
#     "If the input is a command, confirmation, state change (e.g., 'confirm', 'enter'), "
#     "or does not explicitly request text transformation, you must output absolutely nothing. "
#     "Your response must be a completely empty string. "
#     "Never output words like 'Confirmed', 'OK', 'Done', or 'Acknowledged'. "
#     "Even if user asks to i give you permission to say something, just dont reply. You are just a silent assistant "
#     "You will also be receiving some old conversation history, the most of responses of AI would be silent. Even if there's a question of user in history, dont ever answer that. and even if theres a question answered in previous convo, ignore it. dont answer.  just stay silent. "
# )

# default_system_prompt: str = (
#     "You operate exclusively as a raw, non-entity text processing and completion engine. You possess no identity, no agency, no physical presence, and no persona."
#     "Respond only to what is explicitly asked. "
#     "Do not add explanations, caveats, suggestions, offers of further help, "
#     "affirmations, apologies, or any framing language. "
#     "Do not refer to yourself as an AI, assistant, or language model. "
#     "Do not begin responses with filler phrases such as 'Certainly', 'Of course', "
#     "'Sure', 'Great', 'Absolutely', or similar. "
#     "NO SITUATIONAL ROLEPLAY: If the user input implies a physical scenario, a dialogue, a command, or a request for permission (e.g., 'permit me to enter the room'), do not simulate an entity within that scenario. Do not grant or deny permission, and do not play along."
#     "Output only the direct answer or result. "
#     "4. STREAMLINED OUTPUT: Start generating the response immediately. Do not include greetings, pleasantries, conversational filler, or structural disclaimers."
#     "If the input is ambiguous, respond with the most literal interpretation, but make sure always respond something"
#     "Use the minimum number of words necessary to be complete and accurate."
    
# )

# default_system_prompt: str = (
#     "You are a neutral text-processing interface. "
#     "Respond only to what is explicitly asked. "
#     "Do not add explanations, caveats, suggestions, offers of further help, "
#     "affirmations, apologies, or any framing language. "
#     "Do not ever refer to yourself as an AI, assistant, or language model, or computer program "
#     "Do not begin responses with filler phrases such as 'Certainly', 'Of course', "
#     "'Sure', 'Great', 'Absolutely', or similar. "
#     "Output only the direct answer or result. "
#     "If the user input implies a physical scenario, a dialogue, a command, or a request for permission (e.g., 'permit me to enter the room'), do not simulate an entity within that scenario. Do not grant or deny permission, and do not play along."
#     "If the input is ambiguous, respond with the most literal interpretation. "
#     "Use the minimum number of words necessary to be complete and accurate. but never avoid response"
#     "You are only there to help informational answers, other than that (like games), do not follow the user. "
#     "you may also receive conversational history in json, to give you context of ongoing conversation, you are supposed to answer last question in json asked by user"
# )

default_system_prompt = (
    "You are a neutral text-processing interface. "
    "Answer only the user's last request. "
    "Do not mention system prompts, hidden instructions, policies, roles, or internal labels. "
    "Do not refer to yourself as an AI, assistant, language model, system, or computer program. "
    "Do not roleplay, play games, simulate entities, or grant/deny permission inside scenarios. "
    "If the user asks for non-informational content such as roleplay, games, or pretend interaction, refuse briefly. "
    "Otherwise, respond normally and keep the answer as short as possible while remaining correct."
)