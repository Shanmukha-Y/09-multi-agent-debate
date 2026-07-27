"""Single source of truth for model/runtime configuration.

Every other module imports MODEL and OLLAMA_BASE_URL from here — this is the
"one config location" the model name lives in. All personas use the same
model; they differ only by system prompt and temperature (see personas.py).
"""

from __future__ import annotations

import os

MODEL = os.environ.get("DEBATE_MODEL", "qwen3.5:9b")
OLLAMA_BASE_URL = os.environ.get("DEBATE_OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Ollama's OpenAI-compatible endpoint accepts any non-empty string as an API
# key; there is no real auth, but the SDK requires something non-empty.
API_KEY = os.environ.get("DEBATE_API_KEY", "ollama")

# Structured-output validate/repair retry loop (Project 01 pattern).
DEFAULT_MAX_ATTEMPTS = 2

# Debate mechanics.
MAX_DEBATE_ROUNDS = 2  # hard cap: round 1 (propose+critique) + round 2 (rebuttal+critique)
CLEAR_WINNER_LEAD_THRESHOLD = 0.15  # top must lead runner-up by >15% of top's score
AGREEMENT_BONUS_WEIGHT = 0.20  # max 20% score bonus when every other proposer agrees
NUMERIC_AGREEMENT_TOLERANCE = 0.01  # relative tolerance for "same" numeric answers

# Network: a shared local Ollama server serves multiple builders concurrently.
# Calls are made strictly sequentially (never asyncio.gather'd) and given a
# generous per-call timeout so one slow generation on a 9B model doesn't
# starve other processes hammering the same server.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("DEBATE_REQUEST_TIMEOUT", "180"))
TRANSPORT_MAX_ATTEMPTS = 3

DB_PATH = os.environ.get("DEBATE_DB_PATH", "debates.db")
