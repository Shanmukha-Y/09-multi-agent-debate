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
#
# 180s was too tight and measurably so, not just in theory: a live subset
# bench run against an otherwise-IDLE dedicated server (no contention) still
# timed out 7 of 18 arm attempts, every one of them an estimation question in
# a baseline arm. Server logs during that run show qwen3.5:9b's step-by-step
# persona prompts driving single completions past 10,000+ decoded tokens on
# open-ended Fermi-estimate questions (observed up to ~11,300 tokens on one
# call, still climbing when it was cut off) -- at this server's measured
# ~65-68 tok/s, that alone is ~170s of pure generation before accounting for
# prompt processing or queueing, i.e. it was landing right on the 180s wall.
# self_consistency's higher temperature (0.7 vs personas' base 0.3) makes
# this worse: more sampling diversity also means more rambling, discursive
# reasoning chains on questions that don't have a short closed-form answer.
# 600s gives calls like that comfortable headroom (~40,000 decoded tokens'
# worth at this server's measured throughput) while still being a real
# ceiling -- a call that needs longer than that is an honest, informative
# failure, not an artifact of an arbitrarily tight timeout.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("DEBATE_REQUEST_TIMEOUT", "600"))
TRANSPORT_MAX_ATTEMPTS = 3

DB_PATH = os.environ.get("DEBATE_DB_PATH", "debates.db")
