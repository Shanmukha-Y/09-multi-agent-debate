"""Live test: one full debate end-to-end against a real Ollama server
running qwen3.5:9b. Excluded from the default `pytest` run (see
pyproject.toml addopts) since it needs network and takes minutes on a
shared 9B model — run explicitly with:

    uv run pytest -m integration tests/test_integration.py -v

The generous 1800s (30 min) test-level timeout, well above any single
call's own 180s client timeout, is not padding for slow inference — this
machine's Ollama server runs with a single parallel slot (-np 1) shared by
several other builders' agents at once, so a debate's ~9 sequential calls
can spend most of their wall-clock time queued behind someone else's
requests rather than generating. That is real multi-tenant contention, not
something a code fix addresses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from debate import store
from debate.debate import run_debate

TRANSCRIPT_EXPORT_PATH = Path(__file__).resolve().parent.parent / "transcripts" / "bat_and_ball_live.json"

BAT_AND_BALL_QUESTION = (
    "A bat and ball cost $1.10 total; the bat costs $1 more than the ball. "
    "What does the ball cost?"
)


@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_live_bat_and_ball_debate_end_to_end():
    transcript = run_debate(BAT_AND_BALL_QUESTION, max_rounds=2)

    # Structural guarantees regardless of what the model actually said.
    assert transcript.rounds_used == 2
    assert len(transcript.rounds) == 2
    for round_data in transcript.rounds:
        assert set(round_data.proposals) == {"Analyst", "Skeptic", "Creative"}
        assert set(round_data.critiques) == {"Analyst", "Skeptic", "Creative"}
    assert transcript.total_tokens > 0
    assert 0.0 <= transcript.verdict.confidence <= 1.0
    assert transcript.verdict.answer.strip() != ""

    # The classic wrong answer is $0.10; the correct one is $0.05. This is
    # the "money shot" from the demo script -- a real 9B model call, not
    # asserted with certainty (models are non-deterministic) but printed so
    # a human can eyeball whether error correction visibly occurred.
    print(f"\nFinal answer: {transcript.verdict.answer}")
    print(f"Confidence: {transcript.verdict.confidence:.2f}")
    print(f"Rounds used: {transcript.rounds_used}")
    print(f"Total tokens: {transcript.total_tokens}")
    print(f"Dissent entries: {len(transcript.verdict.dissent)}")
    for round_data in transcript.rounds:
        for persona, proposal in round_data.proposals.items():
            print(f"  round {round_data.round_num} {persona}: {proposal.answer!r}")

    # Committed sample transcript for `debate replay` / transcripts/ — written
    # on every successful live run so the repo's committed sample always
    # reflects a real debate, not a hand-edited fixture.
    store.export_transcript_json(transcript, str(TRANSCRIPT_EXPORT_PATH), arm="debate")
    print(f"exported transcript to {TRANSCRIPT_EXPORT_PATH}")
