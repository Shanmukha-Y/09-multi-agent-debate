"""The critic: rubric scoring against rubric/critic_rubric.md, with
proposals anonymized to letters (A/B/C) to control persona bias. The critic
never proposes answers — that role separation is structural, not just
convention: this module has no code path that calls a proposer persona.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import ascii_uppercase

from debate import config
from debate.messages import Critique, CriticOutput, Proposal
from debate.structured import CompletionClient, StructuredResult, call_structured

_RUBRIC_PATH = Path(__file__).resolve().parent.parent.parent / "rubric" / "critic_rubric.md"

CRITIC_TEMPERATURE = 0.2  # low: scoring should be consistent, not creative


def load_rubric_text() -> str:
    return _RUBRIC_PATH.read_text()


def anonymize(persona_names: list[str]) -> dict[str, str]:
    """Assign each persona a stable letter label (A, B, C, ...) in the given
    order. Anonymization hides *identity* from the critic; it does not
    reorder or shuffle, since shuffling adds no bias protection here (the
    critic never learns which letter maps to which persona name) while
    making transcripts harder to read."""
    if len(persona_names) > len(ascii_uppercase):
        raise ValueError("too many personas to label with single letters")
    return {name: ascii_uppercase[i] for i, name in enumerate(persona_names)}


def _build_system_prompt() -> str:
    return (
        "You are the Critic. You score proposed answers against a fixed rubric. "
        "You NEVER propose your own answer to the question — your only job is "
        "scoring and written critique of the proposals given to you.\n\n"
        f"RUBRIC:\n{load_rubric_text()}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{"critiques": [{"proposal_id": "A", "scores": {"correctness_risk": <1-10>, '
        '"completeness": <1-10>, "reasoning_quality": <1-10>}, "text": "<critique>"}, '
        "...one entry per proposal_id you were given...]}. "
        "No text outside the JSON object."
    )


def _build_user_prompt(question: str, anonymized_proposals: dict[str, Proposal]) -> str:
    blocks = []
    for label, proposal in sorted(anonymized_proposals.items()):
        blocks.append(
            f"Proposal {label}:\n"
            f"  answer: {proposal.answer}\n"
            f"  reasoning: {proposal.reasoning}\n"
            f"  self_confidence: {proposal.self_confidence}"
        )
    proposals_text = "\n\n".join(blocks)
    return (
        f"Question: {question}\n\n"
        f"Here are {len(anonymized_proposals)} independently-produced proposals, "
        f"labeled {', '.join(sorted(anonymized_proposals))}. Score each one "
        f"against the rubric.\n\n{proposals_text}"
    )


def critique_round(
    question: str,
    proposals: dict[str, Proposal],
    client: CompletionClient,
    max_attempts: int = config.DEFAULT_MAX_ATTEMPTS,
) -> tuple[dict[str, Critique], dict[str, str], int]:
    """Score `proposals` (persona_name -> Proposal) in one anonymized critic
    call. Returns (critiques_by_persona_name, persona_to_letter, tokens_used).
    """
    persona_to_letter = anonymize(list(proposals.keys()))
    letter_to_persona = {v: k for k, v in persona_to_letter.items()}
    anonymized = {persona_to_letter[name]: p for name, p in proposals.items()}

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(question, anonymized)

    result: StructuredResult = call_structured(
        client, system_prompt, user_prompt, CriticOutput, max_attempts=max_attempts
    )
    output: CriticOutput = result.model  # type: ignore[assignment]

    critiques_by_persona: dict[str, Critique] = {}
    for critique in output.critiques:
        persona_name = letter_to_persona.get(critique.proposal_id)
        if persona_name is None:
            # Model hallucinated a label we didn't ask about; skip rather
            # than crash the whole debate over one stray entry.
            continue
        critiques_by_persona[persona_name] = critique

    missing = set(proposals) - set(critiques_by_persona)
    if missing:
        raise ValueError(
            f"critic did not return scores for: {sorted(missing)} "
            f"(raw critiques returned: {[c.proposal_id for c in output.critiques]})"
        )

    return critiques_by_persona, persona_to_letter, result.total_tokens
