"""Synthesizes the FinalVerdict from an already-decided vote.

Everything that determines *correctness and auditability* — which answer
wins, the confidence score, and whether dissent gets reported — is plain
Python arithmetic in this module, not an LLM call. The one LLM call
`synthesize()` makes only produces the narrative `reasoning_summary` prose;
it cannot change the answer, the confidence number, or the dissent list, so
an off-day from the model here degrades the write-up, never the correctness
of the verdict. If that call fails schema validation entirely, we fall back
to a deterministic summary (`_fallback_summary`) rather than losing the
(already-correct) verdict over a formatting hiccup.

Never averages contradictory answers into mush: the winner (by
voting.score_round) is always picked and attributed by name. When the vote
is a split, we still pick the top scorer as the primary answer but flag
`is_split=True` and drop the confidence via the split penalty below — the
dishonesty this module refuses to commit is pretending a marginal win is a
consensus, not declining to pick at all.
"""

from __future__ import annotations

import logging

from debate import config
from debate.messages import AggregatorSynthesis, Critique, DissentEntry, FinalVerdict, Proposal
from debate.structured import CompletionClient, SchemaEnforcementError, call_structured
from debate.voting import VoteResult, answers_agree, cluster_answers

logger = logging.getLogger(__name__)

AGGREGATOR_TEMPERATURE = 0.3

ROUND_PENALTY_SINGLE = 1.0   # resolved in round 1, no rebuttal needed
ROUND_PENALTY_REBUTTAL = 0.9  # needed a rebuttal round to reach this verdict
SPLIT_PENALTY = 0.7          # vote never cleared the 15% lead threshold


def compute_confidence(vote: VoteResult, rounds_used: int) -> float:
    """confidence = critic_score_fraction * agreement_fraction * round_factor * split_factor

    - critic_score_fraction: winner's critic_total / 30 (max possible rubric score)
    - agreement_fraction: fraction of ALL proposers (winner included) whose
      answer agrees with the winning answer
    - round_factor: penalized if a rebuttal round was needed to converge
    - split_factor: penalized if the vote never reached a clear (>15% lead) winner
    """
    winner = vote.winner
    critic_score_fraction = winner.critic_total / 30.0
    agreement_fraction = (winner.agreeing_peers + 1) / (winner.total_peers + 1)
    round_factor = ROUND_PENALTY_SINGLE if rounds_used == 1 else ROUND_PENALTY_REBUTTAL
    split_factor = SPLIT_PENALTY if not vote.is_clear_winner else 1.0

    confidence = critic_score_fraction * agreement_fraction * round_factor * split_factor
    return max(0.0, min(1.0, confidence))


def compute_dissent(final_proposals: dict[str, Proposal], winner_name: str) -> list[DissentEntry]:
    """Group every proposer whose final-round answer disagrees with the
    winner's answer into dissent entries (proposers who agree with *each
    other* but not the winner share one entry)."""
    winner_answer = final_proposals[winner_name].answer
    dissenters = {
        name: p
        for name, p in final_proposals.items()
        if name != winner_name and not answers_agree(p.answer, winner_answer)
    }
    if not dissenters:
        return []

    dissenter_names = list(dissenters)
    groups_by_index = cluster_answers([dissenters[name].answer for name in dissenter_names])

    entries = []
    for group in groups_by_index:
        names = [dissenter_names[i] for i in group]
        rep = dissenters[names[0]]
        entries.append(DissentEntry(personas=sorted(names), answer=rep.answer, reasoning=rep.reasoning))
    return entries


def _build_summary_prompt(
    question: str,
    winner_name: str,
    winner_proposal: Proposal,
    winner_critique: Critique,
    dissent: list[DissentEntry],
    is_split: bool,
    confidence: float,
) -> tuple[str, str]:
    system_prompt = (
        "You are the Aggregator. A winner has ALREADY been decided by "
        "deterministic vote math — you do not get to change it, pick a "
        "different answer, or blend it with a dissenting answer. Your only "
        "job is to write a short, honest 2-4 sentence explanation of why "
        "this answer won, citing the critic's assessment. If there is "
        "dissent, acknowledge it briefly but do not soften the final answer "
        "into a hedge.\n\n"
        "Respond with ONLY a JSON object: "
        '{"reasoning_summary": "<2-4 sentences>"}. No text outside the JSON object.'
    )
    dissent_text = (
        "\n".join(f"- {', '.join(d.personas)} disagreed, proposing: {d.answer}" for d in dissent)
        if dissent
        else "None — no proposer maintained disagreement in the final round."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Winning answer (from {winner_name}): {winner_proposal.answer}\n"
        f"Winning reasoning: {winner_proposal.reasoning}\n"
        f"Critic's assessment: {winner_critique.text} "
        f"(correctness_risk={winner_critique.scores.correctness_risk}, "
        f"completeness={winner_critique.scores.completeness}, "
        f"reasoning_quality={winner_critique.scores.reasoning_quality})\n"
        f"Vote outcome: {'split (no clear consensus)' if is_split else 'clear winner'}, "
        f"confidence={confidence:.2f}\n"
        f"Dissent:\n{dissent_text}\n\n"
        "Write the reasoning_summary now."
    )
    return system_prompt, user_prompt


def _fallback_summary(winner_name: str, is_split: bool, dissent: list[DissentEntry]) -> str:
    parts = [f"{winner_name}'s answer won the deterministic vote on critic score and peer agreement."]
    if is_split:
        parts.append("The vote did not clear the 15% lead threshold, so this result is reported as a split.")
    if dissent:
        who = ", ".join(p for d in dissent for p in d.personas)
        parts.append(f"{who} maintained a differing answer through the final round.")
    return " ".join(parts)


def synthesize(
    question: str,
    final_proposals: dict[str, Proposal],
    final_critiques: dict[str, Critique],
    vote: VoteResult,
    rounds_used: int,
    client: CompletionClient,
    max_attempts: int = config.DEFAULT_MAX_ATTEMPTS,
) -> tuple[FinalVerdict, int]:
    """Build the FinalVerdict. Returns (verdict, tokens_used_by_this_call)."""
    winner_name = vote.winner.persona
    winner_proposal = final_proposals[winner_name]
    winner_critique = final_critiques[winner_name]

    confidence = compute_confidence(vote, rounds_used)
    dissent = compute_dissent(final_proposals, winner_name)
    is_split = not vote.is_clear_winner

    system_prompt, user_prompt = _build_summary_prompt(
        question, winner_name, winner_proposal, winner_critique, dissent, is_split, confidence
    )

    tokens_used = 0
    try:
        result = call_structured(client, system_prompt, user_prompt, AggregatorSynthesis, max_attempts=max_attempts)
        summary: AggregatorSynthesis = result.model  # type: ignore[assignment]
        reasoning_summary = summary.reasoning_summary
        tokens_used = result.total_tokens
    except SchemaEnforcementError as exc:
        logger.warning("aggregator synthesis call failed schema validation, using fallback summary: %s", exc)
        reasoning_summary = _fallback_summary(winner_name, is_split, dissent)

    verdict = FinalVerdict(
        answer=winner_proposal.answer,
        confidence=confidence,
        is_split=is_split,
        winning_personas=[winner_name],
        reasoning_summary=reasoning_summary,
        dissent=dissent,
        rounds_used=rounds_used,
    )
    return verdict, tokens_used
