"""Deterministic vote math — no LLM in the vote itself, which is the whole
point: the winner is auditable by re-running arithmetic over critic scores,
not by trusting another model's judgment call.

score(persona) = critic_total(persona) * agreement_multiplier(persona)

critic_total is the sum of the three 1-10 rubric dimensions (range 3-30,
higher is better — see rubric/critic_rubric.md). agreement_multiplier
rewards a proposal for having its answer independently corroborated by
other proposers: 1.0 with no corroboration, up to
1 + AGREEMENT_BONUS_WEIGHT with unanimous corroboration.

A "clear winner" is the top score leading its nearest *disagreeing*
competitor by more than CLEAR_WINNER_LEAD_THRESHOLD (15%) of the top score:
    (top - nearest_dissenting_score) / top > threshold
"Disagreeing" matters: if the top-scoring proposals are tied because every
proposer converged on the same answer, that is unanimous consensus, not a
split, even though their scores are literally equal — a tie in score only
signals a split when it is a tie between two *different* answers. If no
other proposer's answer disagrees with the winner's, the round is
automatically a clear winner regardless of score spread. Anything else at
or below the threshold is a "split" — the aggregator must report dissent
rather than silently picking a marginal winner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from debate import config
from debate.messages import Critique

_NUMERIC_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(answer: str) -> float | None:
    """Pull the first numeric value out of an answer string, if any.
    Handles simple currency/formatting like "$0.05" or "1,200 tuners".
    Public because bench/run_bench.py grades numeric_range questions with
    the exact same extraction rule the vote itself uses for agreement."""
    match = _NUMERIC_RE.search(answer.replace(",", ""))
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _normalize_text(answer: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", answer.lower()).strip()


def answers_agree(a: str, b: str, tolerance: float = config.NUMERIC_AGREEMENT_TOLERANCE) -> bool:
    """Two answers "agree" if both parse to numbers within relative
    `tolerance`, or otherwise if their normalized text is identical.
    This is intentionally conservative (no embeddings/fuzzy NLP): a false
    "agree" would inflate a wrong answer's score, so ties go to disagreement.
    """
    num_a, num_b = extract_number(a), extract_number(b)
    if num_a is not None and num_b is not None:
        if num_a == 0 and num_b == 0:
            return True
        denom = max(abs(num_a), abs(num_b), 1e-9)
        return abs(num_a - num_b) / denom <= tolerance
    return _normalize_text(a) == _normalize_text(b)


def cluster_answers(answers: list[str]) -> list[list[int]]:
    """Group answer indices into agreement clusters (pairwise, via
    answers_agree, transitively chained through each cluster's first
    member). Shared by the debate's dissent grouping and the
    self-consistency baseline's majority vote so "what counts as the same
    answer" is defined in exactly one place."""
    groups: list[list[int]] = []
    for i, answer in enumerate(answers):
        for group in groups:
            if answers_agree(answer, answers[group[0]]):
                group.append(i)
                break
        else:
            groups.append([i])
    return groups


@dataclass(frozen=True)
class PersonaScore:
    persona: str
    critic_total: int
    agreement_multiplier: float
    agreeing_peers: int
    total_peers: int
    final_score: float


@dataclass(frozen=True)
class VoteResult:
    scores: list[PersonaScore]  # sorted descending by final_score
    is_clear_winner: bool
    lead_ratio: float
    winner: PersonaScore
    runner_up: PersonaScore | None

    @property
    def winners(self) -> list[PersonaScore]:
        """All personas tied for the top score (usually just one)."""
        top = self.scores[0].final_score
        return [s for s in self.scores if s.final_score == top]


def score_round(
    proposals: dict[str, str],  # persona_name -> answer text
    critiques: dict[str, Critique],  # persona_name -> Critique
) -> VoteResult:
    if set(proposals) != set(critiques):
        raise ValueError(
            f"proposals and critiques must cover the same personas: "
            f"{set(proposals)} vs {set(critiques)}"
        )
    if not proposals:
        raise ValueError("score_round requires at least one proposal")

    persona_scores: list[PersonaScore] = []
    for name, answer in proposals.items():
        others = [other_answer for other_name, other_answer in proposals.items() if other_name != name]
        agreeing = sum(1 for other in others if answers_agree(answer, other))
        agreement_fraction = agreeing / len(others) if others else 0.0
        multiplier = 1.0 + config.AGREEMENT_BONUS_WEIGHT * agreement_fraction

        critic_total = critiques[name].scores.total
        final = critic_total * multiplier

        persona_scores.append(
            PersonaScore(
                persona=name,
                critic_total=critic_total,
                agreement_multiplier=multiplier,
                agreeing_peers=agreeing,
                total_peers=len(others),
                final_score=final,
            )
        )

    # Sort by score desc; break ties deterministically by persona name so
    # results are reproducible given identical inputs.
    persona_scores.sort(key=lambda s: (-s.final_score, s.persona))

    top = persona_scores[0]
    runner_up = persona_scores[1] if len(persona_scores) > 1 else None

    dissenting_competitors = [
        s for s in persona_scores[1:] if not answers_agree(proposals[s.persona], proposals[top.persona])
    ]
    if not dissenting_competitors:
        lead_ratio = 1.0
        is_clear = True
    else:
        nearest = dissenting_competitors[0]  # persona_scores is already sorted desc
        lead_ratio = (top.final_score - nearest.final_score) / top.final_score if top.final_score > 0 else 0.0
        is_clear = lead_ratio > config.CLEAR_WINNER_LEAD_THRESHOLD

    return VoteResult(
        scores=persona_scores,
        is_clear_winner=is_clear,
        lead_ratio=lead_ratio,
        winner=top,
        runner_up=runner_up,
    )
