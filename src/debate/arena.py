"""Baseline comparators the bench uses to answer "is debate worth 7x the
tokens?": single-shot (1 call) and self-consistency (3 samples + majority,
the cheap alternative debate has to beat), alongside the full debate arm.
"""

from __future__ import annotations

from dataclasses import dataclass

from debate import config
from debate.debate import ClientFactory, DebateTranscript, default_client_factory, run_debate
from debate.messages import Proposal
from debate.personas import ANALYST
from debate.structured import call_structured
from debate.voting import cluster_answers

SELF_CONSISTENCY_SAMPLES = 3
SELF_CONSISTENCY_TEMPERATURE = 0.7  # higher than Analyst's default 0.3: sampling diversity is the point


@dataclass(frozen=True)
class ArenaResult:
    arm: str
    answer: str
    confidence: float
    total_tokens: int
    calls_made: int


def run_single_shot(
    question: str,
    client_factory: ClientFactory | None = None,
    max_attempts: int = config.DEFAULT_MAX_ATTEMPTS,
) -> ArenaResult:
    """One Analyst call, no critic, no vote — the cheapest possible baseline."""
    client_factory = client_factory or default_client_factory
    client = client_factory(ANALYST.name, ANALYST.temperature)
    user_prompt = f"Question: {question}\n\nProvide your proposal now."
    result = call_structured(client, ANALYST.system_prompt, user_prompt, Proposal, max_attempts=max_attempts)
    proposal: Proposal = result.model  # type: ignore[assignment]
    return ArenaResult(
        arm="single_shot",
        answer=proposal.answer,
        confidence=proposal.self_confidence,
        total_tokens=result.total_tokens,
        calls_made=1,
    )


def run_self_consistency(
    question: str,
    samples: int = SELF_CONSISTENCY_SAMPLES,
    client_factory: ClientFactory | None = None,
    max_attempts: int = config.DEFAULT_MAX_ATTEMPTS,
) -> ArenaResult:
    """N independent Analyst samples at a higher temperature, majority vote
    on the answer (via voting.cluster_answers — same agreement definition
    the debate's own vote and dissent logic use)."""
    client_factory = client_factory or default_client_factory
    total_tokens = 0
    answers: list[str] = []

    for _ in range(samples):
        client = client_factory(ANALYST.name, SELF_CONSISTENCY_TEMPERATURE)
        user_prompt = f"Question: {question}\n\nProvide your proposal now."
        result = call_structured(client, ANALYST.system_prompt, user_prompt, Proposal, max_attempts=max_attempts)
        proposal: Proposal = result.model  # type: ignore[assignment]
        total_tokens += result.total_tokens
        answers.append(proposal.answer)

    clusters = cluster_answers(answers)
    majority = max(clusters, key=len)
    majority_answer = answers[majority[0]]
    majority_fraction = len(majority) / len(answers)

    return ArenaResult(
        arm="self_consistency",
        answer=majority_answer,
        confidence=majority_fraction,
        total_tokens=total_tokens,
        calls_made=samples,
    )


def run_debate_arm(
    question: str,
    max_rounds: int = config.MAX_DEBATE_ROUNDS,
    client_factory: ClientFactory | None = None,
    max_attempts: int = config.DEFAULT_MAX_ATTEMPTS,
) -> tuple[ArenaResult, DebateTranscript]:
    transcript = run_debate(question, max_rounds=max_rounds, client_factory=client_factory, max_attempts=max_attempts)
    calls = 3 + 1 + (3 + 1 if max_rounds >= 2 else 0) + 1  # propose + critic (+ rebuttal + critic) + aggregator
    result = ArenaResult(
        arm="debate",
        answer=transcript.verdict.answer,
        confidence=transcript.verdict.confidence,
        total_tokens=transcript.total_tokens,
        calls_made=calls,
    )
    return result, transcript
