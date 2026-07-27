"""Orchestrates one full debate: fan-out proposals -> anonymized critique ->
rebuttal -> re-critique -> deterministic vote -> aggregation.

Calls are made strictly sequentially, never via asyncio.gather. This
machine's Ollama server is shared with other builders running their own
projects concurrently; a debate is already ~8-10 calls, and firing them in
parallel would be an unfriendly way to multiply that load. Each call *is*
independent of its round-mates, so parallelizing later (asyncio.gather over
the three proposers) is a one-line change if the server is ever dedicated.

Max debate rounds is a hard cap of 2 (config.MAX_DEBATE_ROUNDS): round 1 is
propose+critique, round 2 (if requested) is rebuttal+re-critique. There is
no adaptive early-exit on an already-unanimous round 1 — this keeps the
number of LLM calls, and therefore the token-cost story, a simple function
of `--rounds` rather than something that varies question-to-question in a
way the bench can't cleanly compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from debate import aggregator, config, critic, voting
from debate.client import OllamaClient
from debate.messages import Critique, FinalVerdict, Proposal, Rebuttal
from debate.personas import PROPOSER_PERSONAS, Persona
from debate.structured import CompletionClient, call_structured
from debate.voting import VoteResult

ClientFactory = Callable[[str, float], CompletionClient]

EventCallback = Callable[[str, dict], None]


def default_client_factory(role: str, temperature: float) -> CompletionClient:
    return OllamaClient(temperature=temperature)


def _emit(on_event: EventCallback | None, event_type: str, **kwargs) -> None:
    if on_event is not None:
        on_event(event_type, kwargs)


@dataclass(frozen=True)
class RoundData:
    round_num: int
    proposals: dict[str, Proposal]
    critiques: dict[str, Critique]
    persona_to_letter: dict[str, str]
    vote: VoteResult


@dataclass(frozen=True)
class DebateTranscript:
    question: str
    rounds: list[RoundData]
    verdict: FinalVerdict
    total_tokens: int
    rounds_used: int


def _propose(persona: Persona, question: str, client: CompletionClient, max_attempts: int) -> tuple[Proposal, int]:
    user_prompt = f"Question: {question}\n\nProvide your proposal now."
    result = call_structured(client, persona.system_prompt, user_prompt, Proposal, max_attempts=max_attempts)
    return result.model, result.total_tokens  # type: ignore[return-value]


def _rebuttal_system_prompt(persona: Persona) -> str:
    return (
        f"{persona.system_prompt}\n\n"
        "You are now in the rebuttal round. You will see your own original "
        "proposal, the critic's critique of it, and your peers' (anonymized) "
        "proposals and critiques. Decide whether to REVISE your answer (the "
        "critique or a peer's reasoning changed your mind) or DEFEND it "
        "explicitly (you considered the critique and still believe your "
        "original answer). Stay in your reasoning-procedure role while doing "
        "so.\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{"answer": "<final answer after this round>", "reasoning": '
        '"<updated reasoning>", "self_confidence": <0.0-1.0>, '
        '"stance": "revise" | "defend", "changes_summary": '
        '"<what changed vs your original proposal, or why nothing did>"}. '
        "No text outside the JSON object."
    )


def _build_peer_block(
    self_name: str,
    proposals: dict[str, Proposal],
    critiques: dict[str, Critique],
    letters: dict[str, str],
) -> str:
    blocks = []
    for name, proposal in proposals.items():
        if name == self_name:
            continue
        c = critiques[name]
        blocks.append(
            f"Proposal {letters[name]}:\n"
            f"  answer: {proposal.answer}\n"
            f"  reasoning: {proposal.reasoning}\n"
            f"  critic scores: correctness_risk={c.scores.correctness_risk}, "
            f"completeness={c.scores.completeness}, reasoning_quality={c.scores.reasoning_quality}\n"
            f"  critic critique: {c.text}"
        )
    return "\n\n".join(blocks)


def _rebut(
    persona: Persona,
    question: str,
    own_proposal: Proposal,
    own_critique: Critique,
    peer_block: str,
    client: CompletionClient,
    max_attempts: int,
) -> tuple[Rebuttal, int]:
    system_prompt = _rebuttal_system_prompt(persona)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Your original proposal:\n  answer: {own_proposal.answer}\n  reasoning: {own_proposal.reasoning}\n\n"
        f"Critic's critique of your proposal: {own_critique.text} "
        f"(correctness_risk={own_critique.scores.correctness_risk}, "
        f"completeness={own_critique.scores.completeness}, "
        f"reasoning_quality={own_critique.scores.reasoning_quality})\n\n"
        f"Peer proposals and critiques:\n{peer_block}\n\n"
        "Revise or defend now."
    )
    result = call_structured(client, system_prompt, user_prompt, Rebuttal, max_attempts=max_attempts)
    return result.model, result.total_tokens  # type: ignore[return-value]


def run_debate(
    question: str,
    max_rounds: int = config.MAX_DEBATE_ROUNDS,
    client_factory: ClientFactory | None = None,
    max_attempts: int = config.DEFAULT_MAX_ATTEMPTS,
    on_event: EventCallback | None = None,
) -> DebateTranscript:
    """Run one full debate. `client_factory(role_name, temperature) ->
    CompletionClient` defaults to real Ollama calls; tests inject a fake."""
    if not (1 <= max_rounds <= config.MAX_DEBATE_ROUNDS):
        raise ValueError(f"max_rounds must be between 1 and {config.MAX_DEBATE_ROUNDS}, got {max_rounds}")

    client_factory = client_factory or default_client_factory
    total_tokens = 0
    rounds: list[RoundData] = []

    # --- Round 1: propose ---
    proposals_r1: dict[str, Proposal] = {}
    for persona in PROPOSER_PERSONAS:
        client = client_factory(persona.name, persona.temperature)
        proposal, tokens = _propose(persona, question, client, max_attempts)
        total_tokens += tokens
        proposals_r1[persona.name] = proposal
        _emit(on_event, "proposal", round=1, persona=persona.name, proposal=proposal)

    critic_client = client_factory("Critic", critic.CRITIC_TEMPERATURE)
    critiques_r1, letters_r1, tokens = critic.critique_round(question, proposals_r1, critic_client, max_attempts)
    total_tokens += tokens
    _emit(on_event, "critique", round=1, critiques=critiques_r1, letters=letters_r1)

    vote1 = voting.score_round({n: p.answer for n, p in proposals_r1.items()}, critiques_r1)
    _emit(on_event, "vote", round=1, vote=vote1)

    rounds.append(RoundData(1, dict(proposals_r1), critiques_r1, letters_r1, vote1))

    if max_rounds >= 2:
        # --- Round 2: rebuttal ---
        proposals_r2: dict[str, Proposal] = {}
        for persona in PROPOSER_PERSONAS:
            peer_block = _build_peer_block(persona.name, proposals_r1, critiques_r1, letters_r1)
            client = client_factory(persona.name, persona.temperature)
            rebuttal, tokens = _rebut(
                persona, question, proposals_r1[persona.name], critiques_r1[persona.name], peer_block, client, max_attempts
            )
            total_tokens += tokens
            proposals_r2[persona.name] = rebuttal
            _emit(on_event, "rebuttal", round=2, persona=persona.name, proposal=rebuttal)

        critic_client2 = client_factory("Critic", critic.CRITIC_TEMPERATURE)
        critiques_r2, letters_r2, tokens = critic.critique_round(question, proposals_r2, critic_client2, max_attempts)
        total_tokens += tokens
        _emit(on_event, "critique", round=2, critiques=critiques_r2, letters=letters_r2)

        vote2 = voting.score_round({n: p.answer for n, p in proposals_r2.items()}, critiques_r2)
        _emit(on_event, "vote", round=2, vote=vote2)

        rounds.append(RoundData(2, proposals_r2, critiques_r2, letters_r2, vote2))
        final_proposals, final_critiques, final_vote, rounds_used = proposals_r2, critiques_r2, vote2, 2
    else:
        final_proposals, final_critiques, final_vote, rounds_used = proposals_r1, critiques_r1, vote1, 1

    agg_client = client_factory("Aggregator", aggregator.AGGREGATOR_TEMPERATURE)
    verdict, tokens = aggregator.synthesize(
        question, final_proposals, final_critiques, final_vote, rounds_used, agg_client, max_attempts
    )
    total_tokens += tokens
    _emit(on_event, "verdict", verdict=verdict)

    return DebateTranscript(
        question=question,
        rounds=rounds,
        verdict=verdict,
        total_tokens=total_tokens,
        rounds_used=rounds_used,
    )
