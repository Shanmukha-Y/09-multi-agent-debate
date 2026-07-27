"""Full debate flow driven by scripted agent outputs — no network, no LLM.

Two scenarios the spec calls out explicitly:
1. Skeptic catches a planted flaw in round 1 (Analyst gives the classic
   bat-and-ball wrong answer); by the rebuttal round Analyst self-corrects
   and the debate converges unanimously on the right answer.
2. A proposer persists in disagreement through the rebuttal round: the
   final verdict must carry a non-empty dissent appendix and a LOWER
   confidence than the unanimous scenario above — persistent disagreement
   is never allowed to silently vanish into an averaged answer.

Anonymization is deterministic in this codebase (letters assigned in
persona iteration order: Analyst=A, Skeptic=B, Creative=C, every round),
which is what makes it possible to script critic responses by letter here.
"""

from __future__ import annotations

import json

import pytest

from debate.debate import run_debate


class ScriptedClient:
    """Pops queued raw-text responses in call order; records every call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        self.calls.append((system_prompt, user_prompt))
        if not self._responses:
            raise AssertionError("ScriptedClient ran out of queued responses")
        return self._responses.pop(0), 100


def make_factory(scripts: dict[str, list[str]]):
    """role_name -> ordered list of raw responses for that role's successive
    calls (e.g. Analyst's [round1 propose, round2 rebuttal]). The same
    ScriptedClient instance is reused across calls to the same role so its
    queue is consumed in call order."""
    clients: dict[str, ScriptedClient] = {}

    def factory(role: str, temperature: float):
        if role not in clients:
            clients[role] = ScriptedClient(scripts[role])
        return clients[role]

    return factory, clients


def proposal_json(answer: str, reasoning: str, confidence: float) -> str:
    return json.dumps({"answer": answer, "reasoning": reasoning, "self_confidence": confidence})


def rebuttal_json(answer: str, reasoning: str, confidence: float, stance: str, changes: str) -> str:
    return json.dumps(
        {
            "answer": answer,
            "reasoning": reasoning,
            "self_confidence": confidence,
            "stance": stance,
            "changes_summary": changes,
        }
    )


def critique_json(scores_by_label: dict[str, tuple[int, int, int, str]]) -> str:
    return json.dumps(
        {
            "critiques": [
                {
                    "proposal_id": label,
                    "scores": {"correctness_risk": r, "completeness": c, "reasoning_quality": q},
                    "text": text,
                }
                for label, (r, c, q, text) in scores_by_label.items()
            ]
        }
    )


def aggregator_json(summary: str) -> str:
    return json.dumps({"reasoning_summary": summary})


QUESTION = (
    "A bat and ball cost $1.10 total; the bat costs $1 more than the ball. "
    "What does the ball cost?"
)


class TestErrorCorrectionFlipsAnswer:
    def test_skeptic_catches_flaw_and_analyst_self_corrects(self):
        scripts = {
            "Analyst": [
                proposal_json("$0.10", "bat+ball=1.10, bat=1 more, so ball=0.10", 0.7),
                rebuttal_json(
                    "$0.05", "critic caught my error: 2x+1=1.10 means x=0.05, not 0.10",
                    0.9, "revise", "corrected the algebra after the critique",
                ),
            ],
            "Skeptic": [
                proposal_json("$0.05", "let ball=x, bat=x+1, 2x+1=1.10, x=0.05", 0.9),
                rebuttal_json("$0.05", "still correct, defending", 0.9, "defend", "no change needed"),
            ],
            "Creative": [
                proposal_json("$0.12", "rough intuition, bat is much pricier", 0.4),
                rebuttal_json(
                    "$0.05", "the algebraic framing in the critique is more reliable than my intuition",
                    0.85, "revise", "adopted the algebraic answer",
                ),
            ],
            "Critic": [
                critique_json(
                    {
                        "A": (3, 6, 5, "Analyst's answer fails to satisfy 'bat costs $1 more' when checked"),
                        "B": (9, 8, 9, "Skeptic's algebra is correct and verified"),
                        "C": (2, 5, 4, "Creative's answer is an unverified guess"),
                    }
                ),
                critique_json(
                    {
                        "A": (8, 8, 8, "Analyst correctly revised to the algebraic answer"),
                        "B": (9, 9, 9, "Skeptic's answer remains correct and well-justified"),
                        "C": (8, 8, 8, "Creative adopted the correct algebraic answer"),
                    }
                ),
            ],
            "Aggregator": [aggregator_json("The debate converged on $0.05 after the critique caught the intuitive error.")],
        }
        factory, clients = make_factory(scripts)

        transcript = run_debate(QUESTION, max_rounds=2, client_factory=factory)

        # Round 1: Analyst's own recorded proposal is the WRONG answer.
        assert transcript.rounds[0].proposals["Analyst"].answer == "$0.10"
        # Round 2: Analyst explicitly revised, and now agrees with the group.
        round2_analyst = transcript.rounds[1].proposals["Analyst"]
        assert round2_analyst.answer == "$0.05"
        assert round2_analyst.stance == "revise"

        # The final verdict reflects the corrected, unanimous answer.
        assert transcript.verdict.answer == "$0.05"
        assert transcript.verdict.is_split is False
        assert transcript.verdict.dissent == []
        assert transcript.rounds_used == 2

        # Every scripted response was actually consumed (proves both rounds ran).
        for client in clients.values():
            assert client._responses == []


class TestPersistentDisagreementProducesDissent:
    def test_holdout_persona_produces_dissent_and_lowers_confidence(self):
        scripts = {
            "Analyst": [
                proposal_json("$0.10", "bat+ball=1.10, bat=1 more, so ball=0.10", 0.7),
                rebuttal_json(
                    "$0.10", "I considered the critique but I still believe 0.10 is right",
                    0.6, "defend", "no change, holding position",
                ),
            ],
            "Skeptic": [
                proposal_json("$0.05", "let ball=x, bat=x+1, 2x+1=1.10, x=0.05", 0.9),
                rebuttal_json("$0.05", "still correct, defending", 0.9, "defend", "no change needed"),
            ],
            "Creative": [
                proposal_json("$0.05", "double-checked via the algebraic framing, matches Skeptic", 0.8),
                rebuttal_json("$0.05", "confirmed again, defending", 0.8, "defend", "no change needed"),
            ],
            "Critic": [
                critique_json(
                    {
                        "A": (3, 6, 5, "fails the constraint check"),
                        "B": (9, 8, 9, "correct and verified"),
                        "C": (8, 8, 7, "correct, matches Skeptic"),
                    }
                ),
                critique_json(
                    {
                        "A": (3, 6, 5, "still fails the constraint check after rebuttal"),
                        "B": (9, 8, 9, "remains correct and well-justified"),
                        "C": (8, 8, 8, "remains correct"),
                    }
                ),
            ],
            "Aggregator": [aggregator_json("Skeptic and Creative converged on $0.05; Analyst held out.")],
        }
        factory, _ = make_factory(scripts)

        transcript = run_debate(QUESTION, max_rounds=2, client_factory=factory)

        assert transcript.verdict.answer == "$0.05"
        assert transcript.verdict.is_split is False  # winner still clears the 15% lead despite the holdout
        assert len(transcript.verdict.dissent) == 1
        assert transcript.verdict.dissent[0].personas == ["Analyst"]
        assert transcript.verdict.dissent[0].answer == "$0.10"


class TestDissentLowersConfidenceVsUnanimous:
    def test_confidence_with_dissent_below_confidence_without(self):
        unanimous_scripts = {
            "Analyst": [
                proposal_json("$0.10", "x", 0.7),
                rebuttal_json("$0.05", "revised", 0.9, "revise", "fixed"),
            ],
            "Skeptic": [
                proposal_json("$0.05", "x", 0.9),
                rebuttal_json("$0.05", "defended", 0.9, "defend", "none"),
            ],
            "Creative": [
                proposal_json("$0.12", "x", 0.4),
                rebuttal_json("$0.05", "revised", 0.85, "revise", "fixed"),
            ],
            "Critic": [
                critique_json({"A": (3, 6, 5, "x"), "B": (9, 8, 9, "x"), "C": (2, 5, 4, "x")}),
                critique_json({"A": (8, 8, 8, "x"), "B": (9, 9, 9, "x"), "C": (8, 8, 8, "x")}),
            ],
            "Aggregator": [aggregator_json("converged")],
        }
        holdout_scripts = {
            "Analyst": [
                proposal_json("$0.10", "x", 0.7),
                rebuttal_json("$0.10", "held", 0.6, "defend", "none"),
            ],
            "Skeptic": [
                proposal_json("$0.05", "x", 0.9),
                rebuttal_json("$0.05", "defended", 0.9, "defend", "none"),
            ],
            "Creative": [
                proposal_json("$0.05", "x", 0.8),
                rebuttal_json("$0.05", "defended", 0.8, "defend", "none"),
            ],
            "Critic": [
                critique_json({"A": (3, 6, 5, "x"), "B": (9, 8, 9, "x"), "C": (8, 8, 7, "x")}),
                critique_json({"A": (3, 6, 5, "x"), "B": (9, 8, 9, "x"), "C": (8, 8, 8, "x")}),
            ],
            "Aggregator": [aggregator_json("holdout")],
        }

        unanimous_factory, _ = make_factory(unanimous_scripts)
        holdout_factory, _ = make_factory(holdout_scripts)

        unanimous_transcript = run_debate(QUESTION, max_rounds=2, client_factory=unanimous_factory)
        holdout_transcript = run_debate(QUESTION, max_rounds=2, client_factory=holdout_factory)

        assert unanimous_transcript.verdict.dissent == []
        assert len(holdout_transcript.verdict.dissent) == 1
        assert holdout_transcript.verdict.confidence < unanimous_transcript.verdict.confidence


class TestSingleRoundDebate:
    def test_rounds_1_skips_rebuttal_entirely(self):
        scripts = {
            "Analyst": [proposal_json("$0.05", "x", 0.9)],
            "Skeptic": [proposal_json("$0.05", "x", 0.9)],
            "Creative": [proposal_json("$0.05", "x", 0.9)],
            "Critic": [critique_json({"A": (8, 8, 8, "x"), "B": (9, 9, 9, "x"), "C": (8, 8, 8, "x")})],
            "Aggregator": [aggregator_json("single round consensus")],
        }
        factory, clients = make_factory(scripts)

        transcript = run_debate(QUESTION, max_rounds=1, client_factory=factory)

        assert transcript.rounds_used == 1
        assert len(transcript.rounds) == 1
        assert transcript.verdict.answer == "$0.05"
        # Only one call per proposer, one critic call, one aggregator call.
        assert clients["Analyst"].calls.__len__() == 1
        assert clients["Critic"].calls.__len__() == 1


class TestInvalidRoundsRejected:
    @pytest.mark.parametrize("bad_rounds", [0, 3, -1])
    def test_out_of_range_rounds_raises(self, bad_rounds):
        with pytest.raises(ValueError):
            run_debate(QUESTION, max_rounds=bad_rounds, client_factory=lambda r, t: ScriptedClient([]))
