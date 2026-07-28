"""Message-schema validation and the structured-output repair loop.

All tests use scripted clients and make no network calls.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from debate.messages import (
    Critique,
    CriticOutput,
    CritiqueScores,
    DissentEntry,
    FinalVerdict,
    Proposal,
    Rebuttal,
)
from debate.structured import SchemaEnforcementError, call_structured


class TestProposal:
    def test_valid_proposal(self):
        proposal = Proposal(answer="0.05", reasoning="algebra", self_confidence=0.9)
        assert proposal.answer == "0.05"

    def test_reasoning_list_is_normalized_losslessly(self):
        proposal = Proposal(
            answer="0.05",
            reasoning=["Set up x + (x + 1) = 1.10", "Solve 2x = 0.10"],  # type: ignore[arg-type]
            self_confidence=0.9,
        )
        assert proposal.reasoning == "Set up x + (x + 1) = 1.10\nSolve 2x = 0.10"

    @pytest.mark.parametrize(
        "reasoning",
        [[], ["valid", 3], ["   "]],
    )
    def test_invalid_reasoning_lists_are_rejected(self, reasoning):
        with pytest.raises(ValidationError):
            Proposal(answer="x", reasoning=reasoning, self_confidence=0.5)

    def test_blank_answer_rejected(self):
        with pytest.raises(ValidationError):
            Proposal(answer="   ", reasoning="x", self_confidence=0.5)

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Proposal(answer="x", reasoning="y", self_confidence=1.5)

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            Proposal(answer="x", self_confidence=0.5)  # type: ignore[call-arg]


class TestRebuttal:
    def test_valid_rebuttal(self):
        rebuttal = Rebuttal(
            answer="0.05",
            reasoning="revised",
            self_confidence=0.8,
            stance="revise",
            changes_summary="fixed arithmetic",
        )
        assert rebuttal.stance == "revise"

    def test_invalid_stance_rejected(self):
        with pytest.raises(ValidationError):
            Rebuttal(
                answer="0.05",
                reasoning="x",
                self_confidence=0.8,
                stance="maybe",  # type: ignore[arg-type]
                changes_summary="x",
            )


class TestCritiqueScores:
    def test_valid_scores(self):
        scores = CritiqueScores(
            correctness_risk=8,
            completeness=7,
            reasoning_quality=9,
        )
        assert scores.total == 24

    @pytest.mark.parametrize(
        "field",
        ["correctness_risk", "completeness", "reasoning_quality"],
    )
    def test_score_out_of_range_rejected(self, field):
        kwargs = {
            "correctness_risk": 5,
            "completeness": 5,
            "reasoning_quality": 5,
        }
        kwargs[field] = 11
        with pytest.raises(ValidationError):
            CritiqueScores(**kwargs)

    def test_score_zero_rejected(self):
        with pytest.raises(ValidationError):
            CritiqueScores(
                correctness_risk=0,
                completeness=5,
                reasoning_quality=5,
            )


class TestCritique:
    def test_multi_char_label_rejected(self):
        with pytest.raises(ValidationError):
            Critique(
                proposal_id="AB",
                scores=CritiqueScores(
                    correctness_risk=5,
                    completeness=5,
                    reasoning_quality=5,
                ),
                text="x",
            )


class TestCriticOutput:
    def test_requires_at_least_one_critique(self):
        with pytest.raises(ValidationError):
            CriticOutput(critiques=[])


class TestFinalVerdict:
    def test_rounds_used_bounded(self):
        with pytest.raises(ValidationError):
            FinalVerdict(
                answer="x",
                confidence=0.5,
                is_split=False,
                winning_personas=["A"],
                reasoning_summary="s",
                rounds_used=3,
            )

    def test_dissent_defaults_empty(self):
        verdict = FinalVerdict(
            answer="x",
            confidence=0.5,
            is_split=False,
            winning_personas=["A"],
            reasoning_summary="s",
            rounds_used=1,
        )
        assert verdict.dissent == []

    def test_dissent_entry_requires_persona(self):
        with pytest.raises(ValidationError):
            DissentEntry(personas=[], answer="x", reasoning="y")


class FakeClient:
    """Return queued responses in order and record every call."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        self.calls.append((system_prompt, user_prompt))
        response = self._responses.pop(0)
        return response, 42


class TestCallStructured:
    def test_valid_first_attempt_succeeds_without_repair(self):
        client = FakeClient(
            ['{"answer": "0.05", "reasoning": "r", "self_confidence": 0.9}']
        )
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.attempts == 1
        assert result.model.answer == "0.05"  # type: ignore[union-attr]
        assert len(client.calls) == 1

    def test_reasoning_array_is_normalized_without_repair_call(self):
        client = FakeClient(
            [
                '{"answer": "0.05", "reasoning": ["set up equation", "solve it"], '
                '"self_confidence": 0.9}'
            ]
        )
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.attempts == 1
        assert result.model.reasoning == "set up equation\nsolve it"  # type: ignore[union-attr]
        assert len(client.calls) == 1

    def test_malformed_json_triggers_repair_then_succeeds(self):
        client = FakeClient(
            [
                "not json at all, sorry",
                '{"answer": "0.05", "reasoning": "r", "self_confidence": 0.9}',
            ]
        )
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.attempts == 2
        assert result.model.answer == "0.05"  # type: ignore[union-attr]
        assert len(client.calls) == 2
        assert "not json at all" in client.calls[1][1]

    def test_schema_violation_triggers_repair_then_succeeds(self):
        client = FakeClient(
            [
                '{"answer": "0.05", "reasoning": "r", "self_confidence": 5.0}',
                '{"answer": "0.05", "reasoning": "r", "self_confidence": 0.9}',
            ]
        )
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.attempts == 2
        assert result.model.self_confidence == 0.9  # type: ignore[union-attr]

    def test_exhausted_attempts_raises(self):
        client = FakeClient(["garbage", "still garbage"])
        with pytest.raises(SchemaEnforcementError) as exc_info:
            call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert exc_info.value.attempts == 2
        assert len(client.calls) == 2

    def test_json_embedded_in_prose_is_extracted(self):
        client = FakeClient(
            [
                'Sure: {"answer": "0.05", "reasoning": "r", '
                '"self_confidence": 0.9} hope that helps!'
            ]
        )
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.model.answer == "0.05"  # type: ignore[union-attr]

    def test_total_tokens_summed_across_attempts(self):
        client = FakeClient(
            [
                "garbage",
                '{"answer": "0.05", "reasoning": "r", "self_confidence": 0.9}',
            ]
        )
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.total_tokens == 84
