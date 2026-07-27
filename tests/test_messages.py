"""Schema validation (Pydantic models reject bad shapes) and the
malformed-output repair loop in structured.py (a scripted fake client, no
network) that turns a bad first attempt into a valid second one.
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


# --- schema validation ------------------------------------------------


class TestProposal:
    def test_valid_proposal(self):
        p = Proposal(answer="0.05", reasoning="algebra", self_confidence=0.9)
        assert p.answer == "0.05"

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
        r = Rebuttal(answer="0.05", reasoning="revised", self_confidence=0.8, stance="revise", changes_summary="fixed arithmetic")
        assert r.stance == "revise"

    def test_invalid_stance_rejected(self):
        with pytest.raises(ValidationError):
            Rebuttal(
                answer="0.05", reasoning="x", self_confidence=0.8,
                stance="maybe", changes_summary="x",  # type: ignore[arg-type]
            )


class TestCritiqueScores:
    def test_valid_scores(self):
        s = CritiqueScores(correctness_risk=8, completeness=7, reasoning_quality=9)
        assert s.total == 24

    @pytest.mark.parametrize("field", ["correctness_risk", "completeness", "reasoning_quality"])
    def test_score_out_of_range_rejected(self, field):
        kwargs = {"correctness_risk": 5, "completeness": 5, "reasoning_quality": 5}
        kwargs[field] = 11
        with pytest.raises(ValidationError):
            CritiqueScores(**kwargs)

    def test_score_zero_rejected(self):
        with pytest.raises(ValidationError):
            CritiqueScores(correctness_risk=0, completeness=5, reasoning_quality=5)


class TestCritique:
    def test_multi_char_label_rejected(self):
        with pytest.raises(ValidationError):
            Critique(
                proposal_id="AB",
                scores=CritiqueScores(correctness_risk=5, completeness=5, reasoning_quality=5),
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
                answer="x", confidence=0.5, is_split=False, winning_personas=["A"],
                reasoning_summary="s", rounds_used=3,
            )

    def test_dissent_defaults_empty(self):
        v = FinalVerdict(
            answer="x", confidence=0.5, is_split=False, winning_personas=["A"],
            reasoning_summary="s", rounds_used=1,
        )
        assert v.dissent == []

    def test_dissent_entry_requires_persona(self):
        with pytest.raises(ValidationError):
            DissentEntry(personas=[], answer="x", reasoning="y")


# --- structured.py: validate/repair retry loop (scripted, no network) ---


class FakeClient:
    """Returns queued responses in order; records every call it received."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        self.calls.append((system_prompt, user_prompt))
        response = self._responses.pop(0)
        return response, 42


class TestCallStructured:
    def test_valid_first_attempt_succeeds_without_repair(self):
        client = FakeClient(['{"answer": "0.05", "reasoning": "r", "self_confidence": 0.9}'])
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.attempts == 1
        assert result.model.answer == "0.05"  # type: ignore[union-attr]
        assert len(client.calls) == 1

    def test_malformed_json_triggers_repair_then_succeeds(self):
        client = FakeClient(
            [
                'not json at all, sorry',
                '{"answer": "0.05", "reasoning": "r", "self_confidence": 0.9}',
            ]
        )
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.attempts == 2
        assert result.model.answer == "0.05"  # type: ignore[union-attr]
        assert len(client.calls) == 2
        # The repair prompt must include the earlier bad output so the model
        # can see what it needs to fix.
        assert "not json at all" in client.calls[1][1]

    def test_schema_violation_triggers_repair_then_succeeds(self):
        client = FakeClient(
            [
                '{"answer": "0.05", "reasoning": "r", "self_confidence": 5.0}',  # out of range
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
            ['Sure, here you go: {"answer": "0.05", "reasoning": "r", "self_confidence": 0.9} hope that helps!']
        )
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.model.answer == "0.05"  # type: ignore[union-attr]

    def test_total_tokens_summed_across_attempts(self):
        client = FakeClient(
            [
                'garbage',
                '{"answer": "0.05", "reasoning": "r", "self_confidence": 0.9}',
            ]
        )
        result = call_structured(client, "sys", "user", Proposal, max_attempts=2)
        assert result.total_tokens == 84  # 42 per call, 2 calls
