"""Validated message contracts for every debate stage.

Free-form prose passed directly between agents compounds errors silently, so
proposals, critiques, rebuttals, and final verdicts all cross typed Pydantic
boundaries.

All three critic dimensions use a 1-10 scale where 10 is best, including
``correctness_risk`` (10 means low risk of being wrong). The rubric keeps the
historical name while normalizing scale direction so deterministic voting can
sum the dimensions directly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Stance = Literal["revise", "defend"]


class Proposal(BaseModel):
    """One proposer's answer, used for both initial and rebuttal rounds."""

    answer: str = Field(
        ...,
        min_length=1,
        description="The proposed answer, as concise as the question allows",
    )
    reasoning: str = Field(..., min_length=1, description="Step-by-step justification")
    self_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proposer's own confidence, 0-1",
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def _normalize_reasoning_steps(cls, value: Any) -> Any:
        """Accept a common JSON-shape variation without discarding content.

        Local models sometimes emit ``reasoning`` as an array of step strings
        even when the requested schema says string. Joining a non-empty list of
        strings is lossless for this field and avoids spending a second model
        call on a purely representational mismatch. Other types still fail.
        """
        if not isinstance(value, list):
            return value
        if not value or not all(isinstance(step, str) for step in value):
            raise ValueError("reasoning list must contain one or more strings")
        normalized = [step.strip() for step in value if step.strip()]
        if not normalized:
            raise ValueError("reasoning list must contain non-blank strings")
        return "\n".join(normalized)

    @field_validator("answer", "reasoning")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class Rebuttal(Proposal):
    """A round-two proposal plus an explicit response to prior criticism."""

    stance: Stance = Field(
        ...,
        description="'revise' if changing position, 'defend' if holding it",
    )
    changes_summary: str = Field(
        ...,
        min_length=1,
        description="What changed versus round one, or why nothing did",
    )


class CritiqueScores(BaseModel):
    correctness_risk: int = Field(
        ...,
        ge=1,
        le=10,
        description="10 = very likely correct, 1 = very likely wrong",
    )
    completeness: int = Field(..., ge=1, le=10, description="10 = fully addresses the question")
    reasoning_quality: int = Field(..., ge=1, le=10, description="10 = sound, well-justified reasoning")

    @property
    def total(self) -> int:
        return self.correctness_risk + self.completeness + self.reasoning_quality


class Critique(BaseModel):
    """The critic's assessment of one anonymized proposal."""

    proposal_id: str = Field(
        ...,
        min_length=1,
        max_length=1,
        description="Anonymized label, for example 'A'",
    )
    scores: CritiqueScores
    text: str = Field(..., min_length=1, description="Written critique explaining the scores")


class CriticOutput(BaseModel):
    """Critiques for every anonymized proposal in a round."""

    critiques: list[Critique] = Field(..., min_length=1)


class DissentEntry(BaseModel):
    """One persisted disagreement, attributed to the persona(s) holding it."""

    personas: list[str] = Field(..., min_length=1)
    answer: str
    reasoning: str


class FinalVerdict(BaseModel):
    """A deterministic verdict plus the aggregator's prose summary.

    Answer, confidence, split state, winners, and dissent are computed in code.
    Only ``reasoning_summary`` is model-generated.
    """

    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_split: bool
    winning_personas: list[str]
    reasoning_summary: str
    dissent: list[DissentEntry] = Field(default_factory=list)
    rounds_used: int = Field(..., ge=1, le=2)


class AggregatorSynthesis(BaseModel):
    """The only prose field produced by the aggregator model call."""

    reasoning_summary: str = Field(..., min_length=1)
