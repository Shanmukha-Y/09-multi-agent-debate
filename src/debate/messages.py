"""Every inter-agent message is a validated Pydantic model — structured
swarms, not chat soup. A free-text LLM turn between debate stages is exactly
the failure mode project-05 taught us to avoid: errors compound silently
when one stage's prose becomes the next stage's unparsed input.

All three rubric dimensions on Critique.scores are 1-10 where **10 is
always best**, including `correctness_risk` (10 = very low risk of being
wrong, 1 = very high risk). This inverted-sounding name is kept because the
rubric document calls it "correctness risk," but the *scale direction* is
normalized so voting.py can sum the three dimensions directly — see
rubric/critic_rubric.md for the full definition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Stance = Literal["revise", "defend"]


class Proposal(BaseModel):
    """One proposer's answer to the question, round 1 or a rebuttal round."""

    answer: str = Field(..., min_length=1, description="The proposed answer, as concise as the question allows")
    reasoning: str = Field(..., min_length=1, description="Step-by-step justification")
    self_confidence: float = Field(..., ge=0.0, le=1.0, description="Proposer's own confidence, 0-1")

    @field_validator("answer", "reasoning")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v


class Rebuttal(Proposal):
    """A round-2 proposal: same shape as Proposal, plus an explicit stance
    on the round-1 critique and a note on what changed (if anything)."""

    stance: Stance = Field(..., description="'revise' if changing position, 'defend' if holding it")
    changes_summary: str = Field(
        ..., min_length=1, description="What changed vs round 1, or why nothing did"
    )


class CritiqueScores(BaseModel):
    correctness_risk: int = Field(..., ge=1, le=10, description="10 = very likely correct, 1 = very likely wrong")
    completeness: int = Field(..., ge=1, le=10, description="10 = fully addresses the question")
    reasoning_quality: int = Field(..., ge=1, le=10, description="10 = sound, well-justified reasoning")

    @property
    def total(self) -> int:
        return self.correctness_risk + self.completeness + self.reasoning_quality


class Critique(BaseModel):
    """The critic's assessment of ONE anonymized proposal (labeled A/B/C)."""

    proposal_id: str = Field(..., min_length=1, max_length=1, description="Anonymized label, e.g. 'A'")
    scores: CritiqueScores
    text: str = Field(..., min_length=1, description="Written critique explaining the scores")


class CriticOutput(BaseModel):
    """The critic scores every anonymized proposal in a debate round in one call."""

    critiques: list[Critique] = Field(..., min_length=1)


class DissentEntry(BaseModel):
    """One persisted disagreement, attributed to the persona(s) who held it."""

    personas: list[str] = Field(..., min_length=1)
    answer: str
    reasoning: str


class FinalVerdict(BaseModel):
    """The aggregator's output: a deterministic pick/split plus an
    LLM-written narrative summary. The `answer`, `confidence`, `is_split`,
    and `dissent` fields are computed in code (aggregator.py), never by the
    LLM — only `reasoning_summary` is model-generated prose."""

    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_split: bool
    winning_personas: list[str]
    reasoning_summary: str
    dissent: list[DissentEntry] = Field(default_factory=list)
    rounds_used: int = Field(..., ge=1, le=2)


class AggregatorSynthesis(BaseModel):
    """The one piece of the verdict the aggregator LLM call actually produces:
    a short prose explanation of why the (already-decided) answer won."""

    reasoning_summary: str = Field(..., min_length=1)
