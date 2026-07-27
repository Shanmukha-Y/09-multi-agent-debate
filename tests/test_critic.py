"""Direct unit tests for critic.py's anonymization and error handling —
paths that the full mocked debate flow (test_debate_mocked.py) only
exercises on the happy path. No network.
"""

from __future__ import annotations

import json

import pytest

from debate.critic import anonymize, critique_round, load_rubric_text
from debate.messages import Proposal


class FakeClient:
    def __init__(self, response: str):
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int]:
        self.calls.append((system_prompt, user_prompt))
        return self._response, 77


def critique_json(scores: dict[str, tuple[int, int, int, str]]) -> str:
    return json.dumps(
        {
            "critiques": [
                {
                    "proposal_id": label,
                    "scores": {"correctness_risk": r, "completeness": c, "reasoning_quality": q},
                    "text": text,
                }
                for label, (r, c, q, text) in scores.items()
            ]
        }
    )


def make_proposals(names: list[str]) -> dict[str, Proposal]:
    # Deliberately content that does NOT embed the persona name, so a test
    # checking anonymization can't accidentally pass/fail on fixture leakage
    # rather than on critic.py's own behavior.
    return {name: Proposal(answer="42", reasoning="because arithmetic", self_confidence=0.5) for name in names}


class TestAnonymize:
    def test_assigns_letters_in_order(self):
        mapping = anonymize(["Analyst", "Skeptic", "Creative"])
        assert mapping == {"Analyst": "A", "Skeptic": "B", "Creative": "C"}

    def test_single_persona(self):
        assert anonymize(["Analyst"]) == {"Analyst": "A"}

    def test_too_many_personas_rejected(self):
        with pytest.raises(ValueError):
            anonymize([f"persona-{i}" for i in range(27)])

    def test_empty_list(self):
        assert anonymize([]) == {}


class TestLoadRubricText:
    def test_rubric_loads_and_documents_all_three_dimensions(self):
        text = load_rubric_text()
        assert "correctness_risk" in text
        assert "completeness" in text
        assert "reasoning_quality" in text
        assert len(text) > 100


class TestCritiqueRound:
    def test_happy_path_maps_letters_back_to_personas(self):
        proposals = make_proposals(["Analyst", "Skeptic", "Creative"])
        response = critique_json(
            {"A": (8, 8, 8, "good"), "B": (7, 7, 7, "ok"), "C": (6, 6, 6, "meh")}
        )
        client = FakeClient(response)

        critiques, letters, tokens = critique_round("Q?", proposals, client)

        assert set(critiques) == {"Analyst", "Skeptic", "Creative"}
        assert letters == {"Analyst": "A", "Skeptic": "B", "Creative": "C"}
        assert critiques["Analyst"].scores.correctness_risk == 8
        assert critiques["Creative"].text == "meh"
        assert tokens == 77

    def test_prompt_never_reveals_persona_names(self):
        # The whole point of anonymization: the user prompt sent to the
        # critic must not contain the literal persona names, only letters.
        proposals = make_proposals(["Analyst", "Skeptic", "Creative"])
        response = critique_json({"A": (8, 8, 8, "x"), "B": (7, 7, 7, "x"), "C": (6, 6, 6, "x")})
        client = FakeClient(response)

        critique_round("Q?", proposals, client)

        _, user_prompt = client.calls[0]
        assert "Analyst" not in user_prompt
        assert "Skeptic" not in user_prompt
        assert "Creative" not in user_prompt
        assert "Proposal A" in user_prompt

    def test_hallucinated_extra_label_is_skipped_not_fatal(self):
        proposals = make_proposals(["Analyst", "Skeptic"])
        # Critic invents a "C" that was never asked about.
        response = critique_json({"A": (8, 8, 8, "x"), "B": (7, 7, 7, "x"), "C": (5, 5, 5, "hallucinated")})
        client = FakeClient(response)

        critiques, letters, _ = critique_round("Q?", proposals, client)

        assert set(critiques) == {"Analyst", "Skeptic"}

    def test_missing_label_raises_with_informative_message(self):
        proposals = make_proposals(["Analyst", "Skeptic", "Creative"])
        # Critic forgot to score "C".
        response = critique_json({"A": (8, 8, 8, "x"), "B": (7, 7, 7, "x")})
        client = FakeClient(response)

        with pytest.raises(ValueError, match="Creative"):
            critique_round("Q?", proposals, client)
