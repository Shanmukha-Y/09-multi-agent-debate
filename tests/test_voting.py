"""Pure math: no LLM, no network. Winner/split thresholds, agreement
scoring, tie-breaking, and the confidence formula — all deterministic given
inputs, so exhaustively testable.
"""

from __future__ import annotations

import pytest

from debate.aggregator import compute_confidence, compute_dissent
from debate.messages import Critique, CritiqueScores, Proposal
from debate.voting import answers_agree, cluster_answers, score_round


def make_critique(label: str, correctness_risk: int, completeness: int, reasoning_quality: int, text: str = "ok") -> Critique:
    return Critique(
        proposal_id=label,
        scores=CritiqueScores(
            correctness_risk=correctness_risk, completeness=completeness, reasoning_quality=reasoning_quality
        ),
        text=text,
    )


def make_proposal(answer: str, reasoning: str = "because", confidence: float = 0.8) -> Proposal:
    return Proposal(answer=answer, reasoning=reasoning, self_confidence=confidence)


# --- answers_agree -----------------------------------------------------


class TestAnswersAgree:
    def test_identical_numbers_agree(self):
        assert answers_agree("0.05", "0.05")

    def test_numbers_within_tolerance_agree(self):
        assert answers_agree("$0.05", "$0.0501")

    def test_numbers_outside_tolerance_disagree(self):
        assert not answers_agree("$0.05", "$0.10")

    def test_both_zero_agrees(self):
        assert answers_agree("0", "0.0 tuners")

    def test_identical_text_agrees(self):
        assert answers_agree("Yes, ship it", "yes ship it")

    def test_different_text_disagrees(self):
        assert not answers_agree("Yes, ship it", "No, do not ship it")

    def test_currency_and_bare_number_agree(self):
        assert answers_agree("$1.10", "1.10")

    def test_thousands_separator_parses(self):
        assert answers_agree("1,200 tuners", "1200")


class TestClusterAnswers:
    def test_all_agree_one_cluster(self):
        clusters = cluster_answers(["0.05", "0.05", "$0.05"])
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_no_agreement_all_singletons(self):
        clusters = cluster_answers(["0.05", "0.10", "0.20"])
        assert len(clusters) == 3

    def test_majority_plus_outlier(self):
        clusters = cluster_answers(["0.05", "0.05", "0.10"])
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [1, 2]

    def test_empty_input(self):
        assert cluster_answers([]) == []


# --- score_round: winner / split threshold ------------------------------


class TestScoreRound:
    def test_requires_matching_personas(self):
        with pytest.raises(ValueError):
            score_round({"A": "1"}, {"B": make_critique("A", 5, 5, 5)})

    def test_requires_at_least_one_proposal(self):
        with pytest.raises(ValueError):
            score_round({}, {})

    def test_clear_winner_above_threshold(self):
        # Analyst: 30/30, no agreement. Skeptic: 15/30, no agreement.
        # (30 - 15) / 30 = 50% lead > 15% -> clear winner.
        proposals = {"Analyst": "42", "Skeptic": "7"}
        critiques = {
            "Analyst": make_critique("A", 10, 10, 10),
            "Skeptic": make_critique("B", 5, 5, 5),
        }
        result = score_round(proposals, critiques)
        assert result.is_clear_winner
        assert result.winner.persona == "Analyst"
        assert result.lead_ratio == pytest.approx(0.5)

    def test_split_at_exact_threshold_boundary(self):
        # Construct scores where lead_ratio is exactly 0.15 -> NOT > threshold -> split.
        # top=20, runner_up=17 -> (20-17)/20 = 0.15 exactly.
        proposals = {"A": "1", "B": "2"}
        critiques = {
            "A": Critique(proposal_id="A", scores=CritiqueScores(correctness_risk=7, completeness=7, reasoning_quality=6), text="x"),
            "B": Critique(proposal_id="B", scores=CritiqueScores(correctness_risk=6, completeness=6, reasoning_quality=5), text="x"),
        }
        result = score_round(proposals, critiques)
        assert result.lead_ratio == pytest.approx(0.15)
        assert not result.is_clear_winner  # boundary is exclusive

    def test_split_below_threshold(self):
        proposals = {"Analyst": "42", "Skeptic": "43"}
        critiques = {
            "Analyst": make_critique("A", 8, 8, 8),
            "Skeptic": make_critique("B", 7, 8, 8),
        }
        result = score_round(proposals, critiques)
        assert not result.is_clear_winner

    def test_exact_tie_is_split_and_deterministic(self):
        proposals = {"Skeptic": "5", "Analyst": "10"}
        critiques = {
            "Skeptic": make_critique("A", 6, 6, 6),
            "Analyst": make_critique("B", 6, 6, 6),
        }
        result = score_round(proposals, critiques)
        assert not result.is_clear_winner
        assert result.lead_ratio == 0.0
        # Tie-break is alphabetical by persona name: Analyst < Skeptic.
        assert result.winner.persona == "Analyst"

    def test_single_proposal_is_trivially_clear(self):
        proposals = {"Analyst": "42"}
        critiques = {"Analyst": make_critique("A", 5, 5, 5)}
        result = score_round(proposals, critiques)
        assert result.is_clear_winner
        assert result.runner_up is None
        assert result.lead_ratio == 1.0

    def test_agreement_bonus_breaks_a_near_tie_into_clear_winner(self):
        # Two proposers with identical critic scores, but one is corroborated
        # by a third proposer's matching answer -> agreement bonus should be
        # enough to separate them given a large enough gap is NOT required
        # here; we just check the bonus is applied in the right direction.
        proposals = {"Analyst": "42", "Skeptic": "42", "Creative": "7"}
        critiques = {
            "Analyst": make_critique("A", 7, 7, 7),
            "Skeptic": make_critique("B", 7, 7, 7),
            "Creative": make_critique("C", 7, 7, 7),
        }
        result = score_round(proposals, critiques)
        by_name = {s.persona: s for s in result.scores}
        assert by_name["Analyst"].agreement_multiplier > by_name["Creative"].agreement_multiplier
        assert by_name["Analyst"].final_score > by_name["Creative"].final_score

    def test_unanimous_agreement_with_tied_scores_is_clear_not_split(self):
        # All three proposers give the SAME answer and get identical critic
        # scores, so their final_score is a literal tie -- but a tie between
        # identical answers is unanimous consensus, not a split.
        proposals = {"Analyst": "42", "Skeptic": "42", "Creative": "42"}
        labels = {"Analyst": "A", "Skeptic": "B", "Creative": "C"}
        critiques = {name: make_critique(labels[name], 10, 10, 10) for name in proposals}
        result = score_round(proposals, critiques)
        assert result.is_clear_winner
        assert result.lead_ratio == 1.0

    def test_tied_scores_between_different_answers_is_split(self):
        proposals = {"Analyst": "42", "Skeptic": "7"}
        critiques = {
            "Analyst": make_critique("A", 8, 8, 8),
            "Skeptic": make_critique("B", 8, 8, 8),
        }
        result = score_round(proposals, critiques)
        assert not result.is_clear_winner
        assert result.lead_ratio == 0.0

    def test_scores_sorted_descending(self):
        proposals = {"A": "1", "B": "2", "C": "3"}
        critiques = {
            "A": make_critique("A", 3, 3, 3),
            "B": make_critique("B", 9, 9, 9),
            "C": make_critique("C", 5, 5, 5),
        }
        result = score_round(proposals, critiques)
        totals = [s.final_score for s in result.scores]
        assert totals == sorted(totals, reverse=True)
        assert result.scores[0].persona == "B"


# --- confidence formula ---------------------------------------------------


class TestComputeConfidence:
    def _vote(self, proposals, critiques):
        return score_round(proposals, critiques)

    def test_perfect_unanimous_single_round_is_high_confidence(self):
        proposals = {"A": "42", "B": "42", "C": "42"}
        critiques = {name: make_critique(name, 10, 10, 10) for name in proposals}
        vote = self._vote(proposals, critiques)
        confidence = compute_confidence(vote, rounds_used=1)
        assert confidence == pytest.approx(1.0)

    def test_rebuttal_round_applies_penalty_vs_single_round(self):
        proposals = {"A": "42", "B": "42", "C": "42"}
        critiques = {name: make_critique(name, 10, 10, 10) for name in proposals}
        vote = self._vote(proposals, critiques)
        conf_1round = compute_confidence(vote, rounds_used=1)
        conf_2round = compute_confidence(vote, rounds_used=2)
        assert conf_2round < conf_1round

    def test_split_applies_larger_penalty_than_clear_winner(self):
        proposals_clear = {"A": "42", "B": "7"}
        critiques_clear = {"A": make_critique("A", 10, 10, 10), "B": make_critique("B", 3, 3, 3)}
        vote_clear = self._vote(proposals_clear, critiques_clear)

        proposals_split = {"A": "42", "B": "43"}
        critiques_split = {"A": make_critique("A", 8, 8, 8), "B": make_critique("B", 7, 8, 8)}
        vote_split = self._vote(proposals_split, critiques_split)

        conf_clear = compute_confidence(vote_clear, rounds_used=1)
        conf_split = compute_confidence(vote_split, rounds_used=1)
        assert conf_split < conf_clear

    def test_confidence_bounded_zero_to_one(self):
        proposals = {"A": "42", "B": "7", "C": "3"}
        critiques = {
            "A": make_critique("A", 1, 1, 1),
            "B": make_critique("B", 1, 1, 1),
            "C": make_critique("C", 1, 1, 1),
        }
        vote = self._vote(proposals, critiques)
        confidence = compute_confidence(vote, rounds_used=2)
        assert 0.0 <= confidence <= 1.0

    def test_no_agreement_lowers_confidence_vs_full_agreement(self):
        proposals_agree = {"A": "42", "B": "42", "C": "42"}
        critiques_agree = {name: make_critique(name, 8, 8, 8) for name in proposals_agree}
        vote_agree = self._vote(proposals_agree, critiques_agree)

        proposals_disagree = {"A": "42", "B": "7", "C": "3"}
        critiques_disagree = {name: make_critique(name, 8, 8, 8) for name in proposals_disagree}
        vote_disagree = self._vote(proposals_disagree, critiques_disagree)

        conf_agree = compute_confidence(vote_agree, rounds_used=1)
        conf_disagree = compute_confidence(vote_disagree, rounds_used=1)
        assert conf_agree > conf_disagree


# --- dissent grouping -----------------------------------------------------


class TestComputeDissent:
    def test_no_dissent_when_all_agree(self):
        final = {
            "Analyst": make_proposal("42"),
            "Skeptic": make_proposal("42"),
            "Creative": make_proposal("42"),
        }
        assert compute_dissent(final, winner_name="Analyst") == []

    def test_single_dissenter_reported(self):
        final = {
            "Analyst": make_proposal("42"),
            "Skeptic": make_proposal("42"),
            "Creative": make_proposal("7", reasoning="lateral angle"),
        }
        dissent = compute_dissent(final, winner_name="Analyst")
        assert len(dissent) == 1
        assert dissent[0].personas == ["Creative"]
        assert dissent[0].answer == "7"

    def test_two_dissenters_agreeing_with_each_other_grouped(self):
        final = {
            "Analyst": make_proposal("42"),
            "Skeptic": make_proposal("7"),
            "Creative": make_proposal("7"),
        }
        dissent = compute_dissent(final, winner_name="Analyst")
        assert len(dissent) == 1
        assert dissent[0].personas == ["Creative", "Skeptic"]

    def test_two_distinct_dissenting_positions_kept_separate(self):
        final = {
            "Analyst": make_proposal("42"),
            "Skeptic": make_proposal("7"),
            "Creative": make_proposal("99"),
        }
        dissent = compute_dissent(final, winner_name="Analyst")
        assert len(dissent) == 2
