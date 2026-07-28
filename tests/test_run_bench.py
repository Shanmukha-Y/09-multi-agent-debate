"""Regression tests for two real incidents on a live subset bench run that
was externally terminated (twice) partway through:

1. run_bench.py only wrote results to disk once, at the very end -- fixed
   with per-attempt checkpointing (`on_result` fires after every arm, a
   grading exception is now caught alongside the arena call).
2. Even with checkpointing, a killed run had no way to pick back up where
   it left off short of re-running (and re-paying for) everything --
   fixed with --resume: (question_id, arm) pairs already present in a
   prior checkpoint's results[] are skipped, and pairs that previously
   failed are retried rather than skipped.

No network in any of these.
"""

from __future__ import annotations

import json

import pytest

import bench.run_bench as run_bench_module
from bench.run_bench import QuestionResult, grade, load_checkpoint, run_bench


QUESTIONS = [
    {
        "id": "q01", "category": "reasoning_puzzle", "question": "2+2?",
        "match_type": "numeric_range", "reference_low": 4, "reference_high": 4, "subset": True,
    },
    {
        "id": "q02", "category": "reasoning_puzzle", "question": "3+3?",
        "match_type": "numeric_range", "reference_low": 6, "reference_high": 6, "subset": True,
    },
]


class FakeArenaResult:
    def __init__(self, answer: str, confidence: float = 0.9, total_tokens: int = 100, calls_made: int = 1):
        self.answer = answer
        self.confidence = confidence
        self.total_tokens = total_tokens
        self.calls_made = calls_made


def test_run_arm_exception_is_caught_and_run_continues(monkeypatch):
    """The exact shape of the real incident: one arm on one question raises
    (e.g. SchemaEnforcementError after exhausting repair attempts) -- the
    run must not die, and every OTHER (question, arm) pair must still
    complete and appear in the results."""

    call_log: list[tuple[str, str]] = []

    def fake_run_arm(arm: str, question_text: str):
        call_log.append((arm, question_text))
        if arm == "self_consistency" and question_text == "2+2?":
            raise ValueError("Failed to produce a valid Proposal after 2 attempt(s)")
        return FakeArenaResult(answer="4" if "2+2" in question_text else "6")

    monkeypatch.setattr(run_bench_module, "run_arm", fake_run_arm)

    results, errors = run_bench(QUESTIONS, arms=("single_shot", "self_consistency", "debate"), verbose=False)

    # 2 questions x 3 arms = 6 attempts total; 1 fails, 5 succeed.
    assert len(results) == 5
    assert len(errors) == 1
    assert errors[0] == {
        "question_id": "q01", "arm": "self_consistency",
        "seconds": pytest.approx(errors[0]["seconds"]),  # timing is non-deterministic, just check the shape
        "error": "Failed to produce a valid Proposal after 2 attempt(s)",
    }
    # Every arm was still attempted for both questions -- the failure on
    # q01/self_consistency didn't skip q01/debate or any of q02.
    assert len(call_log) == 6


def test_grading_exception_is_also_caught_not_just_arena_call(monkeypatch):
    """grade() used to run OUTSIDE the try/except -- a bad question dict
    (unknown match_type, missing reference field) would crash the whole
    run even though the arena call itself succeeded fine."""

    bad_questions = [
        {"id": "qbad", "category": "x", "question": "?", "match_type": "not_a_real_type", "subset": True},
    ]

    def fake_run_arm(arm: str, question_text: str):
        return FakeArenaResult(answer="42")

    monkeypatch.setattr(run_bench_module, "run_arm", fake_run_arm)

    results, errors = run_bench(bad_questions, arms=("single_shot",), verbose=False)

    assert results == []
    assert len(errors) == 1
    assert errors[0]["question_id"] == "qbad"
    assert "unknown match_type" in errors[0]["error"]


def test_on_result_checkpoint_fires_after_every_attempt_success_and_failure(monkeypatch):
    snapshots: list[tuple[int, int]] = []  # (n_results, n_errors) at each checkpoint

    def fake_run_arm(arm: str, question_text: str):
        if arm == "debate" and question_text == "3+3?":
            raise RuntimeError("simulated external kill mid-call")
        return FakeArenaResult(answer="4" if "2+2" in question_text else "6")

    monkeypatch.setattr(run_bench_module, "run_arm", fake_run_arm)

    def on_result(results, errors):
        snapshots.append((len(results), len(errors)))

    results, errors = run_bench(
        QUESTIONS, arms=("single_shot", "debate"), verbose=False, on_result=on_result
    )

    # 2 questions x 2 arms = 4 attempts; checkpoint fires exactly once per attempt.
    assert len(snapshots) == 4
    # Snapshots are monotonically non-decreasing in total attempts recorded,
    # and the LAST snapshot matches the final returned results/errors --
    # i.e. a checkpoint write at any point during the run reflects real,
    # already-completed progress, not a stale or incomplete state.
    for n_results, n_errors in snapshots:
        assert n_results + n_errors <= len(results) + len(errors)
    assert snapshots[-1] == (len(results), len(errors))


def test_checkpoint_written_to_disk_reflects_progress_after_every_attempt(tmp_path, monkeypatch):
    """End-to-end proof of the actual fix: write a real file to disk on
    every checkpoint. If the process were killed at any point (right after
    the first attempt, say), the file on disk at that instant already
    contains that first result -- not empty, not absent, not stale from a
    single end-of-run write. This is what turns "one crash loses the whole
    run" into "one crash loses at most the in-flight attempt.\""""

    out_path = tmp_path / "partial_results.json"
    file_states_seen: list[int] = []

    def fake_run_arm(arm: str, question_text: str):
        return FakeArenaResult(answer="4" if "2+2" in question_text else "6")

    monkeypatch.setattr(run_bench_module, "run_arm", fake_run_arm)

    def on_result(results, errors):
        out_path.write_text(json.dumps({"results": [r.__dict__ for r in results], "errors": errors}))
        # Read back what's actually on disk right now, as a "what would
        # survive a kill this instant" check.
        file_states_seen.append(len(json.loads(out_path.read_text())["results"]))

    run_bench(QUESTIONS, arms=("single_shot",), verbose=False, on_result=on_result)

    # 2 questions x 1 arm = 2 attempts -> checkpoint fires twice, and disk
    # state grows monotonically: 1 result after the first, 2 after the second.
    assert file_states_seen == [1, 2]

    # After the full run, the file reflects everything -- and this exact
    # content is what a killed-after-attempt-1 run would have left behind,
    # proven by file_states_seen[0] == 1 above rather than 0.
    final = json.loads(out_path.read_text())
    assert len(final["results"]) == 2


class TestResume:
    def test_already_completed_pairs_are_skipped_not_rerun(self, monkeypatch):
        call_log: list[tuple[str, str]] = []

        def fake_run_arm(arm: str, question_text: str):
            call_log.append((arm, question_text))
            return FakeArenaResult(answer="4" if "2+2" in question_text else "6")

        monkeypatch.setattr(run_bench_module, "run_arm", fake_run_arm)

        # Pretend q01/single_shot already succeeded on a prior (killed) run.
        prior_result = QuestionResult(
            question_id="q01", category="reasoning_puzzle", arm="single_shot",
            answer="4", confidence=0.9, correct=True, tokens=100, calls=1, seconds=5.0,
        )

        results, errors = run_bench(
            QUESTIONS, arms=("single_shot", "self_consistency"), verbose=False,
            initial_results=[prior_result],
        )

        # q01/single_shot must NOT have been called again.
        assert ("single_shot", "2+2?") not in call_log
        # Everything else still ran: q01/self_consistency, q02/single_shot, q02/self_consistency.
        assert len(call_log) == 3
        # The final results include the carried-forward prior result plus the 3 new ones.
        assert len(results) == 4
        assert prior_result in results

    def test_previously_failed_pairs_are_retried_not_skipped(self, monkeypatch):
        call_log: list[tuple[str, str]] = []

        def fake_run_arm(arm: str, question_text: str):
            call_log.append((arm, question_text))
            return FakeArenaResult(answer="4" if "2+2" in question_text else "6")

        monkeypatch.setattr(run_bench_module, "run_arm", fake_run_arm)

        prior_error = {"question_id": "q01", "arm": "single_shot", "seconds": 3.0, "error": "simulated prior failure"}

        results, errors = run_bench(
            QUESTIONS, arms=("single_shot",), verbose=False, initial_errors=[prior_error],
        )

        # The previously-failed pair WAS retried (unlike a succeeded pair).
        assert ("single_shot", "2+2?") in call_log
        # It succeeded this time, so the stale error entry is dropped from
        # the final errors list -- a retry that succeeds must not still be
        # reported as a failure.
        assert prior_error not in errors
        assert any(r.question_id == "q01" and r.arm == "single_shot" for r in results)

    def test_load_checkpoint_reconstructs_results_and_errors(self, tmp_path):
        out_path = tmp_path / "checkpoint.json"
        out_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "question_id": "q01", "category": "reasoning_puzzle", "arm": "single_shot",
                            "answer": "4", "confidence": 0.9, "correct": True, "tokens": 100, "calls": 1, "seconds": 5.0,
                        }
                    ],
                    "errors": [{"question_id": "q02", "arm": "debate", "seconds": 1.0, "error": "boom"}],
                }
            )
        )

        results, errors = load_checkpoint(str(out_path))

        assert len(results) == 1
        assert isinstance(results[0], QuestionResult)
        assert results[0].question_id == "q01"
        assert errors == [{"question_id": "q02", "arm": "debate", "seconds": 1.0, "error": "boom"}]

    def test_load_checkpoint_missing_file_returns_empty(self, tmp_path):
        results, errors = load_checkpoint(str(tmp_path / "does_not_exist.json"))
        assert results == []
        assert errors == []
