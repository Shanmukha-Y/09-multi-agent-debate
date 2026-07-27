"""Regression tests for a real incident: a live subset bench run was
externally terminated partway through and lost all completed work because
run_bench.py only wrote results to disk once, at the very end. Two things
are tested here, no network:

1. run_bench() never lets one bad (question, arm) attempt kill the loop --
   both the arena call and grading are inside the same try/except, and a
   raised exception is recorded in `errors` rather than propagating.
2. The `on_result` checkpoint callback fires after EVERY arm attempt
   (success or failure), so a caller writing it to disk each time means a
   run that dies mid-way still has all completed work on disk, not none.
"""

from __future__ import annotations

import json

import pytest

import bench.run_bench as run_bench_module
from bench.run_bench import grade, run_bench


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
