"""30-question bench across three arms: single-shot, self-consistency (3
samples + majority), and full debate. Metrics: accuracy vs. references,
token cost per arm, and a simple confidence-calibration breakdown (when an
arm reports high confidence, is it right more often?).

Grading is intentionally simple and reproducible, not an LLM judge:
- numeric_range: extract the first number from the answer (same extraction
  voting.py uses for agreement) and check it falls in [reference_low, high].
- contains: lowercase-normalized substring check against reference_keywords.
  Every "contains" question in questions.jsonl explicitly instructs the
  model to answer with a short, distinctive label ("Job A" vs "Job B",
  "Approve" vs "Request Changes") specifically so this coarse check stays
  reliable — a bare keyword grader would false-positive on hedged prose.

Run the full 30 questions:  uv run python bench/run_bench.py
Run the ~6-question subset: uv run python bench/run_bench.py --subset
Or via the CLI:             uv run debate bench [--subset]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from debate import arena, voting

console = Console()

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

ARMS = ("single_shot", "self_consistency", "debate")

CONFIDENCE_BINS = [(0.7, 1.01, "high (>=0.7)"), (0.4, 0.7, "medium (0.4-0.7)"), (0.0, 0.4, "low (<0.4)")]


def load_questions(path: Path = QUESTIONS_PATH, subset_only: bool = False) -> list[dict]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if subset_only and not q.get("subset", False):
                continue
            questions.append(q)
    return questions


def grade(answer: str, question: dict) -> bool:
    if question["match_type"] == "numeric_range":
        num = voting.extract_number(answer)
        if num is None:
            return False
        return question["reference_low"] <= num <= question["reference_high"]
    elif question["match_type"] == "contains":
        normalized = answer.lower()
        return any(kw in normalized for kw in question["reference_keywords"])
    raise ValueError(f"unknown match_type: {question['match_type']}")


@dataclass
class QuestionResult:
    question_id: str
    category: str
    arm: str
    answer: str
    confidence: float
    correct: bool
    tokens: int
    calls: int
    seconds: float


def run_arm(arm: str, question_text: str) -> arena.ArenaResult:
    if arm == "single_shot":
        return arena.run_single_shot(question_text)
    elif arm == "self_consistency":
        return arena.run_self_consistency(question_text)
    elif arm == "debate":
        result, _transcript = arena.run_debate_arm(question_text)
        return result
    raise ValueError(f"unknown arm: {arm}")


def dedupe_errors(errors: list[dict]) -> list[dict]:
    """Collapse to at most one error entry per (question_id, arm) pair,
    keeping the LATEST entry for any pair that appears more than once
    (e.g. a schema failure on one run, then a timeout on a resume's retry)."""
    deduped: dict[tuple[str, str], dict] = {}
    for e in errors:
        deduped[(e["question_id"], e["arm"])] = e  # later entries win
    return list(deduped.values())


CheckpointCallback = Callable[[list[QuestionResult], list[dict]], None]


def run_bench(
    questions: list[dict],
    arms: tuple[str, ...] = ARMS,
    verbose: bool = True,
    on_result: CheckpointCallback | None = None,
    initial_results: list[QuestionResult] | None = None,
    initial_errors: list[dict] | None = None,
) -> tuple[list[QuestionResult], list[dict]]:
    """Runs every (question, arm) pair. Never lets one bad call kill the run:
    both the arena call itself AND grading are inside the same try/except,
    so a malformed model output, a schema-validation exhaustion, or a
    grading error on one arm is caught, recorded in `errors`, and the loop
    moves on to the next arm.

    If `on_result` is given, it fires after EVERY single arm attempt
    (success or failure) with the results/errors accumulated so far. Callers
    use this to checkpoint partial progress to disk after each of the ~18-90
    individual LLM calls a bench run makes, not just once at the end -- a
    multi-hour live bench run getting killed (crash, external signal, box
    reboot, doesn't matter which) should never again be able to lose
    everything completed up to that point.

    `initial_results` / `initial_errors` seed a RESUME: any (question_id,
    arm) pair already present in `initial_results` is skipped rather than
    re-run (see load_checkpoint()). Pairs that previously failed
    (`initial_errors`) are NOT skipped -- a transient failure deserves
    another attempt, not a permanent skip, so they're carried forward into
    the returned `errors` list but re-attempted like anything else.

    At most one error entry is ever kept per (question_id, arm) pair. A
    pair that fails across multiple resumes (e.g. a schema failure on the
    first run, then a timeout on a retry) previously stacked a fresh entry
    on top of the stale one every time -- `initial_errors` is deduped down
    to the latest entry per pair up front, and a fresh failure during this
    run replaces rather than appends to any existing entry for that pair.
    """
    results: list[QuestionResult] = list(initial_results or [])
    already_done = {(r.question_id, r.arm) for r in results}
    errors: list[dict] = dedupe_errors(initial_errors or [])

    for i, q in enumerate(questions, 1):
        if verbose:
            console.print(f"[dim]({i}/{len(questions)}) {q['id']} [{q['category']}][/dim] {q['question'][:70]}...")
        for arm in arms:
            if (q["id"], arm) in already_done:
                if verbose:
                    console.print(f"  [dim]{arm} already completed for {q['id']}, skipping (resume)[/dim]")
                continue
            t0 = time.monotonic()
            try:
                arena_result = run_arm(arm, q["question"])
                elapsed = time.monotonic() - t0
                correct = grade(arena_result.answer, q)
            except Exception as exc:  # noqa: BLE001 - one bad arm shouldn't kill the whole bench run
                elapsed = time.monotonic() - t0
                errors = [e for e in errors if (e["question_id"], e["arm"]) != (q["id"], arm)]
                errors.append({"question_id": q["id"], "arm": arm, "seconds": elapsed, "error": str(exc)})
                if verbose:
                    console.print(f"  [red]{arm} failed on {q['id']}: {exc}[/red]")
                if on_result is not None:
                    on_result(results, errors)
                continue

            results.append(
                QuestionResult(
                    question_id=q["id"],
                    category=q["category"],
                    arm=arm,
                    answer=arena_result.answer,
                    confidence=arena_result.confidence,
                    correct=correct,
                    tokens=arena_result.total_tokens,
                    calls=arena_result.calls_made,
                    seconds=elapsed,
                )
            )
            if verbose:
                mark = "[green]correct[/green]" if correct else "[red]wrong[/red]"
                console.print(
                    f"    {arm:16s} {mark:20s} conf={arena_result.confidence:.2f} "
                    f"tokens={arena_result.total_tokens:5d} ({elapsed:.1f}s)"
                )
            if on_result is not None:
                on_result(results, errors)

    # A pair that failed on a prior run but succeeded on this resume's retry
    # now has both a stale error entry (carried forward from initial_errors)
    # and a real result -- drop the stale entry so the final error list only
    # ever reflects pairs that are STILL missing a result, not ones that
    # eventually succeeded.
    succeeded = {(r.question_id, r.arm) for r in results}
    errors = [e for e in errors if (e["question_id"], e["arm"]) not in succeeded]
    return results, errors


def summarize(results: list[QuestionResult], arms: tuple[str, ...] = ARMS) -> dict:
    summary = {}
    for arm in arms:
        arm_results = [r for r in results if r.arm == arm]
        if not arm_results:
            continue
        n = len(arm_results)
        n_correct = sum(1 for r in arm_results if r.correct)
        total_tokens = sum(r.tokens for r in arm_results)
        calibration = {}
        for lo, hi, label in CONFIDENCE_BINS:
            bucket = [r for r in arm_results if lo <= r.confidence < hi]
            if bucket:
                bucket_acc = sum(1 for r in bucket if r.correct) / len(bucket)
                calibration[label] = {"n": len(bucket), "accuracy": bucket_acc}
        summary[arm] = {
            "n_questions": n,
            "accuracy": n_correct / n,
            "total_tokens": total_tokens,
            "avg_tokens_per_question": total_tokens / n,
            "calibration": calibration,
        }
    return summary


def print_summary_table(summary: dict) -> None:
    table = Table(title="Bench results")
    table.add_column("Arm")
    table.add_column("N")
    table.add_column("Accuracy")
    table.add_column("Total tokens")
    table.add_column("Avg tokens/Q")
    for arm, s in summary.items():
        table.add_row(
            arm, str(s["n_questions"]), f"{s['accuracy']:.0%}", str(s["total_tokens"]),
            f"{s['avg_tokens_per_question']:.0f}",
        )
    console.print(table)

    for arm, s in summary.items():
        if not s["calibration"]:
            continue
        console.print(f"\n[bold]{arm} confidence calibration[/bold]")
        for label, c in s["calibration"].items():
            console.print(f"  {label}: n={c['n']}, accuracy={c['accuracy']:.0%}")


def load_checkpoint(out_path: str) -> tuple[list[QuestionResult], list[dict]]:
    """Reconstruct (results, errors) from a previously-written checkpoint
    file, for --resume. Returns ([], []) if the file doesn't exist yet --
    resuming a run that never started is just starting it."""
    path = Path(out_path)
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text())
    results = [QuestionResult(**r) for r in payload.get("results", [])]
    errors = list(payload.get("errors", []))
    return results, errors


def _write_checkpoint(
    out_path: str, label: str, questions_path: str, n_questions: int, n_arms: int,
    results: list[QuestionResult], errors: list[dict], status: str,
) -> None:
    payload = {
        "label": label,
        "questions_path": questions_path,
        "n_questions": n_questions,
        "status": status,  # "in_progress" until the run finishes, then "complete"
        "n_arm_attempts_done": len(results) + len(errors),
        "n_arm_attempts_total": n_questions * n_arms,
        "summary": summarize(results) if results else {},
        "results": [asdict(r) for r in results],
        "errors": errors,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))


def main(args: argparse.Namespace) -> None:
    questions = load_questions(Path(args.questions), subset_only=args.subset)
    label = "subset" if args.subset else "full"
    console.print(f"[bold]Running bench: {label} ({len(questions)} questions x {len(ARMS)} arms)[/bold]")

    out_path = args.out
    if out_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = str(RESULTS_DIR / f"bench_{label}_{timestamp}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    initial_results: list[QuestionResult] = []
    initial_errors: list[dict] = []
    if getattr(args, "resume", False):
        initial_results, initial_errors = load_checkpoint(out_path)
        if initial_results or initial_errors:
            console.print(
                f"[dim]resuming from {out_path}: {len(initial_results)} result(s) already done, "
                f"{len(initial_errors)} prior error(s) will be retried[/dim]"
            )

    def checkpoint(results: list[QuestionResult], errors: list[dict]) -> None:
        _write_checkpoint(out_path, label, args.questions, len(questions), len(ARMS), results, errors, "in_progress")

    # Write an immediate in_progress checkpoint before any new work happens.
    # Without this, a --resume run's output file keeps whatever `status` the
    # PREVIOUS run left behind (typically "complete") until the first
    # genuinely new attempt fires the on_result callback -- skipped,
    # already-done pairs don't trigger it. That stale "complete" reads as a
    # false completion signal to anyone polling the file early in a resume
    # (a monitoring script, a teammate, a human), when the run has actually
    # just started retrying failures.
    checkpoint(initial_results, dedupe_errors(initial_errors))

    results, errors = run_bench(
        questions, on_result=checkpoint, initial_results=initial_results, initial_errors=initial_errors
    )
    summary = summarize(results)
    print_summary_table(summary)
    if errors:
        console.print(f"\n[yellow]{len(errors)} arm attempt(s) failed and were skipped (see 'errors' in {out_path})[/yellow]")

    _write_checkpoint(out_path, label, args.questions, len(questions), len(ARMS), results, errors, "complete")
    console.print(f"\n[dim]wrote results to {out_path}[/dim]")


def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the debate bench")
    parser.add_argument("--subset", action="store_true", help="run only the ~6-question stratified subset")
    parser.add_argument("--questions", type=str, default=str(QUESTIONS_PATH))
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--resume", action="store_true",
        help="skip (question, arm) pairs already completed in --out's existing checkpoint file",
    )
    return parser


if __name__ == "__main__":
    main(_build_standalone_parser().parse_args())
