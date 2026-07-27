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


def run_bench(questions: list[dict], arms: tuple[str, ...] = ARMS, verbose: bool = True) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    for i, q in enumerate(questions, 1):
        if verbose:
            console.print(f"[dim]({i}/{len(questions)}) {q['id']} [{q['category']}][/dim] {q['question'][:70]}...")
        for arm in arms:
            t0 = time.monotonic()
            try:
                arena_result = run_arm(arm, q["question"])
            except Exception as exc:  # noqa: BLE001 - one bad call shouldn't kill the whole bench run
                console.print(f"  [red]{arm} failed on {q['id']}: {exc}[/red]")
                continue
            elapsed = time.monotonic() - t0
            correct = grade(arena_result.answer, q)
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
    return results


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


def main(args: argparse.Namespace) -> None:
    questions = load_questions(Path(args.questions), subset_only=args.subset)
    label = "subset" if args.subset else "full"
    console.print(f"[bold]Running bench: {label} ({len(questions)} questions x {len(ARMS)} arms)[/bold]")

    results = run_bench(questions)
    summary = summarize(results)
    print_summary_table(summary)

    out_path = args.out
    if out_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = str(RESULTS_DIR / f"bench_{label}_{timestamp}.json")

    payload = {
        "label": label,
        "questions_path": args.questions,
        "n_questions": len(questions),
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    console.print(f"\n[dim]wrote results to {out_path}[/dim]")


def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the debate bench")
    parser.add_argument("--subset", action="store_true", help="run only the ~6-question stratified subset")
    parser.add_argument("--questions", type=str, default=str(QUESTIONS_PATH))
    parser.add_argument("--out", type=str, default=None)
    return parser


if __name__ == "__main__":
    main(_build_standalone_parser().parse_args())
