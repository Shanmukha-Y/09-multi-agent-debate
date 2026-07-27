"""CLI: `debate ask "question" [--rounds 2] [--live] [--export]`,
`debate replay <id>`, `debate list`, `debate bench [...]`.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from debate import config, store

console = Console()


def _live_event(event_type: str, data: dict) -> None:
    if event_type == "proposal":
        p = data["proposal"]
        console.print(
            Panel(
                f"[bold]{p.answer}[/bold]\n\n{p.reasoning}\n\nself_confidence: {p.self_confidence:.2f}",
                title=f"Round {data['round']} proposal — {data['persona']}",
                border_style="cyan",
            )
        )
    elif event_type == "rebuttal":
        p = data["proposal"]
        console.print(
            Panel(
                f"[bold]{p.answer}[/bold]  (stance: {p.stance})\n\n{p.reasoning}\n\n"
                f"changes: {p.changes_summary}",
                title=f"Round {data['round']} rebuttal — {data['persona']}",
                border_style="magenta",
            )
        )
    elif event_type == "critique":
        table = Table(title=f"Round {data['round']} critic scores")
        table.add_column("Persona")
        table.add_column("Label")
        table.add_column("Correctness")
        table.add_column("Completeness")
        table.add_column("Reasoning")
        table.add_column("Total")
        letters = data["letters"]
        for persona, c in data["critiques"].items():
            table.add_row(
                persona,
                letters.get(persona, "?"),
                str(c.scores.correctness_risk),
                str(c.scores.completeness),
                str(c.scores.reasoning_quality),
                str(c.scores.total),
            )
        console.print(table)
        for persona, c in data["critiques"].items():
            console.print(f"  [dim]{persona}:[/dim] {c.text}")
    elif event_type == "vote":
        vote = data["vote"]
        status = "[green]clear winner[/green]" if vote.is_clear_winner else "[yellow]split[/yellow]"
        console.print(
            f"Round {data['round']} vote: {status} — leader [bold]{vote.winner.persona}[/bold] "
            f"(score={vote.winner.final_score:.2f}, lead_ratio={vote.lead_ratio:.2%})"
        )
    elif event_type == "verdict":
        v = data["verdict"]
        border = "yellow" if v.is_split else "green"
        body = f"[bold]{v.answer}[/bold]\n\nconfidence: {v.confidence:.2f}\n\n{v.reasoning_summary}"
        if v.dissent:
            body += "\n\n[bold]Dissent:[/bold]"
            for d in v.dissent:
                body += f"\n- {', '.join(d.personas)}: {d.answer} — {d.reasoning}"
        console.print(Panel(body, title="Final verdict", border_style=border))


def cmd_ask(args: argparse.Namespace) -> None:
    from debate.debate import run_debate

    on_event = _live_event if args.live else None
    transcript = run_debate(args.question, max_rounds=args.rounds, on_event=on_event)

    if not args.live:
        v = transcript.verdict
        console.print(f"\n[bold]Answer:[/bold] {v.answer}")
        console.print(f"[bold]Confidence:[/bold] {v.confidence:.2f}")
        console.print(f"[bold]Rounds used:[/bold] {transcript.rounds_used}  "
                       f"[bold]Total tokens:[/bold] {transcript.total_tokens}")
        console.print(f"[bold]Summary:[/bold] {v.reasoning_summary}")
        if v.dissent:
            console.print("[bold yellow]Dissent:[/bold yellow]")
            for d in v.dissent:
                console.print(f"  - {', '.join(d.personas)}: {d.answer}")

    debate_id = store.save_debate(transcript, arm="debate")
    console.print(f"\n[dim]saved as debate #{debate_id}[/dim]")

    if args.export:
        store.export_transcript_json(transcript, args.export, arm="debate")
        console.print(f"[dim]exported transcript to {args.export}[/dim]")


def cmd_replay(args: argparse.Namespace) -> None:
    record = store.get_debate(args.id)
    if record is None:
        console.print(f"[red]no debate with id {args.id}[/red]")
        sys.exit(1)

    console.print(Panel(record["question"], title=f"Debate #{record['id']} ({record['arm']})"))
    for round_data in record["rounds"]:
        console.print(f"\n[bold underline]Round {round_data['round_num']}[/bold underline]")
        for persona, proposal in round_data["proposals"].items():
            console.print(Panel(f"{proposal['answer']}\n\n{proposal['reasoning']}", title=persona))
        for persona, critique in round_data["critiques"].items():
            s = critique["scores"]
            console.print(
                f"  critic on {persona} ({round_data['persona_to_letter'].get(persona, '?')}): "
                f"correctness_risk={s['correctness_risk']}, completeness={s['completeness']}, "
                f"reasoning_quality={s['reasoning_quality']} — {critique['text']}"
            )
        vote = round_data["vote"]
        console.print(
            f"  vote: {'clear winner' if vote['is_clear_winner'] else 'split'} "
            f"— {vote['winner']['persona']} (lead_ratio={vote['lead_ratio']:.2%})"
        )

    verdict = record["verdict"]
    console.print(
        Panel(
            f"{verdict['answer']}\n\nconfidence: {verdict['confidence']:.2f}\n\n{verdict['reasoning_summary']}",
            title="Final verdict",
        )
    )


def cmd_list(args: argparse.Namespace) -> None:
    debates = store.list_debates(limit=args.limit)
    table = Table(title="Debates")
    table.add_column("ID")
    table.add_column("Arm")
    table.add_column("Question")
    table.add_column("Answer")
    table.add_column("Confidence")
    table.add_column("Split")
    table.add_column("Tokens")
    for d in debates:
        table.add_row(
            str(d["id"]), d["arm"], d["question"][:50], d["answer"][:40],
            f"{d['confidence']:.2f}", "yes" if d["is_split"] else "no", str(d["total_tokens"]),
        )
    console.print(table)


def cmd_bench(args: argparse.Namespace) -> None:
    from bench.run_bench import main as bench_main

    bench_main(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="debate")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="run a debate on a question")
    ask.add_argument("question")
    ask.add_argument("--rounds", type=int, default=config.MAX_DEBATE_ROUNDS)
    ask.add_argument("--live", action="store_true")
    ask.add_argument("--export", type=str, default=None, help="also write a JSON transcript to this path")
    ask.set_defaults(func=cmd_ask)

    replay = sub.add_parser("replay", help="replay a saved debate")
    replay.add_argument("id", type=int)
    replay.set_defaults(func=cmd_replay)

    ls = sub.add_parser("list", help="list saved debates")
    ls.add_argument("--limit", type=int, default=50)
    ls.set_defaults(func=cmd_list)

    bench = sub.add_parser("bench", help="run the accuracy/cost benchmark")
    bench.add_argument("--subset", action="store_true", help="run only the ~6-question stratified subset")
    bench.add_argument("--questions", type=str, default="bench/questions.jsonl")
    bench.add_argument("--out", type=str, default=None)
    bench.set_defaults(func=cmd_bench)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
