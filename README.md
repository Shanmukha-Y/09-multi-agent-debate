# Multi-Agent Debate System

Three copies of the same 9B model debate under an anonymized critic and deterministic vote math, testing whether orchestration alone — no bigger model — buys a measurable accuracy gain over a single call.

## What it does

- Three proposer personas (Analyst, Skeptic, Creative) share one model (`qwen3.5:9b`) and differ only in system prompt and temperature — each is given a *mandatory reasoning procedure* (not a personality), so their transcripts genuinely diverge in how they reach an answer.
- A critic scores proposals against a versioned rubric after they're relabeled A/B/C — persona names are never in its prompt, and it has no code path to propose an answer itself, so bias control is structural, not a prompt request.
- A rebuttal round lets proposers respond to anonymized peer critiques before a final re-score.
- The winner is picked by pure-Python arithmetic (`critic_total × agreement_multiplier`), with a >15% lead over the nearest *disagreeing* competitor required for a clear winner — zero LLM involvement, zero non-determinism to work around in tests.
- The aggregator makes exactly one LLM call, for prose only; the winning answer, confidence, split flag, and dissent list are all decided by code before that call, and a failed summary falls back to a deterministic template rather than losing a correct verdict.
- A 3-arm bench (single-shot, self-consistency, debate) grades against reference ranges or instructed labels, not an LLM judge, with per-attempt checkpointing and `--resume`.

## Quick start

```
# requires: Python 3.11+, uv, a local Ollama server with qwen3.5:9b pulled
uv sync
curl -s http://localhost:11434/api/version   # sanity check the model is reachable

uv run pytest                                 # 70 tests, zero network, <1s
uv run pytest -m integration tests/test_integration.py -v   # one live end-to-end debate, needs Ollama

uv run debate ask "A bat and ball cost \$1.10 total; the bat costs \$1 more than the ball. What does the ball cost?" --live
uv run python bench/run_bench.py --subset --out bench/results/subset_run.json
uv run python bench/run_bench.py --subset --resume --out bench/results/subset_run.json   # picks up where a killed run left off
```

`debateRender.html` is a static rendered view of the committed live bat-and-ball transcript — open it directly to see the debate play out round by round without running anything.

## Learnings

- **ag2 turned out to be two different frameworks under one name.** The spec called for AutoGen (ag2) with an Ollama-compatible client. The PyPI package literally named `ag2` is an unrelated namespace-squatted agent-protocol project (`a2a`, `mcp_ui`, `hitl` modules, nothing to do with debate). The real historical package, `pyautogen`, now installs Microsoft's rewritten `autogen-agentchat`/`autogen-core` (v0.4+), an async actor-model runtime that dropped the classic `ConversableAgent` + `llm_config` API entirely. Both problems surfaced within ~15 minutes of the authorized 30-minute investigation budget, and this project's hardest requirements — anonymizing proposals before the critic sees them, deterministic zero-LLM vote math, rebuttal prompts assembled from a specific mix of own-critique plus anonymized peer critiques — are tight pipeline control flow, not conversational turn-taking, so the fallback to a hand-rolled OpenAI-SDK-pointed-at-Ollama pipeline (the same pattern project-01 established) was taken immediately rather than fighting an actor-model abstraction for five milestones.
- **A literal score tie needed a second check to mean the right thing.** The vote logic initially risked treating any tied top score as a split. It was corrected so a tie only signals disagreement when it's a tie between proposers who gave *different* answers — a tie between proposers who agree is unanimous consensus, not a split. `tests/test_voting.py` covers the threshold boundary, exact ties, and this unanimous-tie edge case explicitly.
- **Timeouts tuned for typical call length quietly favor debate for the wrong reason.** At this server's measured ~65-68 tok/s, qwen3.5:9b's persona-mandated step-by-step prompting drives open-ended Fermi-estimate completions past 10,000+ decoded tokens (observed up to ~11,300 on one call) — enough alone to blow a 180s ceiling before queueing or prompt processing. A live subset run against an otherwise idle server timed out 7 of 18 attempts this way, every one an estimation question in a baseline arm. The ceiling was raised to 600s with that measured rationale recorded in `config.py`. The deeper lesson: comparing arms with different call-count shapes at a fixed timeout will make the arm with many short calls (debate) look artificially more robust than the arm with one long call (single-shot/self-consistency) — not because it reasons better, but because its call shape survives timeouts better. The bench table's own 100%-completed accuracy for debate is flagged as survivor-biased for exactly this reason, not presented as a clean win.
- **Retry layers stack multiplicatively, not additively.** Three independent retry layers exist (transport retries, schema-repair retries, self-consistency sampling); worst case for one arm on one question is 3 samples × 2 schema attempts × 3 transport retries × 600s — a theoretical 10,800s if every layer maxes out simultaneously, despite no single layer looking unreasonable alone. None were designed with the others in mind; a shared wall-clock budget per logical call is flagged as future work instead of three independently generous ceilings.
- **Two external process kills, zero lost results.** A multi-hour live bench run was killed twice by what the timing evidence points to as the session's own background-task lifecycle (not a code bug), plus one deliberate operator wall-clock cap. Per-attempt checkpointing (`on_result` firing after every arm attempt, success or failure) and `--resume` meant all three interruptions cost only the one in-flight attempt each — the committed subset run reports its own status as `"capped"` with a `capped_note`, honestly: 18 of 18 pairs attempted, 11 completed, 7 recorded (not silently dropped) failures, most of them the same `reasoning` field arriving as a JSON array instead of a string.

See `readme.html` for the full write-up, including the bench table, the measured 7.97x token-cost multiple against single-shot, and the live bat-and-ball transcript walkthrough.
