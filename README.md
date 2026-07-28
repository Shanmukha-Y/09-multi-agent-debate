# Multi-Agent Debate System

Three structured instances of the same local 9B model debate under an anonymized critic and deterministic vote, testing whether orchestration alone can improve answer quality enough to justify its additional inference cost.

## What it does

- Uses three proposer procedures—Analyst, Skeptic, and Creative—on the same `qwen3.5:9b` model. They differ in mandatory reasoning procedure and temperature rather than fictional personality alone.
- Relabels proposals as A/B/C before criticism. The critic never receives persona names and has no code path for proposing its own answer.
- Runs one rebuttal round so proposers can revise or defend their position after receiving anonymized feedback.
- Selects the winner through pure-Python score and agreement arithmetic. The aggregator receives a verdict already fixed by code and generates prose only.
- Requires a greater-than-15% lead over the nearest disagreeing candidate for a clear winner and retains explicit dissent when the vote is split.
- Benchmarks single-shot, self-consistency, and debate arms against reference labels or ranges rather than another LLM judge.
- Checkpoints every arm attempt and supports `--resume`, preserving completed work across interrupted multi-hour runs.
- Normalizes a common local-model schema variation where `reasoning` arrives as a non-empty list of strings, joining the steps without paying for a repair call. Other malformed shapes still fail validation.

## Quick start

```bash
uv sync
ollama pull qwen3.5:9b
curl -s http://localhost:11434/api/version

# Fast offline suite
uv run pytest

# One live debate
uv run pytest -m integration tests/test_integration.py -v
uv run debate ask "A bat and ball cost $1.10 total; the bat costs $1 more than the ball. What does the ball cost?" --live

# Resumable benchmark subset
uv run python bench/run_bench.py --subset --out bench/results/subset_run.json
uv run python bench/run_bench.py --subset --resume --out bench/results/subset_run.json
```

Open [`debateRender.html`](debateRender.html) for a static rendering of the committed live transcript.

## Interpretation boundary

Three copies of one model are **correlated samples**, not independent experts. Shared pretraining, prompting conventions, and failure modes can produce confident consensus around the same wrong answer. Anonymization reduces one source of critic bias; it does not make the critic objective. Agreement bonuses and deterministic voting make the mechanism reproducible, not necessarily correct.

The committed benchmark is a small, interrupted subset with unequal completion counts across arms. Its completed debate cases should not be compared naively with baseline accuracy because long single-shot generations timed out more often. The recorded 7.97× token multiple is a real cost observation for that run; it is not evidence that debate has a favorable quality-per-dollar trade-off in general.

## Learnings

- **The framework name in the original plan was ambiguous.** The `ag2` package on PyPI was unrelated to the expected classic AutoGen API, while current Microsoft AutoGen had moved to a different actor-style architecture. A hand-rolled pipeline gave tighter control over anonymization, rebuttal construction, and deterministic voting.
- **A tied score is not automatically a split.** A tie between proposals with the same answer is consensus; only tied top scores attached to different answers indicate disagreement.
- **Per-call timeouts biased the comparison.** Long baseline completions exceeded the ceiling more often than debate's larger number of shorter calls. Completed-case accuracy therefore has survivor bias.
- **Retry layers multiply.** Transport retries, schema repairs, and self-consistency samples can create a much larger worst-case wall time than any individual timeout suggests. A shared logical-call deadline remains follow-up work.
- **A repeated schema failure is now handled without hiding the evidence.** The recorded run contains four failures where `Proposal.reasoning` was a JSON array. The message contract now losslessly joins non-empty string arrays and tests that the first attempt succeeds, but the historical benchmark file has not been rewritten or retroactively rescored.
- **Checkpointing paid off.** Two external process terminations and one operator cap lost only the in-flight attempt because every completed arm was persisted.

See [`readme.html`](readme.html) for the full methodology, benchmark table, cost analysis, and caveats.
