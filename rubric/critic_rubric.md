# Critic Scoring Rubric — v1

The critic scores every proposal on three dimensions, each **1-10, where 10
is always best**. This direction is normalized across all three dimensions
on purpose: `voting.py` sums them into a single 3-30 score with no sign
flips, which keeps the vote math auditable by inspection.

## Dimensions

### correctness_risk (1-10, 10 = lowest risk of being wrong)
Independently work through whether the proposal's answer is actually
correct — do not just trust the proposer's stated confidence. A confident,
well-written proposal that gets the arithmetic or logic wrong scores **low**
here, not high. Score:
- 9-10: you verified the answer yourself and it holds up
- 6-8: plausible, minor unverified assumption
- 3-5: a specific error or unjustified leap you can point to
- 1-2: clearly wrong, or contradicts the question's own constraints

### completeness (1-10, 10 = fully addresses the question)
Does the proposal answer what was actually asked, with the caveats,
ranges, or edge cases the question calls for? A technically-correct answer
that ignores part of the question scores low here even if correctness_risk
is high.

### reasoning_quality (1-10, 10 = sound and well-justified)
Independent of whether the final answer is right: is the reasoning chain
itself valid? Are the steps justified, or does the proposal jump to a
conclusion? A proposal can have a lucky-correct answer with poor reasoning
(scores low here despite high correctness_risk) — flag that explicitly in
the critique text, since the rebuttal round is where this gets caught.

## Anonymization

The critic only ever sees proposals labeled by letter (`A`, `B`, `C`),
never by persona name. This is a bias control: knowing "this is the
Creative persona's answer" primes the critic to expect (and forgive)
unconventional reasoning, or conversely to over-discount it. The critic
role never generates proposals of its own — role separation is enforced in
code (`critic.py` has no code path that calls a proposer persona).

## Output contract

One critic call scores *all* proposals in a round together (not one call
per proposal) so the critique is comparative and consistent — "A is more
complete than B" is a judgment the critic can only make by seeing both at
once. Output is a `CriticOutput` (see `messages.py`): one `Critique` per
anonymized proposal, each with the three integer scores above and a short
written justification.

## Versioning

Bump this file's version header and note the change below whenever the
rubric text changes — `voting.py`'s score math assumes the 1-10 / higher-is-
better convention above, so a rubric change that flips a dimension's
direction requires a matching code change.

- v1 (initial): three dimensions above, all 1-10 higher-is-better.
