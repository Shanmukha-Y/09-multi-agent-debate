"""Multi-Agent Debate System.

Three same-model personas propose, a rubric critic scores them anonymized,
a rebuttal round lets proposers accept or rebut criticism, a deterministic
vote picks a winner (or flags a split), and an aggregator reports the final
answer with a confidence score and an honest dissent appendix.
"""

from __future__ import annotations

__version__ = "0.1.0"
