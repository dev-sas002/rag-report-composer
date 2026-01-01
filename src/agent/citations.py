"""
Citation verification.

A report that names its sources is only better than one that does not if the
names are real. A model asked to cite `[S1]`-style markers will occasionally
cite `[S7]` when six sources were supplied, or cite nothing at all and write
confidently instead. Both are silent failures: the output still looks like a
sourced report.

So the markers are checked against the citations that were actually put in the
prompt, and what the check finds travels with the report rather than being
logged and forgotten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from ..evaluation.grounding import groundedness, ungrounded_words
from .context import Citation

CITATION_RE = re.compile(r"\[(S\d+)\]")


@dataclass
class CitationAudit:
    """What the report cited, and whether it was entitled to."""

    #: Citation ids the report used that were supplied in the prompt.
    valid: List[str]
    #: Citation ids the report used that were never supplied — invented.
    invalid: List[str]
    #: Supplied citations the report never used. Not an error; it means the
    #: prompt carried context the answer did not need, which is a cost signal.
    unused: List[str]
    #: Share of the report's content words present in the cited text.
    groundedness: float
    #: A sample of the words that were not, for debugging a low score.
    ungrounded_sample: List[str]

    @property
    def has_invented_citations(self) -> bool:
        return bool(self.invalid)

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "invalid": self.invalid,
            "unused": self.unused,
            "groundedness": round(self.groundedness, 4),
            "ungrounded_sample": self.ungrounded_sample,
        }


def extract_citation_ids(text: str) -> List[str]:
    """Citation ids in the order they first appear, without duplicates."""
    seen: List[str] = []
    for match in CITATION_RE.findall(text or ""):
        if match not in seen:
            seen.append(match)
    return seen


def audit_report(
    report: str,
    citations: Sequence[Citation],
    query: str = "",
    sample_size: int = 12,
) -> CitationAudit:
    """
    Check a report's citation markers and score it against its own sources.

    Grounding is measured against the citation text — the chunks that were
    actually assembled into the prompt — not against the whole corpus. Scoring
    against everything indexed would give credit for a word the model could not
    have seen.

    `query` is excluded from the score because reports echo the question in a
    heading, and the user's own words are neither grounded nor invented.
    """
    supplied = [c.id for c in citations]
    used = extract_citation_ids(report)

    valid = [cid for cid in used if cid in supplied]
    invalid = [cid for cid in used if cid not in supplied]
    unused = [cid for cid in supplied if cid not in used]

    source_texts = [c.text for c in citations]
    score = groundedness(report, source_texts, ignore=query)
    missing = sorted(ungrounded_words(report, source_texts, ignore=query))[:sample_size]

    return CitationAudit(
        valid=valid,
        invalid=invalid,
        unused=unused,
        groundedness=score,
        ungrounded_sample=missing,
    )
