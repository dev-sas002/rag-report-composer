"""
Deterministic grounding measurement.

One implementation, used in two places: the evaluation harness scores a fixed
question set with it, and the graph attaches the same number to every report it
produces. They must not drift — a groundedness score shown to a user and a
groundedness score printed by CI that were computed differently is worse than
having only one of them.

Standard library only, on purpose. This module is imported by
`src/evaluation/harness.py`, which is deliberately installable-free so the
metrics can be tested without chromadb, torch and tiktoken.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

#: Words shorter than this carry no signal and inflate the denominator.
MIN_WORD_CHARS = 3

#: Prose the pipeline writes about itself. It is not a claim about the source
#: material, so counting it as unsupported would penalise a report for its own
#: scaffolding.
SCAFFOLDING = (
    "the following passages were retrieved from indexed documents and "
    "are reproduced verbatim written offline by extractive writer every "
    "line above appears in source material configure openai api key for "
    "synthesised narrative report question answer summary context "
    "executive introduction conclusion overview findings section sources"
)


def content_words(text: str) -> set[str]:
    """Lowercase alphanumeric tokens long enough to mean something."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= MIN_WORD_CHARS}


def _claimed_words(report: str, ignore: Optional[str]) -> set[str]:
    """
    The report's words that are actually claims about the source material.

    Two categories are removed. The pipeline's own scaffolding, because it is
    prose about the report rather than a statement about the documents. And the
    user's question, when it is supplied: reports echo it in a heading, and
    echoing the question the user asked is neither grounded nor invented — it
    is the prompt. Counting it made a perfectly faithful report score 0.84,
    with every missing word a word the user had typed.
    """
    excluded = content_words(SCAFFOLDING)
    if ignore:
        excluded |= content_words(ignore)
    return content_words(report) - excluded


def groundedness(report: str, source_texts: Iterable[str], ignore: Optional[str] = None) -> float:
    """
    Share of the report's content words that appear in the retrieved text.

    A blunt instrument, and it is meant to be: it cannot tell a paraphrase from
    an invention, so it is a floor rather than a verdict. What it does catch is
    the failure that matters — a report drifting onto subjects the sources never
    mentioned, which shows up as the score falling.

    `ignore` is text whose words count as neither grounded nor invented; the
    caller passes the query.

    Returns 0.0 when there is nothing to check against, and 1.0 when the report
    contains nothing but scaffolding.
    """
    sources = list(source_texts)
    if not report.strip() or not sources:
        return 0.0

    source_vocabulary: set[str] = set()
    for text in sources:
        source_vocabulary |= content_words(text)

    claimed = _claimed_words(report, ignore)
    if not claimed:
        return 1.0

    return len(claimed & source_vocabulary) / len(claimed)


def ungrounded_words(
    report: str, source_texts: Iterable[str], ignore: Optional[str] = None
) -> set[str]:
    """
    The content words a report uses that no source contains.

    The score says how bad it is; this says which words caused it, which is
    what someone debugging a low score actually needs.
    """
    source_vocabulary: set[str] = set()
    for text in source_texts:
        source_vocabulary |= content_words(text)

    return _claimed_words(report, ignore) - source_vocabulary
