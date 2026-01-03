"""
Context assembly: turning retrieved chunks into the text a model is paid to read.

This is the most expensive line in a RAG system's bill and the least examined.
Every character assembled here is billed as an input token on every query, so
the job is not "concatenate the chunks" — it is "concatenate the chunks that
earn their place, once each, within a budget".

Three things happen, in this order, and each one is measured by
`scripts/benchmark.py`:

1. **Relevance pruning.** Retrieval returns exactly `k` chunks whether or not
   `k` chunks are relevant. A query answered by one paragraph still pays for
   four more. Chunks whose distance from the query is far worse than the best
   hit are dropped.
2. **Overlap de-duplication.** `RecursiveCharacterTextSplitter` deliberately
   overlaps adjacent chunks by `CHUNK_OVERLAP` characters so a sentence is not
   cut in half. When retrieval returns two adjacent chunks — which is common,
   because adjacent chunks are about the same thing — that overlap is sent to
   the model twice and paid for twice.
3. **Budgeting.** What survives is truncated to `MAX_CONTEXT_CHARS` rather than
   allowed to grow with `k` until the model rejects the request.

Each surviving chunk is labelled with a citation id (`[S1]`, `[S2]`, ...) that
the report is asked to cite and `src/agent/citations.py` later verifies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

#: Shortest overlap worth trimming. Below this, two chunks sharing a few
#: characters is coincidence (a shared "The ") rather than splitter overlap.
MIN_TRIMMABLE_OVERLAP = 24

#: Matches the header this module writes, capturing the citation id. The
#: offline writer splits the prompt on it, so it can attribute each quoted
#: sentence to the source it came from instead of quoting the header as if it
#: were source material.
SOURCE_HEADER_RE = re.compile(r"-{2,}\s*\[(S\d+)\][^\n]*?-{2,}")


def source_header(citation_id: int, source: str) -> str:
    """The one place the in-prompt source marker format is defined."""
    return f"--- [S{citation_id}] {source} ---"


@dataclass
class Citation:
    """A source the report is allowed to cite, and what it points at."""

    id: str
    source: str
    #: The chunk text as it was sent to the model, after trimming.
    text: str
    #: Retrieval distance, when the retriever supplied one. Lower is closer.
    score: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "score": self.score,
            "excerpt": self.text[:280],
        }


@dataclass
class ContextBundle:
    """The assembled context plus the accounting for how it got that way."""

    text: str = ""
    sources: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    truncated: bool = False
    #: Chunks dropped because their distance was far worse than the best hit.
    pruned_by_relevance: int = 0
    #: Chunks dropped because their text was already in the context.
    dropped_as_duplicate: int = 0
    #: Characters removed by trimming splitter overlap off a chunk's head.
    overlap_chars_trimmed: int = 0
    #: Characters the raw chunks would have contributed with none of the above.
    raw_chars: int = 0


def prune_by_relevance(
    scored: Sequence[Tuple[object, Optional[float]]], margin: float
) -> Tuple[List[Tuple[object, Optional[float]]], int]:
    """
    Keep the chunks close enough to the best hit to be worth paying for.

    Distances are compared relative to the best one rather than against an
    absolute threshold, because the absolute scale differs between embedding
    backends (and between distance metrics within one backend). A relative rule
    transfers; a hard-coded `0.8` does not.

    `margin` is a fraction: 0.35 keeps everything within 35% of the best
    distance. A margin of 0 disables pruning entirely. The best hit is always
    kept, so a query never ends up with no context because of this rule.
    """
    usable = [(doc, score) for doc, score in scored]
    if margin <= 0 or len(usable) <= 1:
        return usable, 0

    scores = [s for _, s in usable if isinstance(s, (int, float))]
    if len(scores) != len(usable):
        # A retriever that does not report distances cannot be pruned on them.
        return usable, 0

    best = min(scores)
    if best < 0:
        # Some backends report similarity (higher is better) rather than
        # distance. Pruning with the wrong polarity would drop the best hits,
        # so decline rather than guess.
        return usable, 0

    # `best` of exactly 0 means an identical match; allow an absolute slack so
    # the rule does not collapse to "keep only the one perfect chunk".
    ceiling = best * (1.0 + margin) if best > 0 else margin

    kept = [(doc, score) for doc, score in usable if score <= ceiling]
    if not kept:  # pragma: no cover - `best` always satisfies the ceiling
        kept = [usable[0]]
    return kept, len(usable) - len(kept)


def _trim_leading_overlap(previous: str, current: str, max_overlap: int) -> str:
    """
    Remove the head of `current` that repeats the tail of `previous`.

    Longest match wins, capped at `max_overlap` because that is the most the
    splitter can have duplicated; a longer match means the two chunks really do
    say the same thing and duplicate detection handles it.
    """
    limit = min(max_overlap, len(previous), len(current))
    for size in range(limit, MIN_TRIMMABLE_OVERLAP - 1, -1):
        if previous.endswith(current[:size]):
            return current[size:]
    return current


def assemble_context(
    scored_documents: Sequence[Tuple[object, Optional[float]]],
    *,
    max_chars: int,
    chunk_overlap: int = 0,
    relevance_margin: float = 0.0,
) -> ContextBundle:
    """
    Build the prompt context from retrieved `(document, score)` pairs.

    Ordering is preserved: the retriever returns its best match first and that
    order is meaningful to the model, so nothing here re-sorts.
    """
    bundle = ContextBundle()
    if not scored_documents:
        return bundle

    kept, bundle.pruned_by_relevance = prune_by_relevance(scored_documents, relevance_margin)

    parts: List[str] = []
    accepted_text_by_source: dict[str, List[str]] = {}
    used = 0

    for position, (document, score) in enumerate(kept, start=1):
        metadata = getattr(document, "metadata", None) or {}
        source = metadata.get("source") or f"Document {position}"
        content = getattr(document, "page_content", "") or ""
        bundle.raw_chars += len(content)

        seen = accepted_text_by_source.setdefault(source, [])

        # Whole-chunk duplicate: nothing new to pay for.
        if any(content.strip() and content in earlier for earlier in seen):
            bundle.dropped_as_duplicate += 1
            continue

        trimmed = content
        if chunk_overlap > 0:
            for earlier in seen:
                candidate = _trim_leading_overlap(earlier, trimmed, chunk_overlap)
                if candidate != trimmed:
                    bundle.overlap_chars_trimmed += len(trimmed) - len(candidate)
                    trimmed = candidate
                    break

        if not trimmed.strip():
            bundle.dropped_as_duplicate += 1
            continue

        citation_id = len(bundle.citations) + 1
        part = f"{source_header(citation_id, source)}\n{trimmed}\n"

        if used + len(part) > max_chars:
            bundle.truncated = True
            remaining = max_chars - used
            # Only keep a partial chunk if there is room for meaningfully more
            # than its header, otherwise drop it entirely.
            if remaining > len(part) // 4:
                part = part[:remaining]
            else:
                break

        parts.append(part)
        used += len(part)
        seen.append(content)
        bundle.sources.append(source)
        bundle.citations.append(
            Citation(
                id=f"S{citation_id}",
                source=source,
                text=trimmed,
                score=float(score) if isinstance(score, (int, float)) else None,
            )
        )

        if bundle.truncated:
            break

    bundle.text = "\n".join(parts)
    return bundle
