"""
Offline report writer, used when no OpenAI key is configured.

This is extractive, not generative: it selects and arranges sentences that are
already present in the retrieved documents. That is a deliberate choice rather
than a limitation to apologise for. A fake generative model would have to
invent prose, and inventing prose is exactly the failure a retrieval system
exists to prevent — an offline demo that hallucinated would be teaching the
wrong lesson about the system.

The practical consequence is that every sentence in an offline report is
verifiably present in the source, so the evaluation harness scores its
faithfulness at 1.0 by construction. Retrieval quality is what is actually
being measured offline; writing quality needs a real model.

Each quoted sentence carries the `[S1]`-style marker of the source it was
taken from, exactly as a real model is asked to. That keeps the citation audit
on the default, keyless path rather than only on the path that costs money.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from .context import SOURCE_HEADER_RE

# Standard-library logging, so this module stays importable — and testable —
# without the project's structlog stack and everything behind it.
logger = logging.getLogger(__name__)

#: One quoted sentence and the citation id it came from ("" when unknown).
AttributedSentence = Tuple[str, str]

#: A bullet line in a report this writer produced, and the marker it ends with.
BULLET_LINE_RE = re.compile(r"(?m)^\s*[-*•]\s+(.*\S)\s*$")
TRAILING_CITATION_RE = re.compile(r"\s*\[(S\d+)\]\s*$")

# Sentences shorter than this are usually headings or fragments and read badly
# in a report body.
MIN_SENTENCE_CHARS = 40
MAX_SENTENCES_PER_SECTION = 4

# The opening words of `build_summary_prompt`. Matching on the prompt's own
# preamble is the only reliable signal: searching the whole prompt for
# "summar" classified `--query "Summarize Q4 performance"` as a summary
# request, because the query is echoed into the report prompt.
SUMMARY_PROMPT_PREFIX = "create a concise executive summary"

# The sentence `_report` writes above the quoted passages. When the summary
# step is handed an offline report to condense, this is the writer's own prose
# rather than source material, and summarising it says nothing.
REPORT_PREAMBLE = (
    "The following passages were retrieved from the indexed documents "
    "and are reproduced verbatim."
)


class LocalResponse:
    """Mirrors the shape LangChain chat models return, so callers are unchanged."""

    def __init__(self, content: str) -> None:
        self.content = content


class ExtractiveReportWriter:
    """
    Composes a report from the retrieved context.

    Implements the same `invoke(prompt) -> object with .content` contract as the
    chat models this project uses, so `ReportNodes` needs no branch: it is
    handed a writer and calls it.
    """

    def __init__(self, max_sentences: int = MAX_SENTENCES_PER_SECTION) -> None:
        self.max_sentences = max_sentences

    def invoke(self, prompt: str) -> LocalResponse:
        is_summary = self._is_summary_prompt(prompt)
        query, segments = self._split_prompt(prompt, is_summary=is_summary)

        if not any(text.strip() for _, text in segments):
            return LocalResponse(
                "No source material was retrieved for this question, so there "
                "is nothing to report. Ingest documents first, or widen the query."
            )

        sentences = self._rank_sentences(segments, query)
        if not sentences:
            return LocalResponse(
                "The retrieved documents contained no passages long enough to "
                "quote. Try a broader query."
            )

        if is_summary:
            return LocalResponse(self._summary(sentences))
        return LocalResponse(self._report(query, sentences))

    # -- composition -----------------------------------------------------

    @staticmethod
    def _cite(sentence: str, citation_id: str) -> str:
        """Append the source marker, unless the sentence already carries one."""
        if not citation_id or f"[{citation_id}]" in sentence:
            return sentence
        return f"{sentence} [{citation_id}]"

    def _report(self, query: str, sentences: List[AttributedSentence]) -> str:
        body = "\n\n".join(
            f"- {self._cite(sentence, cid)}"
            for sentence, cid in sentences[: self.max_sentences * 2]
        )
        return (
            f"## {query.strip().rstrip('?').capitalize()}\n\n"
            "The following passages were retrieved from the indexed documents "
            "and are reproduced verbatim.\n\n"
            f"{body}\n\n"
            "---\n\n"
            "*Written offline by the extractive writer: every line above appears "
            "in the source material. Configure `OPENAI_API_KEY` for a synthesised "
            "narrative report.*"
        )

    def _summary(self, sentences: List[AttributedSentence]) -> str:
        return " ".join(
            self._cite(sentence, cid) for sentence, cid in sentences[: self.max_sentences]
        )

    # -- prompt parsing --------------------------------------------------

    @staticmethod
    def _is_summary_prompt(prompt: str) -> bool:
        return prompt.lstrip().lower().startswith(SUMMARY_PROMPT_PREFIX)

    @staticmethod
    def _strip_own_scaffolding(text: str) -> str:
        """
        Remove the framing `_report` added, when a report comes back to be
        summarised. Condensing the writer's own boilerplate is not a summary.
        """
        text = re.sub(r"(?m)^\s*#{1,6}\s.*$", "", text)
        text = re.sub(r"(?s)\n\s*-{3,}\s*\n.*$", "", text)
        return text.replace(REPORT_PREAMBLE, "")

    # -- ranking ---------------------------------------------------------

    def _rank_sentences(
        self, segments: List[Tuple[str, str]], query: str
    ) -> List[AttributedSentence]:
        """
        Order sentences by overlap with the query, keeping document order.

        Each sentence keeps the citation id of the segment it came from, so the
        marker on a quoted line names the source that line is actually in.
        """
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}

        scored: list[tuple[int, int, str, str]] = []
        position = 0
        for citation_id, segment in segments:
            for raw in re.split(r"(?<=[.!?])\s+", segment):
                position += 1
                # Drop a leading list marker so a quoted bullet reads as prose
                # in the summary; the sentence itself is unchanged.
                sentence = re.sub(r"^[-*\u2022]\s+", "", " ".join(raw.split()))
                if len(sentence) < MIN_SENTENCE_CHARS:
                    continue
                words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                scored.append((len(terms & words), position, sentence, citation_id))

        if not scored:
            return []

        top = sorted(scored, key=lambda row: (-row[0], row[1]))[: self.max_sentences * 2]
        # Re-sort the winners into their original order so the report reads as
        # a passage rather than a ranked list.
        return [
            (sentence, citation_id)
            for _, _, sentence, citation_id in sorted(top, key=lambda row: row[1])
        ]

    def _split_prompt(
        self, prompt: str, is_summary: bool = False
    ) -> tuple[str, List[Tuple[str, str]]]:
        """
        Recover the question and the retrieved text out of the built prompt.

        The prompt wraps the context in scaffolding — a role preamble, a
        "Query:" line, source separators and a numbered instruction list. All
        of it has to be stripped, because quoting the instructions back as if
        they were source material is exactly the kind of confident nonsense
        this writer exists to avoid.

        The source separators are not merely stripped but used: splitting on
        them is what tells this writer which citation id each passage belongs
        to. Returns the query and a list of `(citation_id, text)` segments.
        """
        if is_summary:
            # The summary prompt is a header, the report, and a trailing
            # "Executive Summary:" cue. Only the middle is source material,
            # and it already carries whatever markers the report used.
            body = re.sub(
                r"^\s*create a concise executive summary[^\n]*\n*",
                "",
                prompt,
                flags=re.IGNORECASE,
            )
            body = re.sub(r"\n\s*executive summary\s*:\s*$", "", body, flags=re.IGNORECASE)
            return "the retrieved documents", self._segment_by_bullet(
                self._strip_own_scaffolding(body)
            )

        query = ""
        match = re.search(r"^\s*query\s*:\s*(.+)$", prompt, re.IGNORECASE | re.MULTILINE)
        if match:
            query = match.group(1).strip()

        context = prompt
        start = re.search(
            r"context from company documents\s*:\s*|^\s*context\s*:\s*",
            prompt,
            re.IGNORECASE | re.MULTILINE,
        )
        if start:
            context = prompt[start.end() :]

        # Drop everything from the instruction block onward.
        end = re.search(
            r"\n\s*(instructions\s*:|generate a comprehensive report)",
            context,
            re.IGNORECASE,
        )
        if end:
            context = context[: end.start()]

        return query or "the retrieved documents", self._segment_by_source(context)

    @staticmethod
    def _segment_by_source(context: str) -> List[Tuple[str, str]]:
        """
        Split assembled context into `(citation_id, text)` on its source headers.

        `re.split` with a capturing pattern yields the text before the first
        header, then id/body pairs. Text before the first header belongs to no
        source, so it is returned with an empty id and goes uncited rather than
        being attributed to whichever source happens to come next.
        """
        parts = SOURCE_HEADER_RE.split(context)
        segments: List[Tuple[str, str]] = []

        preamble = parts[0]
        if preamble.strip():
            segments.append(("", preamble))

        for index in range(1, len(parts) - 1, 2):
            segments.append((parts[index], parts[index + 1]))

        return segments or [("", context)]

    @staticmethod
    def _segment_by_bullet(report: str) -> List[Tuple[str, str]]:
        """
        Split a report being summarised into `(citation_id, text)` per bullet.

        A report this writer produced is a bullet list whose lines end in the
        marker of the source they came from. Treating the whole thing as one
        blob instead loses that: sentence splitting cuts after the full stop
        and before the marker, so the marker migrates onto the *next*
        sentence — a citation pointing at the wrong source, which is worse than
        no citation at all.

        A report from a real model is prose with markers inside the sentences.
        It has no bullets, so it comes back as a single uncited segment and its
        markers simply travel along inside the text.
        """
        bullets = BULLET_LINE_RE.findall(report)
        if not bullets:
            return [("", report)]

        segments: List[Tuple[str, str]] = []
        for line in bullets:
            match = TRAILING_CITATION_RE.search(line)
            if match:
                segments.append((match.group(1), TRAILING_CITATION_RE.sub("", line)))
            else:
                segments.append(("", line))
        return segments
