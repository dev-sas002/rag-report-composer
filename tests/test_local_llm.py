"""
Tests for the offline report writer.

Its defining property is that it never invents anything: every sentence it
emits must already be present in the retrieved context. A generative stub would
be easier to write and would quietly teach the opposite lesson to the one this
project exists to demonstrate.
"""

import pytest
from langchain.schema import Document

from src.agent.context import assemble_context
from src.agent.local_llm import ExtractiveReportWriter
from src.agent.nodes import build_report_prompt, build_summary_prompt

CONTEXT = (
    "Revenue grew twenty two percent year over year across the group. "
    "The northern region contributed most of the enterprise renewals. "
    "Headcount rose by forty people during the same period. "
    "Short line."
)


def prompt(query: str, context: str = CONTEXT) -> str:
    return f"Question: {query}\n\nContext:\n{context}"


@pytest.fixture
def writer():
    return ExtractiveReportWriter()


class TestGrounding:
    def test_every_quoted_sentence_comes_from_the_context(self, writer):
        report = writer.invoke(prompt("what happened to revenue?")).content
        for line in report.splitlines():
            if line.startswith("- "):
                assert line[2:].strip() in CONTEXT

    def test_an_unrelated_query_still_yields_only_sourced_sentences(self, writer):
        # The heading echoes the question, which is not invention. What matters
        # is that the body never contains a sentence the context does not have.
        report = writer.invoke(prompt("what about lunar manufacturing?")).content
        quoted = [line[2:].strip() for line in report.splitlines() if line.startswith("- ")]
        assert quoted, "expected the writer to quote something"
        for sentence in quoted:
            assert sentence in CONTEXT

    def test_empty_context_is_reported_rather_than_filled_in(self, writer):
        report = writer.invoke(prompt("revenue", context="")).content
        assert "nothing to report" in report.lower()

    def test_context_of_only_fragments_is_reported_honestly(self, writer):
        report = writer.invoke(prompt("revenue", context="Too short. Also short.")).content
        assert "no passages long enough" in report.lower()


class TestSelection:
    def test_it_prefers_sentences_that_share_words_with_the_question(self, writer):
        report = writer.invoke(prompt("how did headcount change?")).content
        assert "Headcount rose by forty people" in report

    def test_it_skips_fragments_below_the_length_floor(self, writer):
        report = writer.invoke(prompt("revenue")).content
        assert "Short line." not in report

    def test_quoted_passages_keep_their_original_order(self, writer):
        report = writer.invoke(prompt("revenue renewals headcount")).content
        positions = [
            report.index(s)
            for s in [
                "Revenue grew twenty two percent",
                "The northern region contributed",
                "Headcount rose by forty",
            ]
            if s in report
        ]
        assert positions == sorted(positions)


class TestPromptScaffolding:
    """
    The real prompt wraps the context in a role preamble, source separators and
    a numbered instruction list. Quoting any of that back as source material is
    a regression, and it is one that happened.
    """

    # Built from the real assembler and the real prompt builder rather than
    # hand-written, so this fixture cannot drift out of date the way the
    # previous hard-coded "--- Source 1: ... ---" string did.
    REAL_PROMPT = build_report_prompt(
        query="How much did revenue grow?",
        context=assemble_context(
            [
                (
                    Document(
                        page_content=(
                            "Total revenue for the financial year reached $48.2 "
                            "million, an increase of twenty-two percent on the "
                            "prior year."
                        ),
                        metadata={"source": "/app/data/raw_data/annual_report_2025.md"},
                    ),
                    0.11,
                )
            ],
            max_chars=12000,
        ).text,
        max_report_length=2000,
    )

    def test_it_does_not_quote_the_instruction_block(self, writer):
        report = writer.invoke(self.REAL_PROMPT).content
        assert "Analyze the provided context" not in report
        assert "Instructions:" not in report

    def test_it_does_not_quote_the_query_line_as_a_finding(self, writer):
        report = writer.invoke(self.REAL_PROMPT).content
        quoted = [line[2:] for line in report.splitlines() if line.startswith("- ")]
        assert not any(q.startswith("Query:") for q in quoted)

    def test_it_strips_the_source_separators(self, writer):
        report = writer.invoke(self.REAL_PROMPT).content
        assert "annual_report_2025.md ---" not in report
        assert "---\n\nTotal revenue" not in report

    def test_it_attributes_each_quotation_to_the_source_it_came_from(self, writer):
        """
        The separator is not merely stripped, it is read: the marker on a
        quoted line is the citation id of the passage that line is in.
        """
        report = writer.invoke(self.REAL_PROMPT).content
        quoted = [line for line in report.splitlines() if line.startswith("- ")]

        assert quoted
        assert all(line.rstrip().endswith("[S1]") for line in quoted)

    def test_it_still_finds_the_actual_content(self, writer):
        report = writer.invoke(self.REAL_PROMPT).content
        assert "48.2 million" in report


class TestSummaryDetection:
    """
    Which of the two prompts arrived has to be decided from the prompt's own
    preamble. Searching the whole prompt for "summar" mistook a *report*
    request whose question happened to contain the word — `--query "Summarize
    Q4 performance"` — for a summary request, and returned a bare paragraph
    where a report was asked for.
    """

    def test_the_real_summary_prompt_is_recognised(self, writer):
        summary = writer.invoke(build_summary_prompt(CONTEXT)).content
        assert not summary.lstrip().startswith("##")
        assert "\n- " not in summary

    def test_a_report_query_containing_the_word_summarize_still_gets_a_report(self, writer):
        report = writer.invoke(
            build_report_prompt(
                query="Summarize Q4 performance",
                context=CONTEXT,
                max_report_length=2000,
            )
        ).content
        assert report.lstrip().startswith("##")
        assert "- " in report

    def test_the_summary_does_not_quote_the_writers_own_boilerplate(self, writer):
        report = writer.invoke(build_report_prompt("revenue", CONTEXT, 2000)).content
        summary = writer.invoke(build_summary_prompt(report)).content

        assert "reproduced verbatim" not in summary
        assert "extractive writer" not in summary
        assert "Revenue grew twenty two percent" in summary

    def test_the_summary_prompt_header_is_not_quoted_as_source(self, writer):
        summary = writer.invoke(build_summary_prompt(CONTEXT)).content
        assert "executive summary" not in summary.lower()


class TestOutputShape:
    def test_it_labels_itself_as_offline(self, writer):
        report = writer.invoke(prompt("revenue")).content
        assert "offline" in report.lower()

    def test_the_response_exposes_content_like_a_chat_model(self, writer):
        assert isinstance(writer.invoke(prompt("revenue")).content, str)
