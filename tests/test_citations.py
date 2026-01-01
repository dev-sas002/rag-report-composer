"""
Tests for citation checking and grounding.

A report that names its sources is only better than one that does not if the
names are real. Both silent failures are covered here: a marker that points at
a source never supplied, and prose that drifts onto subjects the sources never
mentioned.
"""

import pytest

from src.agent.citations import audit_report, extract_citation_ids
from src.agent.context import Citation
from src.evaluation.grounding import content_words, groundedness, ungrounded_words

REVENUE = Citation(
    id="S1",
    source="annual_report.md",
    text="Revenue reached forty eight million, up twenty two percent on enterprise renewals.",
)
SATISFACTION = Citation(
    id="S2",
    source="customer_satisfaction.md",
    text="Net Promoter Score rose to forty one, with reporting named as the top complaint.",
)


class TestMarkerExtraction:
    def test_markers_are_returned_in_first_appearance_order(self):
        assert extract_citation_ids("a [S2] b [S1] c") == ["S2", "S1"]

    def test_a_repeated_marker_is_listed_once(self):
        assert extract_citation_ids("[S1] and again [S1]") == ["S1"]

    def test_text_with_no_markers_yields_nothing(self):
        assert extract_citation_ids("a report with no citations at all") == []

    def test_empty_text_does_not_raise(self):
        assert extract_citation_ids("") == []


class TestSuppliedVersusUsed:
    def test_a_marker_that_was_supplied_counts_as_valid(self):
        audit = audit_report("Revenue rose [S1].", [REVENUE, SATISFACTION])

        assert audit.valid == ["S1"]
        assert audit.invalid == []

    def test_a_marker_that_was_never_supplied_is_flagged_as_invented(self):
        """
        The failure this catches is silent: the output still looks like a
        sourced report, the marker just points at nothing.
        """
        audit = audit_report("Revenue rose [S7].", [REVENUE])

        assert audit.invalid == ["S7"]
        assert audit.has_invented_citations

    def test_supplied_sources_the_report_never_used_are_reported_separately(self):
        """
        Not an error — it means the prompt carried context the answer did not
        need, which is a cost signal rather than a correctness one.
        """
        audit = audit_report("Revenue rose [S1].", [REVENUE, SATISFACTION])

        assert audit.unused == ["S2"]
        assert not audit.has_invented_citations

    def test_a_report_with_no_citations_at_all_is_not_an_invention(self):
        audit = audit_report("Revenue rose.", [REVENUE])

        assert audit.valid == []
        assert audit.invalid == []
        assert audit.unused == ["S1"]


class TestGroundedness:
    def test_a_report_quoting_its_sources_scores_high(self):
        audit = audit_report(
            "Revenue reached forty eight million on enterprise renewals [S1].", [REVENUE]
        )

        assert audit.groundedness > 0.9

    def test_a_report_inventing_content_scores_low(self):
        audit = audit_report(
            "The lunar titanium division tripled output after the Antarctic merger [S1].",
            [REVENUE],
        )

        assert audit.groundedness < 0.4

    def test_the_words_that_caused_a_low_score_are_named(self):
        """The score says how bad it is; debugging needs to know which words."""
        audit = audit_report("Antarctic titanium production [S1].", [REVENUE])

        assert "antarctic" in audit.ungrounded_sample
        assert "titanium" in audit.ungrounded_sample

    def test_grounding_is_measured_against_the_prompt_not_the_corpus(self):
        """
        Scoring against everything indexed would give credit for a word the
        model could not have seen.
        """
        score = groundedness("Net Promoter Score rose", [REVENUE.text])

        assert score < 0.5

    def test_the_question_echoed_in_a_heading_is_neither_grounded_nor_invented(self):
        """
        Reports open with the question as a heading. Counting the user's own
        words against the report scored a perfectly faithful extract at 0.84.
        """
        report = "## How much did revenue grow\n\nRevenue reached forty eight million [S1]."

        without = audit_report(report, [REVENUE])
        with_query = audit_report(report, [REVENUE], query="How much did revenue grow?")

        assert with_query.groundedness > without.groundedness
        assert with_query.groundedness == pytest.approx(1.0)

    def test_the_pipelines_own_boilerplate_does_not_count_against_it(self):
        report = (
            "The following passages were retrieved from the indexed documents "
            "and are reproduced verbatim. Revenue reached forty eight million [S1]."
        )

        assert audit_report(report, [REVENUE]).groundedness == pytest.approx(1.0)

    def test_nothing_to_check_against_scores_zero_rather_than_one(self):
        """An unscored report must not look like a perfect one."""
        assert groundedness("Revenue rose.", []) == 0.0
        assert groundedness("   ", [REVENUE.text]) == 0.0

    def test_a_report_of_pure_scaffolding_is_vacuously_grounded(self):
        assert groundedness("Executive summary. Findings.", [REVENUE.text]) == 1.0


class TestWordExtraction:
    def test_short_tokens_are_ignored(self):
        """Two-letter words carry no signal and inflate the denominator."""
        assert content_words("an ox is in the way") == {"the", "way"}

    def test_case_and_punctuation_do_not_create_separate_words(self):
        assert content_words("Revenue, revenue; REVENUE!") == {"revenue"}

    def test_ungrounded_words_excludes_scaffolding_as_well_as_sources(self):
        missing = ungrounded_words("verbatim antarctic", [REVENUE.text])

        assert missing == {"antarctic"}


class TestSerialisation:
    def test_the_audit_serialises_for_an_api_response(self):
        payload = audit_report("Revenue rose [S1] and [S9].", [REVENUE]).as_dict()

        assert payload["valid"] == ["S1"]
        assert payload["invalid"] == ["S9"]
        assert isinstance(payload["groundedness"], float)
