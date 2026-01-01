"""
Tests for the evaluation harness.

The harness is the thing that tells us whether a change to chunking, the
embedding model or a prompt made retrieval better or worse, so its own scoring
has to be trustworthy before any number it produces means anything.
"""

from dataclasses import dataclass, field
from typing import List

from src.evaluation import EvalCase, RetrievalEvaluator


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict = field(default_factory=dict)


REVENUE = FakeDoc(
    "Revenue grew twenty two percent year over year, driven by enterprise "
    "renewals across the northern region.",
    {"source": "annual_report.pdf"},
)
HIRING = FakeDoc(
    "Headcount increased by forty people, mostly in engineering and support.",
    {"source": "hr_summary.docx"},
)


def evaluator(docs: List[FakeDoc], report: str = "", backend: str = "test"):
    return RetrievalEvaluator(
        retrieve=lambda _q: docs,
        generate=(lambda _q: report) if report else None,
        backend=backend,
    )


class TestRecall:
    def test_full_recall_when_expected_source_is_retrieved(self):
        report = evaluator([REVENUE]).run(
            [EvalCase(id="r1", query="revenue", expect_sources=["annual_report"])]
        )
        assert report.mean_recall == 1.0
        assert report.passed == 1

    def test_zero_recall_when_the_expected_source_is_missing(self):
        report = evaluator([HIRING]).run(
            [EvalCase(id="r2", query="revenue", expect_sources=["annual_report"])]
        )
        assert report.mean_recall == 0.0
        assert report.passed == 0
        assert "annual_report" in report.cases[0].failures[0]

    def test_partial_recall_is_reported_as_a_fraction(self):
        report = evaluator([REVENUE]).run(
            [
                EvalCase(
                    id="r3",
                    query="everything",
                    expect_sources=["annual_report", "hr_summary"],
                )
            ]
        )
        assert report.mean_recall == 0.5


class TestPrecision:
    def test_precision_falls_when_irrelevant_documents_come_back(self):
        # One of two retrieved documents was wanted.
        report = evaluator([REVENUE, HIRING]).run(
            [EvalCase(id="p1", query="revenue", expect_sources=["annual_report"])]
        )
        assert report.mean_precision == 0.5

    def test_precision_is_zero_when_nothing_is_retrieved(self):
        report = evaluator([]).run([EvalCase(id="p2", query="revenue")])
        assert report.mean_precision == 0.0


class TestFaithfulness:
    def test_a_report_quoting_the_source_scores_high(self):
        report = evaluator([REVENUE], report="Revenue grew twenty two percent year over year.").run(
            [EvalCase(id="f1", query="revenue")]
        )
        assert report.mean_faithfulness > 0.9

    def test_a_report_inventing_content_scores_low(self):
        # This is the failure the metric exists to catch: confident prose about
        # subjects the retrieved documents never mentioned.
        report = evaluator(
            [REVENUE],
            report=(
                "The lunar manufacturing division tripled its titanium output "
                "following the Antarctic partnership announcement."
            ),
        ).run([EvalCase(id="f2", query="revenue")])
        assert report.mean_faithfulness < 0.4

    def test_an_empty_report_is_not_treated_as_faithful(self):
        report = evaluator([REVENUE], report="   ").run([EvalCase(id="f3", query="revenue")])
        assert report.mean_faithfulness == 0.0


class TestExpectedMentions:
    def test_a_missing_mention_fails_the_case(self):
        report = evaluator([REVENUE], report="Revenue grew.").run(
            [EvalCase(id="m1", query="revenue", expect_mentions=["enterprise"])]
        )
        assert report.passed == 0
        assert "enterprise" in report.cases[0].failures[0]

    def test_a_present_mention_passes(self):
        report = evaluator([REVENUE], report="Revenue grew, driven by enterprise renewals.").run(
            [EvalCase(id="m2", query="revenue", expect_mentions=["enterprise"])]
        )
        assert report.passed == 1


class TestReportShape:
    def test_the_summary_line_carries_the_backend_and_the_scores(self):
        line = (
            evaluator([REVENUE], backend="local")
            .run([EvalCase(id="s1", query="revenue", expect_sources=["annual_report"])])
            .format()
        )
        assert "backend=local" in line
        assert "recall=1.00" in line

    def test_an_empty_case_list_does_not_divide_by_zero(self):
        report = evaluator([REVENUE]).run([])
        assert report.total == 0
        assert report.mean_recall == 0.0
