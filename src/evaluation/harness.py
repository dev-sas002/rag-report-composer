"""
Scored evaluation of the retrieval pipeline.

A RAG system that nobody measures is a guess. Swapping the embedding model,
changing the chunk size or editing a prompt all move answer quality, and
without a score the only observable signal is whether the process still runs.

Every metric here is deterministic. No model judges another model's output —
an LLM-as-judge would let the system grade its own homework and would make the
score drift between runs for reasons unrelated to any change in the code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, List, Optional, Sequence

from .grounding import groundedness

# Standard-library logging on purpose. The harness is deliberately importable
# on its own: pulling in the project's structlog setup would drag the whole
# dependency tree — tiktoken, chromadb, torch — into a module whose entire job
# is arithmetic over lists of documents, and would mean the scores could not be
# tested without installing the world.
logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """One scored question."""

    id: str
    query: str
    #: Substrings identifying the documents that should be retrieved. A case
    #: passes on retrieval when all of them appear among the retrieved sources.
    expect_sources: List[str] = field(default_factory=list)
    #: Substrings the report is expected to mention, case-insensitively.
    expect_mentions: List[str] = field(default_factory=list)


@dataclass
class CaseResult:
    id: str
    query: str
    passed: bool
    failures: List[str]
    recall: float
    precision: float
    faithfulness: float
    latency_ms: int
    retrieved: int


@dataclass
class EvalReport:
    backend: str
    total: int
    passed: int
    mean_recall: float
    mean_precision: float
    mean_faithfulness: float
    p95_latency_ms: int
    cases: List[CaseResult]

    def format(self) -> str:
        return (
            f"backend={self.backend} pass={self.passed}/{self.total} "
            f"recall={self.mean_recall:.2f} precision={self.mean_precision:.2f} "
            f"faithfulness={self.mean_faithfulness:.2f} "
            f"p95={self.p95_latency_ms}ms"
        )


class RetrievalEvaluator:
    """
    Scores retrieval and grounding for a set of questions.

    `retrieve` takes a query and returns documents with `page_content` and
    `metadata`; `generate` takes a query and returns report text. Both are
    passed in rather than constructed, so the harness can score the real
    pipeline, a stubbed one, or two configurations against each other.
    """

    def __init__(
        self,
        retrieve: Callable[[str], Sequence],
        generate: Optional[Callable[[str], str]] = None,
        backend: str = "unknown",
    ) -> None:
        self.retrieve = retrieve
        self.generate = generate
        self.backend = backend

    def run(self, cases: Sequence[EvalCase]) -> EvalReport:
        results = [self._run_case(case) for case in cases]
        latencies = sorted(r.latency_ms for r in results)

        return EvalReport(
            backend=self.backend,
            total=len(results),
            passed=sum(1 for r in results if r.passed),
            mean_recall=round(mean([r.recall for r in results]), 4) if results else 0.0,
            mean_precision=(round(mean([r.precision for r in results]), 4) if results else 0.0),
            mean_faithfulness=(
                round(mean([r.faithfulness for r in results]), 4) if results else 0.0
            ),
            p95_latency_ms=self._percentile(latencies, 0.95),
            cases=results,
        )

    # -- one case --------------------------------------------------------

    def _run_case(self, case: EvalCase) -> CaseResult:
        started = time.perf_counter()
        failures: List[str] = []

        documents = list(self.retrieve(case.query))
        sources = [self._source_of(d) for d in documents]

        # Recall: of the documents that should have surfaced, how many did.
        hits = [
            expected
            for expected in case.expect_sources
            if any(expected.lower() in s.lower() for s in sources)
        ]
        recall = 1.0 if not case.expect_sources else len(hits) / len(case.expect_sources)
        for expected in case.expect_sources:
            if not any(expected.lower() in s.lower() for s in sources):
                failures.append(f'expected a chunk from "{expected}", got {sources or "nothing"}')

        # Precision: what share of what came back was actually wanted. Only
        # meaningful when the case declares its expectations.
        if case.expect_sources and documents:
            relevant = sum(
                1 for s in sources if any(e.lower() in s.lower() for e in case.expect_sources)
            )
            precision = relevant / len(documents)
        else:
            precision = 1.0 if documents else 0.0

        report = ""
        faithfulness = 1.0
        if self.generate is not None:
            report = self.generate(case.query)
            faithfulness = self._faithfulness(report, documents)
            for needle in case.expect_mentions:
                if needle.lower() not in report.lower():
                    failures.append(f'report did not mention "{needle}"')

        latency_ms = int((time.perf_counter() - started) * 1000)
        return CaseResult(
            id=case.id,
            query=case.query,
            passed=not failures,
            failures=failures,
            recall=round(recall, 4),
            precision=round(precision, 4),
            faithfulness=round(faithfulness, 4),
            latency_ms=latency_ms,
            retrieved=len(documents),
        )

    @staticmethod
    def _faithfulness(report: str, documents: Sequence) -> float:
        """
        Share of the report's content words that appear in the retrieved text.

        Delegated to `grounding.groundedness`, which is also what the graph
        attaches to every report it produces. One implementation, so a score a
        user is shown and a score CI prints cannot drift apart — two numbers
        with the same name computed two ways is worse than having only one.
        """
        return groundedness(report, [getattr(d, "page_content", "") for d in documents])

    @staticmethod
    def _source_of(document) -> str:
        metadata = getattr(document, "metadata", {}) or {}
        return str(
            metadata.get("source") or metadata.get("file_name") or metadata.get("filename") or ""
        )

    @staticmethod
    def _percentile(sorted_values: List[int], p: float) -> int:
        if not sorted_values:
            return 0
        index = min(len(sorted_values) - 1, max(0, int(round(p * len(sorted_values))) - 1))
        return sorted_values[index]
