#!/usr/bin/env python
"""
Score the live retrieval pipeline against a fixed question set.

    python scripts/run_evaluation.py

Runs against whichever backend is configured, so the same command measures the
offline local model and a live OpenAI setup and prints comparable numbers.
Exits non-zero when a threshold is missed, which is what makes it usable as a
CI gate rather than a report nobody reads.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.evaluation import EvalCase, RetrievalEvaluator
from src.observability.logger import setup_logging
from src.retrieval.vector_store import VectorStore

# Every case is answerable from data/raw_data, and a human can verify each one
# by opening the file. A question set nobody can check by hand is not a test.
CASES = [
    EvalCase(
        id="revenue-growth",
        query="How much did revenue grow and which segment drove it?",
        expect_sources=["annual_report"],
        expect_mentions=["enterprise"],
    ),
    EvalCase(
        id="margin",
        query="What happened to gross margin and why?",
        expect_sources=["annual_report"],
    ),
    EvalCase(
        id="complaints",
        query="What do customers complain about most?",
        expect_sources=["customer_satisfaction"],
        expect_mentions=["reporting"],
    ),
    EvalCase(
        id="nps",
        query="What is the current Net Promoter Score?",
        expect_sources=["customer_satisfaction"],
    ),
    EvalCase(
        id="modules",
        query="Which modules does the platform provide?",
        expect_sources=["product_overview"],
    ),
    EvalCase(
        id="deployment",
        query="Can the platform be self-hosted?",
        expect_sources=["product_overview"],
    ),
]

# Floors rather than targets. Set below perfect deliberately: a hard 1.0 makes
# the gate brittle to a single ranking tweak, while a floor still catches a
# real regression in retrieval.
MIN_RECALL = 0.80
MIN_FAITHFULNESS = 0.50

# Invented citations are not scored on a curve. A marker pointing at a source
# that was never supplied is a defect, not a quality level, and one is enough
# to fail the run.
MAX_INVENTED_CITATIONS = 0


def main() -> int:
    setup_logging()
    settings = get_settings()
    backend = "openai" if settings.use_openai else "local"

    store = VectorStore()
    if store.get_collection_info().get("count", 0) == 0:
        print("✗ The vector store is empty. Run scripts/ingest_sample_data.py first.")
        return 2

    from src.agent import ReportGenerationGraph

    graph = ReportGenerationGraph(store)

    # The audit for each query, kept alongside the score. The harness measures
    # retrieval; this measures whether the report was entitled to the sources
    # it named, which no recall number can tell you.
    audits = {}

    def generate(query: str) -> str:
        state = graph.generate_report(query)
        audits[query] = state.get("citation_audit") or {}
        return state.get("report", "")

    evaluator = RetrievalEvaluator(
        retrieve=lambda q: store.similarity_search(q, k=settings.top_k_results),
        generate=generate,
        backend=backend,
    )

    report = evaluator.run(CASES)
    print("\n" + report.format() + "\n")

    for case in report.cases:
        mark = "✓" if case.passed else "✗"
        print(
            f"  {mark} {case.id:<18} recall={case.recall:.2f} "
            f"precision={case.precision:.2f} faith={case.faithfulness:.2f} "
            f"{case.latency_ms}ms"
        )
        for failure in case.failures:
            print(f"      - {failure}")

    invented = {query: audit["invalid"] for query, audit in audits.items() if audit.get("invalid")}
    grounded = [audit.get("groundedness", 0.0) for audit in audits.values()]
    if grounded:
        print(f"\n  citation groundedness: mean {sum(grounded) / len(grounded):.2f}")

    ok = True
    if invented:
        for query, markers in invented.items():
            print(f"\n\u2717 invented citations {markers} in the report for {query!r}")
        ok = False
    if report.mean_recall < MIN_RECALL:
        print(f"\n✗ recall {report.mean_recall:.2f} is below the {MIN_RECALL} floor")
        ok = False
    if report.mean_faithfulness < MIN_FAITHFULNESS:
        print(
            f"\n✗ faithfulness {report.mean_faithfulness:.2f} is below the "
            f"{MIN_FAITHFULNESS} floor"
        )
        ok = False

    print("\n✅ Evaluation passed." if ok else "\n❌ Evaluation failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
