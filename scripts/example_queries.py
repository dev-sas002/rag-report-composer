#!/usr/bin/env python
"""
Example queries demonstrating the RAG system capabilities.
This script runs several example queries and displays the results.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import ReportGenerationGraph
from src.observability.cost_tracker import get_cost_tracker
from src.observability.logger import get_logger, setup_logging
from src.retrieval.vector_store import VectorStore

EXAMPLE_QUERIES = [
    "What are ACME Corporation's Q4 2024 revenue figures and key metrics?",
    "Summarize customer satisfaction and retention metrics",
    "What products does ACME offer and what are their key features?",
    "What are the company's future strategic initiatives?",
]


def run_example_query(graph: ReportGenerationGraph, query: str, query_num: int):
    """Run a single example query."""
    print("\n" + "=" * 80)
    print(f"EXAMPLE QUERY {query_num}")
    print("=" * 80)
    print(f"Query: {query}\n")

    # Generate report
    result = graph.generate_report(query)

    if result.get("error"):
        print(f"❌ Error: {result['error']}")
        return

    # Display summary
    print("EXECUTIVE SUMMARY:")
    print("-" * 80)
    print(result.get("summary", "No summary generated"))

    # Display abbreviated report
    report = result.get("report", "")
    if len(report) > 500:
        print(f"\n[Report truncated for display - {len(report)} chars total]")
        print(report[:500] + "...")
    else:
        print("\nDETAILED REPORT:")
        print("-" * 80)
        print(report)

    # Display metrics
    print("\n📊 Metrics:")
    print(f"   Sources: {len(result.get('sources', []))}")
    print(f"   Tokens: {result.get('num_tokens_used', 0):,}")
    print(f"   Cost: ${result.get('total_cost', 0.0):.4f}")


def main():
    """Run example queries."""
    # Setup
    setup_logging()
    logger = get_logger(__name__)
    cost_tracker = get_cost_tracker()

    print("🔍 RAG System - Example Queries")
    print("=" * 80)

    # Initialize components
    vector_store = VectorStore()

    # Check if data exists
    stats = vector_store.get_collection_info()
    if stats["count"] == 0:
        print("\n⚠ Warning: Vector store is empty!")
        print("Please run the ingestion script first:")
        print("  python scripts/ingest_sample_data.py")
        sys.exit(1)

    print(f"\n📊 Vector store contains {stats['count']} document chunks")
    print(f"\nRunning {len(EXAMPLE_QUERIES)} example queries...\n")

    # Initialize graph
    graph = ReportGenerationGraph(vector_store)

    # Run each example query
    for i, query in enumerate(EXAMPLE_QUERIES, 1):
        try:
            run_example_query(graph, query, i)
        except Exception as e:
            logger.error("example_query_failed", query=query, error=str(e))
            print(f"\n❌ Query failed: {e}")

    # Show total cost summary
    print("\n" + "=" * 80)
    print("SESSION SUMMARY")
    print("=" * 80)
    summary = cost_tracker.get_session_summary()
    print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
    print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
    print(f"Total Output Tokens: {summary['total_output_tokens']:,}")
    print("\nBy Model:")
    for model, data in summary["by_model"].items():
        print(f"  {model}:")
        print(f"    Cost: ${data['cost_usd']:.4f}")
        print(f"    Tokens: {data['tokens']}")

    print("\n✅ Example queries complete!")


if __name__ == "__main__":
    main()
