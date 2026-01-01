#!/usr/bin/env python
"""
Test script for FastAPI endpoints.
This script tests all API endpoints to ensure they work correctly.
"""

import os
import sys
import time
from pathlib import Path

import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# API base URL - check if custom port is provided
API_PORT = os.environ.get("API_PORT", "8350")
BASE_URL = f"http://localhost:{API_PORT}"


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_health():
    """Test health endpoint."""
    print_section("Testing Health Endpoint")

    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()

        data = response.json()
        print("✓ Health check passed")
        print(f"  Status: {data['status']}")
        print(f"  App: {data['app_name']}")
        print(f"  Version: {data['version']}")
        print(f"  Documents: {data['vector_store_documents']}")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_stats():
    """Test stats endpoint."""
    print_section("Testing Stats Endpoint")

    try:
        response = requests.get(f"{BASE_URL}/stats")
        response.raise_for_status()

        data = response.json()
        print("✓ Stats retrieved")
        print(f"  Collection: {data['collection_name']}")
        print(f"  Total documents: {data['total_documents']}")
        return True
    except Exception as e:
        print(f"✗ Stats retrieval failed: {e}")
        return False


def test_ingest_text():
    """Test text ingestion endpoint."""
    print_section("Testing Text Ingestion")

    try:
        payload = {
            "text": "This is a test document about ACME Corporation. The company achieved record revenue of $150 million in 2024.",
            "source_name": "api_test_document",
        }

        response = requests.post(f"{BASE_URL}/ingest/text", json=payload)
        response.raise_for_status()

        data = response.json()
        print("✓ Text ingested successfully")
        print(f"  Message: {data['message']}")
        print(f"  Chunks added: {data['chunks_added']}")
        print(f"  Total documents: {data['total_documents']}")
        return True
    except Exception as e:
        print(f"✗ Text ingestion failed: {e}")
        return False


def test_query():
    """Test query endpoint."""
    print_section("Testing Query/Report Generation")

    try:
        payload = {
            "query": "What is ACME Corporation's revenue?",
            "save_report": False,
            "format": "markdown",
        }

        print("Generating report (this may take 10-15 seconds)...")
        response = requests.post(f"{BASE_URL}/query", json=payload)
        response.raise_for_status()

        data = response.json()
        print("✓ Report generated successfully")
        print(f"\nQuery: {data['query']}")
        print("\nExecutive Summary:")
        print(data["summary"][:200] + "..." if len(data["summary"]) > 200 else data["summary"])
        print("\nReport Preview:")
        print(data["report"][:300] + "..." if len(data["report"]) > 300 else data["report"])
        print("\nMetrics:")
        print(f"  Sources: {len(data['sources'])}")
        print(f"  Tokens used: {data['num_tokens_used']:,}")
        print(f"  Cost: ${data['total_cost_usd']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Query failed: {e}")
        if hasattr(e, "response") and e.response:
            print(f"  Response: {e.response.text}")
        return False


def test_costs():
    """Test costs endpoint."""
    print_section("Testing Costs Endpoint")

    try:
        response = requests.get(f"{BASE_URL}/costs")
        response.raise_for_status()

        data = response.json()
        print("✓ Cost summary retrieved")
        print(f"  Total cost: ${data['total_cost_usd']:.4f}")
        print(f"  Input tokens: {data['total_input_tokens']:,}")
        print(f"  Output tokens: {data['total_output_tokens']:,}")
        print("\n  By Model:")
        for model, info in data["by_model"].items():
            print(f"    {model}: ${info['cost_usd']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Cost retrieval failed: {e}")
        return False


def test_root():
    """Test root endpoint."""
    print_section("Testing Root Endpoint")

    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()

        data = response.json()
        print("✓ Root endpoint working")
        print(f"  Message: {data['message']}")
        print(f"  Version: {data['version']}")
        return True
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False


def main():
    """Run all API tests."""
    print("=" * 80)
    print("  RAG Company Report Generator - API Testing")
    print("=" * 80)
    print(f"\nBase URL: {BASE_URL}")
    print("\nMake sure the API server is running:")
    print("  python -m src.api.main")
    print("  OR")
    print("  uvicorn src.api.main:app --reload")

    # Wait a moment for user to confirm
    print("\nStarting tests in 3 seconds...")
    time.sleep(3)

    # Run tests
    results = []

    results.append(("Root", test_root()))
    results.append(("Health", test_health()))
    results.append(("Stats", test_stats()))
    results.append(("Text Ingestion", test_ingest_text()))
    results.append(("Query/Report", test_query()))
    results.append(("Costs", test_costs()))

    # Summary
    print_section("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        print(f"\nAPI Documentation: {BASE_URL}/docs")
        print(f"Interactive API: {BASE_URL}/redoc")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
