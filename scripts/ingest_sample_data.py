#!/usr/bin/env python
"""
Script to ingest sample company data into the vector store.
Run this script to populate the database with example data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_ingestion.ingestion_pipeline import IngestionPipeline
from src.observability.logger import get_logger, setup_logging
from src.retrieval.vector_store import VectorStore


def main():
    """Ingest sample data."""
    # Setup
    setup_logging()
    logger = get_logger(__name__)

    logger.info("sample_data_ingestion_started")
    print("🚀 Starting sample data ingestion...")

    # Initialize components
    vector_store = VectorStore()
    pipeline = IngestionPipeline(vector_store)

    # Get data directory
    data_dir = Path(__file__).parent.parent / "data" / "raw_data"

    if not data_dir.exists() or not any(data_dir.iterdir()):
        # Exiting non-zero here would take the container down with it, so this
        # reports and returns instead: an empty index is a usable state.
        print(f"⚠ No source documents in {data_dir}; nothing to ingest.")
        return

    print(f"\n📁 Data directory: {data_dir}")

    # Check existing data
    stats = pipeline.get_collection_stats()
    print(f"\n📊 Current collection: {stats['name']}")
    print(f"   Documents before ingestion: {stats['count']}")

    if stats["count"] > 0:
        # Non-interactive by default. This used to call input(), which blocks
        # forever in a container — the script runs at container start, where
        # there is no terminal to answer it.
        if not sys.stdin.isatty():
            print("\n⚠ Collection already has data; leaving it as-is.")
            print("   Re-run with --replace to clear it first.")
            if "--replace" not in sys.argv:
                print("\n✅ Nothing to do.")
                return
            pipeline.clear_collection()
            print("✓ Collection cleared")
        else:
            response = input("\n⚠ Collection already has data. Clear it first? (yes/no): ")
            if response.lower() == "yes":
                pipeline.clear_collection()
                print("✓ Collection cleared")

    # Ingest data
    print("\n📥 Ingesting sample data...")
    try:
        num_chunks = pipeline.ingest_directory(str(data_dir))
        print(f"\n✓ Successfully ingested {num_chunks} chunks")

        # Show updated stats
        stats = pipeline.get_collection_stats()
        print("\n📊 Final collection statistics:")
        print(f"   Name: {stats['name']}")
        print(f"   Total documents: {stats['count']}")

        print("\n✅ Sample data ingestion complete!")
        print("\nYou can now run queries like:")
        print('  python main.py --query "What are the company\'s revenue trends?"')
        print('  python main.py --query "Summarize customer satisfaction metrics"')
        print('  python main.py --query "What products does ACME offer?"')

    except Exception as e:
        logger.error("sample_data_ingestion_failed", error=str(e))
        print(f"\n❌ Error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
