"""
LangGraph Studio-compatible graph definition.
This module provides the graph in a format compatible with LangGraph Studio UI.
"""

from ..retrieval.vector_store import VectorStore
from .graph import ReportGenerationGraph

# Create a function that returns the compiled graph for LangGraph Studio


def create_graph():
    """
    Create and return the compiled LangGraph for Studio visualization.

    Returns:
        Compiled StateGraph for report generation
    """
    # Initialize vector store
    vector_store = VectorStore()

    # Create and return the graph
    report_gen = ReportGenerationGraph(vector_store)
    return report_gen.graph


# Export the graph for LangGraph Studio
graph = create_graph()

# For LangGraph Studio CLI compatibility
__all__ = ["graph"]
