"""
Agent state definition for LangGraph.
Defines the data structure passed between nodes in the graph.
"""

from typing import Annotated, Any, Dict, List, Optional

from langchain.schema import Document
from typing_extensions import TypedDict

from .context import Citation


def reduce_documents(existing: Optional[List[Document]], new: List[Document]) -> List[Document]:
    """Reducer function for accumulating documents."""
    if existing is None:
        return new
    return existing + new


class AgentState(TypedDict):
    """
    State for the report generation agent.

    This state is passed between nodes in the LangGraph workflow.
    """

    # Input
    query: str
    """The user's query or question about the company data"""

    # Retrieval results
    retrieved_documents: Annotated[List[Document], reduce_documents]
    """Documents retrieved from the vector store"""

    # Analysis
    relevance_scores: Optional[List[float]]
    """Relevance scores for retrieved documents"""

    context: Optional[str]
    """Combined context from relevant documents"""

    citations: Optional[List[Citation]]
    """The sources put in the prompt, each with the marker the report may cite"""

    context_stats: Optional[Dict[str, Any]]
    """What context assembly dropped, trimmed and truncated, for cost accounting"""

    # Report generation
    report: Optional[str]
    """Generated report"""

    summary: Optional[str]
    """Executive summary of the report"""

    citation_audit: Optional[Dict[str, Any]]
    """Which markers the report used, which it invented, and which went unused"""

    groundedness: Optional[float]
    """Share of the report's content words present in the text it was given"""

    # Metadata
    sources: Optional[List[str]]
    """Source documents used in the report"""

    num_tokens_used: int
    """Total tokens used in generation"""

    total_cost: float
    """Total cost of API calls"""

    # Error handling
    error: Optional[str]
    """Error message if any step fails"""
