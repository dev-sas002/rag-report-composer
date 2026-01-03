"""
LangGraph workflow definition for report generation.
Orchestrates the RAG pipeline using a state machine.
"""

from langgraph.graph import END, StateGraph

from ..observability.logger import get_logger
from ..retrieval.vector_store import VectorStore
from .nodes import AgentNodes
from .state import AgentState

logger = get_logger(__name__)


class ReportGenerationGraph:
    """
    LangGraph-based workflow for generating company reports.

    Workflow:
    1. Retrieve relevant documents from vector store
    2. Build context from retrieved documents
    3. Generate detailed report
    4. Generate executive summary
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the report generation graph.

        Args:
            vector_store: VectorStore instance for document retrieval
        """
        self.vector_store = vector_store
        self.nodes = AgentNodes(vector_store)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        # Create graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve_documents", self.nodes.retrieve_documents)
        workflow.add_node("build_context", self.nodes.build_context)
        workflow.add_node("generate_report", self.nodes.generate_report)
        workflow.add_node("generate_summary", self.nodes.generate_summary)

        # Define edges (workflow)
        workflow.set_entry_point("retrieve_documents")
        workflow.add_edge("retrieve_documents", "build_context")
        workflow.add_edge("build_context", "generate_report")
        workflow.add_edge("generate_report", "generate_summary")
        workflow.add_edge("generate_summary", END)

        # Compile graph
        compiled_graph = workflow.compile()

        logger.info("graph_compiled", num_nodes=4)

        return compiled_graph

    def generate_report(self, query: str) -> AgentState:
        """
        Generate a report for the given query.

        Args:
            query: User's query about company data

        Returns:
            Final agent state with report and summary
        """
        logger.info("report_generation_started", query=query)

        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "retrieved_documents": [],
            "relevance_scores": None,
            "context": None,
            "citations": None,
            "context_stats": None,
            "report": None,
            "summary": None,
            "citation_audit": None,
            "groundedness": None,
            "sources": None,
            "num_tokens_used": 0,
            "total_cost": 0.0,
            "error": None,
        }

        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)

            # Check for errors
            if final_state.get("error"):
                logger.error("report_generation_failed", error=final_state["error"])
            else:
                # `or` rather than a .get default: these keys are always
                # present in the state and may hold None, so a default never
                # applies and len(None) raised inside the success path.
                logger.info(
                    "report_generation_complete",
                    query=query,
                    report_length=len(final_state.get("report") or ""),
                    summary_length=len(final_state.get("summary") or ""),
                    num_sources=len(final_state.get("sources") or []),
                    total_tokens=final_state.get("num_tokens_used") or 0,
                    total_cost=final_state.get("total_cost") or 0.0,
                    groundedness=final_state.get("groundedness"),
                )

            return final_state

        except Exception as e:
            logger.error("report_generation_exception", error=str(e), query=query)
            initial_state["error"] = str(e)
            return initial_state

    def get_graph_visualization(self) -> str:
        """
        Get a text representation of the graph structure.

        Returns:
            Graph structure as string
        """
        return """
Report Generation Graph:

┌─────────────────────────┐
│  retrieve_documents     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│    build_context        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   generate_report       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  generate_summary       │
└───────────┬─────────────┘
            │
            ▼
           END
        """
