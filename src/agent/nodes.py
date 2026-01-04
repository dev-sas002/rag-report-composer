"""
LangGraph nodes for the report generation workflow.

Each node is one step of the RAG pipeline and returns only the keys it
changed. The arithmetic each node performs lives in a module of its own —
context assembly in `context.py`, citation checking in `citations.py` — so it
can be tested without a graph, a vector store or a model.
"""

from typing import Any, Dict, List, Optional, Protocol

from ..config import Settings, get_settings
from ..observability.cost_tracker import CostTracker, get_cost_tracker
from ..observability.logger import get_logger
from ..providers import create_llm
from ..retrieval.vector_store import VectorStore
from .citations import audit_report
from .context import Citation, assemble_context
from .state import AgentState

logger = get_logger(__name__)

# What `generate_report` puts in the report slot when retrieval came back with
# nothing. It is a constant because `generate_summary` has to recognise it:
# paying a model to summarise "no information was found" is a billed call that
# cannot tell the reader anything.
NO_CONTEXT_REPORT = "No relevant information found to generate a report."


class LLM(Protocol):
    """Protocol for chat models used by the agent nodes."""

    def invoke(self, prompt: str):  # pragma: no cover - protocol
        ...


def build_report_prompt(query: str, context: str, max_report_length: int) -> str:
    """
    Construct the prompt used to generate the detailed report.

    Each source in the context carries an `[S1]`-style marker, and the model is
    told to reuse those markers rather than name files. Asking for a fixed,
    machine-checkable token is what makes `citations.audit_report` possible:
    "cites information from the sources" is a hope, `[S3]` is a claim that can
    be checked against what was actually supplied.
    """
    return (
        "You are a senior business analyst tasked with creating a comprehensive report "
        "based on company data.\n\n"
        f"Query: {query}\n\n"
        "Context from company documents:\n"
        f"{context}\n\n"
        "Instructions:\n"
        "1. Analyze the provided context thoroughly\n"
        "2. Generate a detailed, well-structured report that addresses the query\n"
        "3. Include specific data points and insights from the documents\n"
        "4. Organize the report with clear sections and headings\n"
        "5. Be objective and factual. Cite every claim with the marker of the "
        "source it came from, e.g. [S1]. Use only markers that appear above; "
        "never invent one\n"
        "6. If the context doesn't fully answer the query, acknowledge the limitations\n\n"
        f"Generate a comprehensive report (aim for ~{max_report_length} tokens):"
    )


def response_text(response: Any) -> str:
    """
    Pull the text out of a chat-model response, or say why it could not.

    Chat models do not all return a plain string: `content` can be a list of
    content blocks, and a misconfigured client can return an object with no
    `content` at all. Reading `.content` blind meant those cases surfaced far
    away from here, as a TypeError inside token counting.
    """
    content = getattr(response, "content", None)
    if content is None:
        raise ValueError(
            f"model response has no 'content' attribute (got {type(response).__name__})"
        )

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Content blocks: keep the text parts, drop images and tool calls.
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        text = "".join(parts)
    else:
        text = str(content)

    if not text.strip():
        raise ValueError("model returned an empty response")

    return text


def build_summary_prompt(report: str) -> str:
    """Construct the prompt used to generate the executive summary."""
    return (
        "Create a concise executive summary (3-5 key bullet points) of the following report:\n\n"
        f"{report}\n\n"
        "Executive Summary:"
    )


class AgentNodes:
    """Collection of nodes for the report generation graph."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm: Optional[LLM] = None,
        cost_tracker: Optional[CostTracker] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize agent nodes.

        Args:
            vector_store: VectorStore instance for retrieval
            llm: Optional chat model to use (defaults to ChatOpenAI)
            cost_tracker: Optional cost tracker (defaults to global tracker)
            settings: Optional settings instance (defaults to global settings)
        """
        self.vector_store = vector_store
        self.settings = settings or get_settings()
        self.cost_tracker = cost_tracker or get_cost_tracker()

        # An injected writer always wins; otherwise the configuration decides.
        # Constructing ChatOpenAI without a key would raise here, which is what
        # made this class impossible to build offline.
        self.llm: LLM = llm or self._default_llm()

    @property
    def _billing_model(self) -> str:
        """
        The model to attribute usage to.

        Reporting `openai_model` unconditionally meant the offline path logged
        gpt-4-turbo pricing for calls that never left the process — the cost
        report showed spend for a run that cost nothing.
        """
        if self.settings.use_openai:
            return self.settings.openai_model
        return "local-extractive-writer"

    def _default_llm(self) -> LLM:
        """
        The writer the configuration asks for.

        Built through the provider registry rather than an `if use_openai`
        branch here, so adding Anthropic or a local llama.cpp server is a
        registration and an environment variable, not an edit to this class.
        """
        return create_llm(self.settings)

    # Nodes return only the keys they changed.
    #
    # `retrieved_documents` carries an accumulating reducer, so every node that
    # echoed the whole state back appended the documents it had just been given
    # to the ones already there: four nodes turned k retrieved chunks into 8k
    # entries in the final state.

    def retrieve_documents(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve relevant documents from the vector store.

        Args:
            state: Current agent state

        Returns:
            State update with the retrieved documents and their scores
        """
        logger.info("node_retrieve_start", query=state["query"])

        try:
            query = state["query"]

            # Retrieve documents with scores
            results = self.vector_store.similarity_search_with_score(
                query, k=self.settings.top_k_results
            )

            documents = [doc for doc, _ in results]
            scores = [float(score) for _, score in results]

            logger.info(
                "node_retrieve_complete",
                num_documents=len(documents),
                avg_score=sum(scores) / len(scores) if scores else 0,
            )

            return {"retrieved_documents": documents, "relevance_scores": scores}

        except Exception as e:
            logger.error("node_retrieve_failed", error=str(e))
            return {"error": f"Document retrieval failed: {str(e)}"}

    def build_context(self, state: AgentState) -> Dict[str, Any]:
        """
        Build the prompt context from the retrieved documents.

        Every character assembled here is billed as an input token on every
        query, so this is where the money is. Three reductions happen, all in
        `context.assemble_context`: chunks far less relevant than the best hit
        are dropped, splitter overlap shared with an already-accepted chunk is
        trimmed, and what survives is capped at `settings.max_context_chars`.
        Without the cap the prompt grows with whatever `top_k_results` and the
        chunk size happen to be, and the first sign of trouble is the model
        rejecting the request for exceeding its context window.

        Args:
            state: Current agent state

        Returns:
            State update with the context, its sources and its citations
        """
        logger.info("node_build_context_start")

        try:
            documents = state.get("retrieved_documents") or []

            if not documents:
                logger.warning("no_documents_for_context")
                return {"context": "", "sources": [], "citations": [], "context_stats": {}}

            scores = state.get("relevance_scores") or []
            # Pair each document with its distance, if retrieval reported one.
            scored = [
                (document, scores[index] if index < len(scores) else None)
                for index, document in enumerate(documents)
            ]

            bundle = assemble_context(
                scored,
                max_chars=self.settings.max_context_chars,
                chunk_overlap=getattr(self.settings, "chunk_overlap", 0),
                relevance_margin=getattr(self.settings, "context_relevance_margin", 0.0),
            )

            if bundle.truncated:
                logger.warning(
                    "context_truncated",
                    max_context_chars=self.settings.max_context_chars,
                    documents_retrieved=len(documents),
                    documents_used=len(bundle.sources),
                )

            stats = {
                "raw_chars": bundle.raw_chars,
                "context_chars": len(bundle.text),
                "pruned_by_relevance": bundle.pruned_by_relevance,
                "dropped_as_duplicate": bundle.dropped_as_duplicate,
                "overlap_chars_trimmed": bundle.overlap_chars_trimmed,
                "truncated": bundle.truncated,
            }

            logger.info("node_build_context_complete", num_sources=len(bundle.sources), **stats)

            return {
                "context": bundle.text,
                "sources": bundle.sources,
                "citations": bundle.citations,
                "context_stats": stats,
            }

        except Exception as e:
            logger.error("node_build_context_failed", error=str(e))
            return {"error": f"Context building failed: {str(e)}"}

    def generate_report(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate detailed report from context.

        Args:
            state: Current agent state

        Returns:
            State update with the generated report and its cost
        """
        logger.info("node_generate_report_start")

        try:
            query = state["query"]
            context = state.get("context") or ""

            if not context:
                logger.warning("no_context_for_report")
                return {"report": NO_CONTEXT_REPORT}

            # Create prompt
            prompt = build_report_prompt(query, context, self.settings.max_report_length)

            # Count input tokens
            input_tokens = self.cost_tracker.count_tokens(prompt, self._billing_model)

            # Generate report
            report = response_text(self.llm.invoke(prompt))

            # Count output tokens
            output_tokens = self.cost_tracker.count_tokens(report, self._billing_model)

            # Track cost
            cost = self.cost_tracker.track_call(
                model=self._billing_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation="generate_report",
                metadata={"query": query},
            )

            citations: List[Citation] = list(state.get("citations") or [])
            audit = audit_report(report, citations, query=query)

            if audit.has_invented_citations:
                # Loud, because the output still looks like a sourced report:
                # the marker is there, it just points at nothing.
                logger.error(
                    "invented_citations",
                    invalid=audit.invalid,
                    supplied=[c.id for c in citations],
                )

            logger.info(
                "node_generate_report_complete",
                report_length=len(report),
                tokens_used=input_tokens + output_tokens,
                cost=cost,
                groundedness=round(audit.groundedness, 4),
                citations_used=len(audit.valid),
                citations_unused=len(audit.unused),
            )

            return {
                "report": report,
                "citation_audit": audit.as_dict(),
                "groundedness": audit.groundedness,
                "num_tokens_used": state.get("num_tokens_used", 0) + input_tokens + output_tokens,
                "total_cost": state.get("total_cost", 0.0) + cost,
            }

        except Exception as e:
            logger.error("node_generate_report_failed", error=str(e))
            return {"error": f"Report generation failed: {str(e)}"}

    def generate_summary(self, state: AgentState) -> Dict[str, Any]:
        """
        Generate executive summary from the report.

        Args:
            state: Current agent state

        Returns:
            State update with the executive summary and its cost
        """
        logger.info("node_generate_summary_start")

        try:
            report = state.get("report") or ""

            if not report:
                logger.warning("no_report_for_summary")
                return {"summary": ""}

            if report == NO_CONTEXT_REPORT:
                logger.warning("no_findings_to_summarise")
                return {"summary": ""}

            # Create prompt
            prompt = build_summary_prompt(report)

            # Count input tokens
            input_tokens = self.cost_tracker.count_tokens(prompt, self._billing_model)

            # Generate summary
            summary = response_text(self.llm.invoke(prompt))

            # Count output tokens
            output_tokens = self.cost_tracker.count_tokens(summary, self._billing_model)

            # Track cost
            cost = self.cost_tracker.track_call(
                model=self._billing_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation="generate_summary",
            )

            logger.info(
                "node_generate_summary_complete",
                summary_length=len(summary),
                tokens_used=input_tokens + output_tokens,
                cost=cost,
            )

            return {
                "summary": summary,
                "num_tokens_used": state.get("num_tokens_used", 0) + input_tokens + output_tokens,
                "total_cost": state.get("total_cost", 0.0) + cost,
            }

        except Exception as e:
            logger.error("node_generate_summary_failed", error=str(e))
            return {"error": f"Summary generation failed: {str(e)}"}
