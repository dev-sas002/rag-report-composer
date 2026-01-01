"""
End-to-end tests for the report-generation path.

These drive the *real* compiled LangGraph — retrieve, build context, write the
report, write the summary — with the vector store and the chat model replaced
by fakes. Nothing here touches a network, an API key or a Chroma file; what is
being tested is the wiring between the nodes, which the per-node unit tests
cannot see.
"""

import re
from unittest.mock import Mock

import pytest
from langchain.schema import Document

from src.agent.graph import ReportGenerationGraph
from src.agent.local_llm import ExtractiveReportWriter
from src.agent.nodes import AgentNodes


class RecordingLLM:
    """A chat model that records its prompts and answers deterministically."""

    def __init__(self) -> None:
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if prompt.lstrip().lower().startswith("create a concise executive summary"):
            return Mock(content="SUMMARY: three bullet points")
        return Mock(content="REPORT: revenue grew twenty two percent")


class SettingsStub:
    openai_model = "gpt-4-turbo-preview"
    openai_api_key = "test-key"
    openai_request_timeout = 30.0
    openai_max_retries = 2
    max_report_length = 2000
    top_k_results = 3
    max_context_chars = 12000
    use_openai = True


DOCUMENTS = [
    Document(
        page_content=(
            "Total revenue for the financial year reached $48.2 million, an "
            "increase of twenty two percent on the prior year."
        ),
        metadata={"source": "annual_report_2025.md"},
    ),
    Document(
        page_content=(
            "Net Promoter Score for the quarter was forty one, up from thirty "
            "four in the previous quarter."
        ),
        metadata={"source": "customer_satisfaction_q4.md"},
    ),
]


@pytest.fixture
def fake_vector_store():
    store = Mock()
    store.similarity_search_with_score.return_value = [
        (DOCUMENTS[0], 0.11),
        (DOCUMENTS[1], 0.42),
    ]
    return store


@pytest.fixture
def cost_tracker():
    tracker = Mock()
    tracker.count_tokens.side_effect = lambda text, model=None: len(text.split())
    tracker.track_call.return_value = 0.02
    return tracker


def build_graph(vector_store, llm, cost_tracker):
    """Build the real graph, then swap in fully injected nodes."""
    graph = ReportGenerationGraph.__new__(ReportGenerationGraph)
    graph.vector_store = vector_store
    graph.nodes = AgentNodes(
        vector_store=vector_store,
        llm=llm,
        cost_tracker=cost_tracker,
        settings=SettingsStub(),
    )
    graph.graph = graph._build_graph()
    return graph


class TestHappyPath:
    def test_the_graph_produces_a_report_a_summary_and_its_sources(
        self, fake_vector_store, cost_tracker
    ):
        graph = build_graph(fake_vector_store, RecordingLLM(), cost_tracker)

        state = graph.generate_report("How much did revenue grow?")

        assert state["error"] is None
        assert state["report"].startswith("REPORT:")
        assert state["summary"].startswith("SUMMARY:")
        assert state["sources"] == [
            "annual_report_2025.md",
            "customer_satisfaction_q4.md",
        ]

    def test_retrieved_documents_are_not_duplicated_by_downstream_nodes(
        self, fake_vector_store, cost_tracker
    ):
        """
        `retrieved_documents` has an accumulating reducer. Every node that
        returned the whole state re-submitted the documents it had just read,
        so two retrieved chunks arrived at the end as sixteen.
        """
        graph = build_graph(fake_vector_store, RecordingLLM(), cost_tracker)

        state = graph.generate_report("How much did revenue grow?")

        assert len(state["retrieved_documents"]) == len(DOCUMENTS)

    def test_the_report_prompt_carries_the_question_and_the_retrieved_text(
        self, fake_vector_store, cost_tracker
    ):
        llm = RecordingLLM()
        graph = build_graph(fake_vector_store, llm, cost_tracker)

        graph.generate_report("How much did revenue grow?")

        report_prompt = llm.prompts[0]
        assert "How much did revenue grow?" in report_prompt
        assert "$48.2 million" in report_prompt
        # The header carries a citation marker, not just a label: it is the
        # token the model is told to cite and the audit later checks.
        assert "--- [S1] annual_report_2025.md ---" in report_prompt
        assert "[S1]" in report_prompt

    def test_the_summary_prompt_is_built_from_the_report_not_the_context(
        self, fake_vector_store, cost_tracker
    ):
        llm = RecordingLLM()
        graph = build_graph(fake_vector_store, llm, cost_tracker)

        graph.generate_report("How much did revenue grow?")

        summary_prompt = llm.prompts[1]
        assert "REPORT: revenue grew twenty two percent" in summary_prompt
        assert "$48.2 million" not in summary_prompt

    def test_cost_and_tokens_accumulate_across_both_generation_steps(
        self, fake_vector_store, cost_tracker
    ):
        graph = build_graph(fake_vector_store, RecordingLLM(), cost_tracker)

        state = graph.generate_report("How much did revenue grow?")

        assert cost_tracker.track_call.call_count == 2
        assert state["total_cost"] == pytest.approx(0.04)
        assert state["num_tokens_used"] > 0


class TestFailurePaths:
    def test_an_empty_index_yields_a_report_that_says_so(self, cost_tracker):
        empty_store = Mock()
        empty_store.similarity_search_with_score.return_value = []
        llm = RecordingLLM()
        graph = build_graph(empty_store, llm, cost_tracker)

        state = graph.generate_report("anything")

        assert state["sources"] == []
        assert "No relevant information found" in state["report"]
        # Nothing was generated, so nothing was billed.
        assert llm.prompts == []
        cost_tracker.track_call.assert_not_called()

    def test_a_retrieval_failure_surfaces_as_an_error_not_an_exception(
        self, fake_vector_store, cost_tracker
    ):
        fake_vector_store.similarity_search_with_score.side_effect = RuntimeError(
            "chroma unreachable"
        )
        graph = build_graph(fake_vector_store, RecordingLLM(), cost_tracker)

        state = graph.generate_report("anything")

        assert "chroma unreachable" in state["error"]

    def test_an_llm_failure_surfaces_as_an_error(self, fake_vector_store, cost_tracker):
        llm = Mock()
        llm.invoke.side_effect = TimeoutError("request timed out")
        graph = build_graph(fake_vector_store, llm, cost_tracker)

        state = graph.generate_report("anything")

        assert "request timed out" in state["error"]


class TestOfflineWriterEndToEnd:
    """The keyless path has to work as a whole, not only per node."""

    def test_the_offline_report_quotes_only_retrieved_text(self, fake_vector_store, cost_tracker):
        graph = build_graph(fake_vector_store, ExtractiveReportWriter(), cost_tracker)

        state = graph.generate_report("How much did revenue grow?")

        quoted = [
            line[2:].strip() for line in state["report"].splitlines() if line.startswith("- ")
        ]
        assert quoted
        corpus = " ".join(d.page_content for d in DOCUMENTS)
        for line in quoted:
            # Every bullet ends in the marker of the source it came from; what
            # precedes the marker must be verbatim corpus text.
            assert re.search(r"\s\[S\d+\]$", line), line
            assert re.sub(r"\s*\[S\d+\]$", "", line) in corpus

    def test_the_offline_report_cites_only_supplied_sources(self, fake_vector_store, cost_tracker):
        """
        The keyless path exercises the citation audit too. If it did not, the
        audit would only ever run on the path that costs money to test.
        """
        graph = build_graph(fake_vector_store, ExtractiveReportWriter(), cost_tracker)

        state = graph.generate_report("How much did revenue grow?")

        audit = state["citation_audit"]
        assert audit["valid"]
        assert audit["invalid"] == []
        assert state["groundedness"] == pytest.approx(1.0)

    def test_the_offline_summary_is_prose_rather_than_a_second_report(
        self, fake_vector_store, cost_tracker
    ):
        graph = build_graph(fake_vector_store, ExtractiveReportWriter(), cost_tracker)

        state = graph.generate_report("How much did revenue grow?")

        assert not state["summary"].lstrip().startswith("##")
        assert not state["summary"].lstrip().startswith("- ")
        assert "\n- " not in state["summary"]
