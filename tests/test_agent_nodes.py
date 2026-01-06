"""Tests for AgentNodes behavior."""

from unittest.mock import Mock

import pytest
from langchain.schema import Document

from src.agent.nodes import (
    AgentNodes,
    build_report_prompt,
    build_summary_prompt,
    response_text,
)


class DummyResponse:
    """Simple container mimicking ChatOpenAI response objects."""

    def __init__(self, content: str) -> None:
        self.content = content


@pytest.fixture
def fake_settings():
    class SettingsStub:
        # This stub was missing `top_k_results`, so retrieve_documents raised
        # AttributeError, the node's catch-all turned it into state["error"],
        # and the tests failed with the misleading message that the vector
        # store had never been called.
        openai_model = "gpt-4-turbo-preview"
        openai_api_key = "test-key"
        openai_request_timeout = 30.0
        openai_max_retries = 2
        max_report_length = 2000
        top_k_results = 5
        max_context_chars = 12000
        chunk_overlap = 200
        context_relevance_margin = 0.0
        use_openai = True

    return SettingsStub()


@pytest.fixture
def fake_cost_tracker():
    tracker = Mock()
    tracker.count_tokens.side_effect = lambda text, model=None: len(text.split())
    tracker.track_call.return_value = 0.01
    return tracker


@pytest.fixture
def fake_vector_store():
    store = Mock()
    store.similarity_search_with_score.return_value = [
        (Document(page_content="content 1", metadata={"source": "doc1"}), 0.1),
        (Document(page_content="content 2", metadata={"source": "doc2"}), 0.2),
    ]
    return store


@pytest.fixture
def fake_llm():
    llm = Mock()
    llm.invoke.side_effect = lambda prompt: DummyResponse(f"RESPONSE FOR: {prompt[:20]}")
    return llm


@pytest.fixture
def agent_nodes(fake_vector_store, fake_llm, fake_cost_tracker, fake_settings):
    return AgentNodes(
        vector_store=fake_vector_store,
        llm=fake_llm,
        cost_tracker=fake_cost_tracker,
        settings=fake_settings,
    )


def test_build_report_prompt_includes_query_and_context(fake_settings):
    prompt = build_report_prompt(
        query="What are revenues?",
        context="Revenue grew by 10%.",
        max_report_length=fake_settings.max_report_length,
    )
    assert "What are revenues?" in prompt
    assert "Revenue grew by 10%." in prompt
    assert str(fake_settings.max_report_length) in prompt


def test_build_summary_prompt_includes_report():
    report = "This is a long report."
    prompt = build_summary_prompt(report)
    assert report in prompt
    assert "Executive Summary" in prompt


def test_retrieve_documents_populates_state(agent_nodes, fake_vector_store):
    state = {"query": "test query"}

    updated = agent_nodes.retrieve_documents(state)

    fake_vector_store.similarity_search_with_score.assert_called_once()
    assert len(updated["retrieved_documents"]) == 2
    assert updated["relevance_scores"] == [0.1, 0.2]


def test_build_context_uses_sources_and_content(agent_nodes):
    docs = [
        Document(page_content="alpha", metadata={"source": "first.txt"}),
        Document(page_content="beta", metadata={"source": "second.txt"}),
    ]
    state = {"retrieved_documents": docs}

    updated = agent_nodes.build_context(state)

    assert "first.txt" in updated["context"]
    assert "second.txt" in updated["context"]
    assert updated["sources"] == ["first.txt", "second.txt"]


def test_build_context_handles_no_documents(agent_nodes):
    state = {"retrieved_documents": []}

    updated = agent_nodes.build_context(state)

    assert updated["context"] == ""
    assert updated["sources"] == []


def test_generate_report_updates_state_and_cost(agent_nodes, fake_cost_tracker, fake_llm):
    state = {
        "query": "test query",
        "context": "some context",
        "num_tokens_used": 0,
        "total_cost": 0.0,
    }

    updated = agent_nodes.generate_report(state)

    fake_llm.invoke.assert_called_once()
    fake_cost_tracker.count_tokens.assert_any_call(
        updated["report"], agent_nodes.settings.openai_model
    )
    assert updated["report"].startswith("RESPONSE FOR:")
    assert updated["num_tokens_used"] > 0
    assert updated["total_cost"] > 0.0


def test_generate_report_handles_missing_context(agent_nodes, fake_llm):
    state = {"query": "test query", "context": ""}

    updated = agent_nodes.generate_report(state)

    # When context is empty, a default message should be set and LLM not invoked.
    fake_llm.invoke.assert_not_called()
    assert "No relevant information found" in updated["report"]


def test_generate_summary_updates_state(agent_nodes, fake_llm, fake_cost_tracker):
    state = {
        "report": "detailed report content",
        "num_tokens_used": 0,
        "total_cost": 0.0,
    }

    updated = agent_nodes.generate_summary(state)

    fake_llm.invoke.assert_called_once()
    fake_cost_tracker.count_tokens.assert_any_call(
        updated["summary"], agent_nodes.settings.openai_model
    )
    assert updated["summary"].startswith("RESPONSE FOR:")
    assert updated["num_tokens_used"] > 0
    assert updated["total_cost"] > 0.0


def test_generate_summary_handles_missing_report(agent_nodes, fake_llm):
    state = {"report": ""}

    updated = agent_nodes.generate_summary(state)

    fake_llm.invoke.assert_not_called()
    assert updated["summary"] == ""


def test_retrieve_documents_returns_only_the_keys_it_changed(agent_nodes):
    """
    Nodes must return a delta, not the whole state.

    `retrieved_documents` carries an accumulating reducer, so a node that
    echoes the state back re-appends the documents it was handed. Four nodes
    doing that turned k retrieved chunks into 8k entries.
    """
    state = {"query": "test query"}

    update = agent_nodes.retrieve_documents(state)

    assert set(update) == {"retrieved_documents", "relevance_scores"}
    assert "query" not in update


def test_build_context_returns_only_the_keys_it_changed(agent_nodes):
    docs = [Document(page_content="alpha", metadata={"source": "first.txt"})]

    update = agent_nodes.build_context({"retrieved_documents": docs})

    assert set(update) == {"context", "sources", "citations", "context_stats"}
    assert "retrieved_documents" not in update


def test_generate_report_does_not_echo_retrieved_documents(agent_nodes):
    docs = [Document(page_content="alpha", metadata={"source": "first.txt"})]
    state = {
        "query": "q",
        "context": "some context",
        "retrieved_documents": docs,
        "num_tokens_used": 0,
        "total_cost": 0.0,
    }

    update = agent_nodes.generate_report(state)

    assert "retrieved_documents" not in update


def test_generate_summary_does_not_echo_retrieved_documents(agent_nodes):
    docs = [Document(page_content="alpha", metadata={"source": "first.txt"})]
    state = {
        "report": "a report",
        "retrieved_documents": docs,
        "num_tokens_used": 0,
        "total_cost": 0.0,
    }

    update = agent_nodes.generate_summary(state)

    assert "retrieved_documents" not in update


def test_retrieve_documents_preserves_ranking_order(agent_nodes, fake_vector_store):
    """The store returns results best-first; the node must not reorder them."""
    ranked = [
        (Document(page_content="best", metadata={"source": "a"}), 0.05),
        (Document(page_content="middle", metadata={"source": "b"}), 0.30),
        (Document(page_content="worst", metadata={"source": "c"}), 0.90),
    ]
    fake_vector_store.similarity_search_with_score.return_value = ranked

    update = agent_nodes.retrieve_documents({"query": "q"})

    assert [d.page_content for d in update["retrieved_documents"]] == [
        "best",
        "middle",
        "worst",
    ]
    assert update["relevance_scores"] == [0.05, 0.30, 0.90]


def test_retrieve_documents_requests_the_configured_k(agent_nodes, fake_vector_store):
    agent_nodes.retrieve_documents({"query": "q"})

    _args, kwargs = fake_vector_store.similarity_search_with_score.call_args
    assert kwargs["k"] == agent_nodes.settings.top_k_results


def test_retrieve_documents_reports_failure_as_state_error(agent_nodes, fake_vector_store):
    fake_vector_store.similarity_search_with_score.side_effect = RuntimeError("chroma down")

    update = agent_nodes.retrieve_documents({"query": "q"})

    assert "chroma down" in update["error"]


class TestContextAssembly:
    """Context is what gets paid for, so its size must be bounded."""

    def test_context_is_capped_at_max_context_chars(self, agent_nodes):
        agent_nodes.settings.max_context_chars = 300
        docs = [
            Document(page_content="x" * 500, metadata={"source": f"doc{i}.txt"}) for i in range(10)
        ]

        update = agent_nodes.build_context({"retrieved_documents": docs})

        assert len(update["context"]) <= 300
        assert len(update["sources"]) < len(docs)

    def test_an_unbounded_context_would_otherwise_grow_with_top_k(self, agent_nodes):
        """With room to spare, every retrieved chunk is used."""
        agent_nodes.settings.max_context_chars = 100_000
        docs = [
            Document(page_content="y" * 500, metadata={"source": f"doc{i}.txt"}) for i in range(10)
        ]

        update = agent_nodes.build_context({"retrieved_documents": docs})

        assert len(update["sources"]) == 10

    def test_sources_are_labelled_in_document_order(self, agent_nodes):
        docs = [
            Document(page_content="alpha", metadata={"source": "first.txt"}),
            Document(page_content="beta", metadata={"source": "second.txt"}),
        ]

        context = agent_nodes.build_context({"retrieved_documents": docs})["context"]

        assert context.index("first.txt") < context.index("second.txt")
        assert "--- [S1] first.txt ---" in context

    def test_a_document_without_source_metadata_still_gets_a_label(self, agent_nodes):
        docs = [Document(page_content="alpha", metadata={})]

        update = agent_nodes.build_context({"retrieved_documents": docs})

        assert update["sources"] == ["Document 1"]

    def test_none_retrieved_documents_is_treated_as_empty(self, agent_nodes):
        update = agent_nodes.build_context({"retrieved_documents": None})

        assert update == {
            "context": "",
            "sources": [],
            "citations": [],
            "context_stats": {},
        }


class TestResponseValidation:
    """
    `response.content` is not always a string, and blindly reading it moved the
    failure to whichever line first did arithmetic on the result.
    """

    def test_a_plain_string_passes_through(self):
        assert response_text(DummyResponse("hello")) == "hello"

    def test_content_blocks_are_joined(self):
        assert response_text(DummyResponse([{"type": "text", "text": "a"}, {"text": "b"}])) == "ab"

    def test_a_response_without_content_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="no 'content'"):
            response_text(object())

    def test_an_empty_response_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            response_text(DummyResponse("   "))

    def test_generate_report_turns_a_bad_response_into_state_error(self, agent_nodes, fake_llm):
        fake_llm.invoke.side_effect = lambda prompt: DummyResponse("")

        update = agent_nodes.generate_report({"query": "q", "context": "ctx"})

        assert "Report generation failed" in update["error"]
        assert "report" not in update


class TestBillingModel:
    def test_offline_runs_are_not_billed_as_gpt4(
        self, fake_vector_store, fake_llm, fake_cost_tracker
    ):
        class LocalSettings:
            openai_model = "gpt-4-turbo-preview"
            openai_api_key = None
            openai_request_timeout = 30.0
            openai_max_retries = 2
            max_report_length = 2000
            top_k_results = 5
            max_context_chars = 12000
            chunk_overlap = 200
            context_relevance_margin = 0.0
            use_openai = False

        nodes = AgentNodes(
            vector_store=fake_vector_store,
            llm=fake_llm,
            cost_tracker=fake_cost_tracker,
            settings=LocalSettings(),
        )

        nodes.generate_report({"query": "q", "context": "ctx"})

        _args, kwargs = fake_cost_tracker.track_call.call_args
        assert kwargs["model"] == "local-extractive-writer"
