"""Tests for the ReportGenerationGraph orchestration."""

from unittest.mock import Mock

from src.agent.graph import ReportGenerationGraph


def test_generate_report_invokes_graph_with_initial_state(monkeypatch):
    """generate_report should delegate to the compiled graph with a well-formed state."""
    fake_vector_store = Mock()
    fake_graph = Mock()
    fake_graph.invoke.return_value = {
        "query": "test",
        "report": "report",
        "summary": "summary",
        "sources": [],
        "num_tokens_used": 10,
        "total_cost": 0.1,
        "error": None,
    }

    # Avoid building a real LangGraph by patching after construction.
    graph = ReportGenerationGraph(vector_store=fake_vector_store)
    graph.graph = fake_graph

    final_state = graph.generate_report("test")

    fake_graph.invoke.assert_called_once()
    (initial_state,) = fake_graph.invoke.call_args.args
    assert initial_state["query"] == "test"
    assert final_state["report"] == "report"
    assert final_state["summary"] == "summary"


def test_generate_report_handles_exceptions_gracefully(monkeypatch):
    """Exceptions raised by the underlying graph should be converted into error state."""
    fake_vector_store = Mock()
    graph = ReportGenerationGraph(vector_store=fake_vector_store)
    graph.graph = Mock()
    graph.graph.invoke.side_effect = RuntimeError("boom")

    final_state = graph.generate_report("test query")

    assert "boom" in final_state["error"]
