"""Tests for the CLI helpers in main.py."""

from unittest.mock import Mock

# main.py lives at the repository root, not inside the src package. The
# previous import (`from src import main`) could never have resolved, so this
# whole file failed at collection and took the rest of the suite with it.
import main as cli_main


def test_ingest_data_uses_pipeline_and_prints(monkeypatch, capsys):
    fake_vector_store = Mock()
    fake_pipeline = Mock()
    fake_pipeline.ingest_file.return_value = 3
    fake_pipeline.get_collection_stats.return_value = {"name": "test", "count": 3}

    monkeypatch.setattr("main.IngestionPipeline", lambda vs: fake_pipeline)

    args = Mock()
    args.ingest_file = "file.txt"
    args.ingest_dir = None

    cli_main.ingest_data(args, fake_vector_store)

    fake_pipeline.ingest_file.assert_called_once_with("file.txt")
    out = capsys.readouterr().out
    assert "Successfully ingested 3 chunks" in out
    assert "Collection Statistics" in out


def test_generate_report_handles_empty_store(capsys):
    fake_vector_store = Mock()
    fake_vector_store.get_collection_info.return_value = {"count": 0}

    args = Mock()
    args.query = "test query"
    args.output = False
    args.format = "text"

    cli_main.generate_report(args, fake_vector_store)

    out = capsys.readouterr().out
    assert "Vector store is empty" in out


def test_generate_report_invokes_graph_and_writes_output(monkeypatch, capsys):
    fake_vector_store = Mock()
    fake_vector_store.get_collection_info.return_value = {"count": 1}

    fake_graph = Mock()
    fake_graph.generate_report.return_value = {
        "summary": "summary",
        "report": "report body",
        "sources": ["src1"],
        "num_tokens_used": 10,
        "total_cost": 0.1,
        "error": None,
    }

    fake_writer = Mock()
    fake_writer.save_report.return_value = "report.txt"

    monkeypatch.setattr("main.ReportGenerationGraph", lambda vs: fake_graph)
    monkeypatch.setattr("main.ReportWriter", lambda: fake_writer)

    fake_cost_tracker = Mock()
    fake_cost_tracker.get_session_summary.return_value = {
        "total_cost_usd": 0.1,
        "total_input_tokens": 5,
        "total_output_tokens": 5,
        "by_model": {"gpt-4-turbo-preview": {"cost_usd": 0.1}},
    }
    monkeypatch.setattr("main.get_cost_tracker", lambda: fake_cost_tracker)

    args = Mock()
    args.query = "test query"
    args.output = True
    args.format = "text"

    cli_main.generate_report(args, fake_vector_store)

    out = capsys.readouterr().out
    assert "EXECUTIVE SUMMARY" in out
    assert "DETAILED REPORT" in out
    assert "COST SUMMARY" in out
    fake_graph.generate_report.assert_called_once_with("test query")
    fake_writer.save_report.assert_called_once()


def test_clear_data_respects_user_input(monkeypatch, capsys):
    fake_vector_store = Mock()
    fake_pipeline = Mock()
    monkeypatch.setattr("main.IngestionPipeline", lambda vs: fake_pipeline)

    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    cli_main.clear_data(fake_vector_store)
    fake_pipeline.clear_collection.assert_called_once()

    # Now simulate a cancel
    fake_pipeline.clear_collection.reset_mock()
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    cli_main.clear_data(fake_vector_store)
    fake_pipeline.clear_collection.assert_not_called()
