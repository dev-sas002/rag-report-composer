"""Tests for the FastAPI application."""

from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.api import main as api_main


def _setup_app_with_fakes(monkeypatch):
    """
    Point the app's factories at fakes for the duration of one test.

    Via monkeypatch rather than plain assignment: these are module-level
    names, so assigning them left every later test in the session running
    against another test's fakes.
    """
    fake_vector_store = Mock()
    fake_vector_store.get_collection_info.return_value = {
        "name": "test_collection",
        "count": 1,
        "metadata": {},
    }

    fake_ingestion_pipeline = Mock()
    fake_ingestion_pipeline.ingest_documents.return_value = 1
    fake_ingestion_pipeline.ingest_file.return_value = 1
    fake_ingestion_pipeline.clear_collection.return_value = None

    fake_report_graph = Mock()
    fake_report_graph.generate_report.return_value = {
        "query": "test",
        "report": "report body",
        "summary": "summary body",
        "sources": ["src1"],
        "num_tokens_used": 42,
        "total_cost": 0.1,
        "error": None,
    }

    monkeypatch.setattr(api_main, "create_vector_store", lambda: fake_vector_store)
    monkeypatch.setattr(
        api_main, "create_ingestion_pipeline", lambda store: fake_ingestion_pipeline
    )
    monkeypatch.setattr(api_main, "create_report_graph", lambda store: fake_report_graph)

    return fake_vector_store, fake_ingestion_pipeline, fake_report_graph


def test_root_and_health_endpoints(monkeypatch):
    _setup_app_with_fakes(monkeypatch)

    with TestClient(api_main.app) as client:
        root_resp = client.get("/")
        assert root_resp.status_code == 200
        data = root_resp.json()
        assert data["message"].startswith("RAG Company Report Generator API")

        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health = health_resp.json()
        assert health["status"] == "healthy"


def test_ingest_text_and_query_flow(monkeypatch):
    fake_store, fake_pipeline, fake_graph = _setup_app_with_fakes(monkeypatch)

    with TestClient(api_main.app) as client:
        ingest_resp = client.post(
            "/ingest/text",
            json={"text": "hello world", "source_name": "unit_test"},
        )
        assert ingest_resp.status_code == 200
        ingest_data = ingest_resp.json()
        assert ingest_data["chunks_added"] == 1

        fake_store.get_collection_info.return_value = {
            "name": "test_collection",
            "count": 1,
            "metadata": {},
        }

        query_resp = client.post(
            "/query", json={"query": "test", "save_report": False, "format": "markdown"}
        )
        assert query_resp.status_code == 200
        body = query_resp.json()
        assert body["report"] == "report body"
        assert body["summary"] == "summary body"


def test_clear_collection_and_costs(monkeypatch):
    fake_store, fake_pipeline, _ = _setup_app_with_fakes(monkeypatch)

    fake_cost_tracker = Mock()
    fake_cost_tracker.get_session_summary.return_value = {
        "total_cost_usd": 0.1,
        "total_input_tokens": 10,
        "total_output_tokens": 5,
        "by_model": {
            "gpt-4-turbo-preview": {"cost_usd": 0.1, "tokens": {"input": 10, "output": 5}}
        },
    }
    monkeypatch.setattr("src.api.main.get_cost_tracker", lambda: fake_cost_tracker)

    with TestClient(api_main.app) as client:
        clear_resp = client.delete("/clear")
        assert clear_resp.status_code == 200
        fake_pipeline.clear_collection.assert_called_once()

        costs_resp = client.get("/costs")
        assert costs_resp.status_code == 200
        costs = costs_resp.json()
        assert costs["total_cost_usd"] == 0.1


def test_query_against_an_empty_index_is_a_client_error_not_a_server_error(monkeypatch):
    """Asking a question of an empty index is the user's problem to fix."""
    fake_store, _pipeline, fake_graph = _setup_app_with_fakes(monkeypatch)
    fake_store.get_collection_info.return_value = {
        "name": "test_collection",
        "count": 0,
        "metadata": {},
    }

    with TestClient(api_main.app) as client:
        resp = client.post("/query", json={"query": "anything"})

    assert resp.status_code == 400
    assert "ingest" in resp.json()["detail"].lower()
    fake_graph.generate_report.assert_not_called()


def test_a_graph_error_is_reported_as_a_server_error(monkeypatch):
    _store, _pipeline, fake_graph = _setup_app_with_fakes(monkeypatch)
    fake_graph.generate_report.return_value = {"error": "chroma unreachable"}

    with TestClient(api_main.app) as client:
        resp = client.post("/query", json={"query": "anything"})

    assert resp.status_code == 500
    assert "chroma unreachable" in resp.json()["detail"]


def test_an_empty_query_is_rejected_before_it_reaches_the_graph(monkeypatch):
    _store, _pipeline, fake_graph = _setup_app_with_fakes(monkeypatch)

    with TestClient(api_main.app) as client:
        resp = client.post("/query", json={"query": ""})

    assert resp.status_code == 422
    fake_graph.generate_report.assert_not_called()


def test_an_unsupported_upload_is_rejected_by_extension(monkeypatch):
    _store, fake_pipeline, _graph = _setup_app_with_fakes(monkeypatch)

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/ingest/file",
            files={"file": ("budget.xlsx", b"binary", "application/octet-stream")},
        )

    assert resp.status_code == 400
    assert ".xlsx" in resp.json()["detail"]
    fake_pipeline.ingest_file.assert_not_called()


def test_an_uploaded_file_is_ingested_and_its_temp_copy_removed(monkeypatch):
    _store, fake_pipeline, _graph = _setup_app_with_fakes(monkeypatch)
    seen = {}

    def record(path):
        seen["path"] = path
        return 4

    fake_pipeline.ingest_file.side_effect = record

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/ingest/file",
            files={"file": ("notes.md", b"# Notes\n\nSome content.\n", "text/markdown")},
        )

    assert resp.status_code == 200
    assert resp.json()["chunks_added"] == 4
    assert not Path(seen["path"]).exists()


def test_a_failed_ingestion_still_removes_the_temp_file(monkeypatch):
    """The cleanup runs in a finally, and must not mask the real failure."""
    _store, fake_pipeline, _graph = _setup_app_with_fakes(monkeypatch)
    seen = {}

    def explode(path):
        seen["path"] = path
        raise RuntimeError("splitter blew up")

    fake_pipeline.ingest_file.side_effect = explode

    with TestClient(api_main.app) as client:
        resp = client.post(
            "/ingest/file",
            files={"file": ("notes.md", b"content", "text/markdown")},
        )

    assert resp.status_code == 500
    assert "splitter blew up" in resp.json()["detail"]
    assert not Path(seen["path"]).exists()


def test_stats_endpoint_reports_the_collection(monkeypatch):
    fake_store, _pipeline, _graph = _setup_app_with_fakes(monkeypatch)
    fake_store.get_collection_info.return_value = {
        "name": "company_data",
        "count": 42,
        "metadata": {"note": "x"},
    }

    with TestClient(api_main.app) as client:
        resp = client.get("/stats")

    assert resp.status_code == 200
    assert resp.json()["total_documents"] == 42
    assert resp.json()["collection_name"] == "company_data"
