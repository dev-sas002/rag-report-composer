"""Tests for vector store functionality."""

from unittest.mock import Mock, patch

import pytest
from langchain.schema import Document

from src.retrieval.vector_store import VectorStore, VectorStoreBackend


@pytest.fixture
def settings():
    """Mock settings for testing."""
    with patch("src.retrieval.vector_store.get_settings") as mock:
        settings = Mock()
        settings.chroma_persist_directory = "./test_data/chroma_db"
        settings.chroma_collection_name = "test_collection"
        settings.top_k_results = 3
        settings.vector_backend = "chroma"
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_backend():
    """Fake Chroma-like backend for unit tests."""
    backend = Mock(spec=VectorStoreBackend)
    backend.add_documents.return_value = ["id1", "id2"]
    backend.similarity_search.return_value = [
        Document(page_content="result 1", metadata={}),
        Document(page_content="result 2", metadata={}),
    ]
    backend.similarity_search_with_score.return_value = [
        (Document(page_content="result 1", metadata={}), 0.1),
        (Document(page_content="result 2", metadata={}), 0.2),
    ]
    backend.as_retriever.return_value = Mock()
    return backend


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            page_content="This is a test document about AI.", metadata={"source": "test1.txt"}
        ),
        Document(
            page_content="Machine learning is a subset of AI.", metadata={"source": "test2.txt"}
        ),
        Document(
            page_content="Deep learning uses neural networks.", metadata={"source": "test3.txt"}
        ),
    ]


def test_vector_store_initialization_uses_backend_injection(settings, mock_backend):
    """VectorStore should use an injected backend without touching Chroma."""
    with (
        patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls,
        patch("src.retrieval.vector_store.get_embeddings"),
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    # An injected client means Chroma is never constructed — which is the
    # whole point of the injection, and what the previous assertion
    # (`assert_called_once`) contradicted.
    client_cls.assert_not_called()
    assert store.client is client_cls.return_value
    assert store.vectorstore is mock_backend


def test_add_documents_with_empty_list_is_noop(settings, mock_backend, caplog):
    """Adding no documents should be a no-op and return an empty list."""
    with (
        patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls,
        patch("src.retrieval.vector_store.get_embeddings"),
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    with caplog.at_level("WARNING"):
        ids = store.add_documents([])

    assert ids == []
    assert "no_documents_to_add" in caplog.text
    mock_backend.add_documents.assert_not_called()


def test_add_documents_delegates_to_backend(settings, mock_backend, sample_documents):
    """Adding documents delegates to the backend and logs success."""
    with (
        patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls,
        patch("src.retrieval.vector_store.get_embeddings"),
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    ids = store.add_documents(sample_documents)

    mock_backend.add_documents.assert_called_once_with(sample_documents)
    assert ids == ["id1", "id2"]


def test_similarity_search_uses_default_k(settings, mock_backend):
    """Similarity search uses settings.top_k_results when k is not provided."""
    with (
        patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls,
        patch("src.retrieval.vector_store.get_embeddings"),
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    results = store.similarity_search("query text")

    mock_backend.similarity_search.assert_called_once_with("query text", k=settings.top_k_results)
    assert len(results) == 2


def test_similarity_search_with_score_uses_default_k(settings, mock_backend):
    """Similarity search with score uses settings.top_k_results when k is not provided."""
    with (
        patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls,
        patch("src.retrieval.vector_store.get_embeddings"),
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    results = store.similarity_search_with_score("query text")

    mock_backend.similarity_search_with_score.assert_called_once_with(
        "query text", k=settings.top_k_results
    )
    assert len(results) == 2
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert isinstance(score, float)


def test_get_retriever_uses_backend(settings, mock_backend):
    """get_retriever should delegate to the backend retriever factory."""
    with (
        patch("src.retrieval.vector_store.chromadb.PersistentClient") as client_cls,
        patch("src.retrieval.vector_store.get_embeddings"),
    ):
        store = VectorStore(client=client_cls.return_value, vectorstore=mock_backend)

    retriever = store.get_retriever()

    mock_backend.as_retriever.assert_called_once()
    assert retriever is mock_backend.as_retriever.return_value


class TestEmbeddingDimensionGuard:
    """
    Reindexing with a different embedding backend against the same Chroma
    directory leaves vectors of one width being queried with another. Chroma
    raises, but at query time and with a message that names neither setting,
    so the cause reads as a Chroma bug rather than a configuration change.
    """

    def _store(self, settings, mock_backend, stored_dimensions):
        collection = Mock()
        collection.count.return_value = 7
        collection.name = "test_collection"
        collection.peek.return_value = {"embeddings": [[0.0] * stored_dimensions]}

        client = Mock()
        client.get_collection.return_value = collection

        with (
            patch("src.retrieval.vector_store.get_embeddings"),
            patch(
                "src.retrieval.vector_store.create_vector_store_backend",
                return_value=mock_backend,
            ),
        ):
            return VectorStore(client=client), collection

    def test_a_mismatch_is_reported_with_both_widths(self, settings, mock_backend, caplog):
        settings.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        with caplog.at_level("ERROR"):
            self._store(settings, mock_backend, stored_dimensions=3072)

        assert "embedding_dimension_mismatch" in caplog.text
        assert "3072" in caplog.text
        assert "384" in caplog.text

    def test_a_matching_backend_is_silent(self, settings, mock_backend, caplog):
        settings.embedding_model_name = "text-embedding-3-large"

        with caplog.at_level("ERROR"):
            self._store(settings, mock_backend, stored_dimensions=3072)

        assert "embedding_dimension_mismatch" not in caplog.text

    def test_an_unknown_model_skips_the_check_rather_than_guessing(
        self, settings, mock_backend, caplog
    ):
        settings.embedding_model_name = "some-future-model"

        with caplog.at_level("ERROR"):
            self._store(settings, mock_backend, stored_dimensions=99)

        assert "embedding_dimension_mismatch" not in caplog.text


class TestCollectionInfo:
    def _store(self, settings, mock_backend, client):
        with patch("src.retrieval.vector_store.get_embeddings"):
            return VectorStore(client=client, vectorstore=mock_backend)

    def test_collection_info_reports_name_and_count(self, settings, mock_backend):
        collection = Mock()
        collection.name = "company_data"
        collection.count.return_value = 12
        collection.metadata = {"kind": "demo"}
        client = Mock()
        client.get_collection.return_value = collection

        info = self._store(settings, mock_backend, client).get_collection_info()

        assert info == {"name": "company_data", "count": 12, "metadata": {"kind": "demo"}}

    def test_a_missing_collection_reads_as_empty_rather_than_raising(self, settings, mock_backend):
        """
        The CLI and the API both call this before doing anything else; an
        exception here would turn "you have not ingested yet" into a crash.
        """
        client = Mock()
        client.get_collection.side_effect = ValueError("collection does not exist")

        info = self._store(settings, mock_backend, client).get_collection_info()

        assert info["count"] == 0
        assert info["name"] == settings.chroma_collection_name


class TestErrorPropagation:
    def test_an_add_failure_is_raised_not_swallowed(self, settings, mock_backend):
        with patch("src.retrieval.vector_store.get_embeddings"):
            store = VectorStore(client=Mock(), vectorstore=mock_backend)
        mock_backend.add_documents.side_effect = RuntimeError("quota exceeded")

        with pytest.raises(RuntimeError, match="quota exceeded"):
            store.add_documents([Document(page_content="x", metadata={})])

    def test_a_search_failure_is_raised_not_swallowed(self, settings, mock_backend):
        with patch("src.retrieval.vector_store.get_embeddings"):
            store = VectorStore(client=Mock(), vectorstore=mock_backend)
        mock_backend.similarity_search.side_effect = RuntimeError("index corrupt")

        with pytest.raises(RuntimeError, match="index corrupt"):
            store.similarity_search("q")

    def test_an_explicit_k_overrides_the_configured_default(self, settings, mock_backend):
        with patch("src.retrieval.vector_store.get_embeddings"):
            store = VectorStore(client=Mock(), vectorstore=mock_backend)

        store.similarity_search("q", k=11)

        mock_backend.similarity_search.assert_called_once_with("q", k=11)
