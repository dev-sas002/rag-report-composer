"""
Tests for embedding backend selection.

No test here constructs a real client or issues a real call: what is being
checked is which backend gets chosen and what limits it is handed, because
getting that wrong either bills an account that was meant to be offline or
leaves a network call with no timeout on it.
"""

from unittest.mock import Mock, patch

import pytest

from src.config import Settings
from src.retrieval.embedding_cache import CachingEmbeddings
from src.retrieval.embeddings import get_embeddings
from src.retrieval.hashing_embeddings import HashingEmbeddings
from src.retrieval.local_embeddings import LocalEmbeddings


def fake_settings(**overrides) -> Settings:
    """
    A real `Settings`, not a Mock.

    Backend selection reads several fields that derive from one another
    (`use_openai`, `resolved_embedding_provider`, `embedding_model_name`). A
    Mock answers every one of them with a Mock, so a test against it passes
    whatever the selection logic does — which is how a half-wired registry
    lookup went unnoticed.
    """
    values = {
        "openai_api_key": "sk-test",
        "openai_embedding_model": "text-embedding-3-large",
        "openai_request_timeout": 45.0,
        "openai_max_retries": 4,
        "local_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "enable_embedding_cache": False,
    }
    values.update(overrides)
    return Settings(**values)


class TestBackendSelection:
    def test_the_local_backend_is_used_when_openai_is_not_configured(self):
        embeddings = get_embeddings(fake_settings(openai_api_key=None))

        assert isinstance(embeddings, LocalEmbeddings)
        assert embeddings.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_the_openai_backend_is_used_when_configured(self):
        with patch("src.retrieval.embeddings.TrackedOpenAIEmbeddings") as tracked:
            embeddings = get_embeddings(fake_settings())

        assert embeddings is tracked.return_value
        tracked.assert_called_once()

    def test_openai_embeddings_are_given_a_timeout_and_a_retry_budget(self):
        """
        Ingestion issues one embedding call per batch. Without a timeout a
        single stalled request stalls the whole run, and the symptom is a
        process that looks busy and never finishes.
        """
        with patch("src.retrieval.embeddings.TrackedOpenAIEmbeddings") as tracked:
            get_embeddings(fake_settings())

        kwargs = tracked.call_args.kwargs
        assert kwargs["timeout"] == 45.0
        assert kwargs["max_retries"] == 4
        assert kwargs["model"] == "text-embedding-3-large"

    def test_an_explicit_provider_overrides_the_key_based_default(self):
        """`EMBEDDING_PROVIDER` names a backend; a key present does not veto it."""
        embeddings = get_embeddings(fake_settings(embedding_provider="hashing"))

        assert isinstance(embeddings, HashingEmbeddings)

    def test_an_unregistered_provider_fails_by_name(self):
        from src.providers import ProviderNotRegisteredError

        with pytest.raises(ProviderNotRegisteredError, match="nowhere"):
            get_embeddings(fake_settings(embedding_provider="nowhere"))

    def test_the_default_configuration_puts_a_cache_in_front(self, tmp_path):
        """
        Caching is on by default, because embedding is a pure function of
        (model, text) and re-running it is money for nothing.
        """
        embeddings = get_embeddings(
            fake_settings(
                openai_api_key=None,
                embedding_provider="hashing",
                enable_embedding_cache=True,
                embedding_cache_path=str(tmp_path / "cache.sqlite"),
            )
        )

        assert isinstance(embeddings, CachingEmbeddings)
        assert isinstance(embeddings.inner, HashingEmbeddings)

    def test_the_cache_is_namespaced_by_the_model_that_filled_it(self, tmp_path):
        """
        A vector from the wrong model does not raise, it silently retrieves the
        wrong chunks. The cache key has to separate the backends.
        """
        openai_like = get_embeddings(
            fake_settings(
                enable_embedding_cache=True,
                embedding_cache_path=str(tmp_path / "cache.sqlite"),
                embedding_provider="openai",
            )
        )
        hashed = get_embeddings(
            fake_settings(
                enable_embedding_cache=True,
                embedding_cache_path=str(tmp_path / "cache.sqlite"),
                embedding_provider="hashing",
            )
        )

        assert openai_like.model_name != hashed.model_name


class TestLocalEmbeddings:
    """The local model is loaded lazily; these tests never load it."""

    def test_the_model_is_not_loaded_at_construction(self):
        embeddings = LocalEmbeddings("some-model")

        assert embeddings._model is None

    def test_embedding_no_documents_does_not_load_the_model(self):
        embeddings = LocalEmbeddings("some-model")

        assert embeddings.embed_documents([]) == []
        assert embeddings._model is None

    def test_documents_are_embedded_as_lists_of_floats(self):
        embeddings = LocalEmbeddings("some-model")
        vector = Mock()
        vector.tolist.return_value = [0.1, 0.2, 0.3]
        embeddings._model = Mock()
        embeddings._model.encode.return_value = [vector, vector]

        result = embeddings.embed_documents(["a", "b"])

        assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        assert embeddings._model.encode.call_args.kwargs["normalize_embeddings"] is True

    def test_a_query_is_embedded_as_a_single_vector(self):
        embeddings = LocalEmbeddings("some-model")
        vector = Mock()
        vector.tolist.return_value = [0.4, 0.5]
        embeddings._model = Mock()
        embeddings._model.encode.return_value = [vector]

        assert embeddings.embed_query("what is revenue?") == [0.4, 0.5]


class TestCostTrackedEmbeddings:
    def test_embedding_documents_counts_tokens_and_records_the_call(self):
        from src.retrieval.embeddings import TrackedOpenAIEmbeddings

        tracker = Mock()
        tracker.count_tokens.side_effect = lambda text: len(text.split())

        embeddings = TrackedOpenAIEmbeddings.__new__(TrackedOpenAIEmbeddings)
        object.__setattr__(embeddings, "_cost_tracker", tracker)

        with (
            patch(
                "src.retrieval.embeddings.OpenAIEmbeddings.embed_documents",
                return_value=[[0.0], [0.0]],
            ),
            patch.object(TrackedOpenAIEmbeddings, "model", "text-embedding-3-large", create=True),
        ):
            result = embeddings.embed_documents(["one two", "three four five"])

        assert result == [[0.0], [0.0]]
        kwargs = tracker.track_call.call_args.kwargs
        assert kwargs["operation"] == "embed_documents"
        assert kwargs["input_tokens"] == 5
        assert kwargs["output_tokens"] == 0

    def test_embedding_a_query_records_the_call(self):
        from src.retrieval.embeddings import TrackedOpenAIEmbeddings

        tracker = Mock()
        tracker.count_tokens.return_value = 3

        embeddings = TrackedOpenAIEmbeddings.__new__(TrackedOpenAIEmbeddings)
        object.__setattr__(embeddings, "_cost_tracker", tracker)

        with (
            patch(
                "src.retrieval.embeddings.OpenAIEmbeddings.embed_query",
                return_value=[0.1, 0.2],
            ),
            patch.object(TrackedOpenAIEmbeddings, "model", "text-embedding-3-large", create=True),
        ):
            assert embeddings.embed_query("how much revenue?") == [0.1, 0.2]

        assert tracker.track_call.call_args.kwargs["operation"] == "embed_query"
