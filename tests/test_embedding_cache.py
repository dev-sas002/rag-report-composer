"""
Tests for the embedding cache.

The cache exists to stop the pipeline paying twice for the same vector. The
property that makes it safe is narrower than "it caches": a cached vector must
never be served for a different model, because a wrong-but-plausible vector
does not raise — it silently retrieves the wrong chunks.
"""

from langchain_core.embeddings import Embeddings

from src.retrieval.embedding_cache import (
    CachingEmbeddings,
    EmbeddingStore,
    cache_key,
    wrap_with_cache,
)


class CountingEmbeddings(Embeddings):
    """Records how much work it was actually asked to do."""

    def __init__(self, dimensions: int = 4) -> None:
        self.dimensions = dimensions
        self.documents_embedded = 0
        self.queries_embedded = 0

    def _vector(self, text: str):
        return [float(len(text) + i) for i in range(self.dimensions)]

    def embed_documents(self, texts):
        self.documents_embedded += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        self.queries_embedded += 1
        return self._vector(text)


def cached(tmp_path, inner=None, model_name="test-model"):
    inner = inner or CountingEmbeddings()
    return CachingEmbeddings(
        inner, EmbeddingStore(tmp_path / "cache.sqlite"), model_name=model_name
    )


class TestKeying:
    def test_the_same_model_and_text_give_the_same_key(self):
        assert cache_key("m", "hello") == cache_key("m", "hello")

    def test_a_different_model_gives_a_different_key(self):
        assert cache_key("openai", "hello") != cache_key("local", "hello")

    def test_the_separator_stops_a_boundary_collision(self):
        """
        Concatenating model and text without a separator makes ("ab", "c") and
        ("a", "bc") the same key. Different vectors under one key is the one
        failure mode a cache must not have.
        """
        assert cache_key("ab", "c") != cache_key("a", "bc")


class TestReuse:
    def test_a_repeated_document_is_embedded_once(self, tmp_path):
        inner = CountingEmbeddings()
        embeddings = cached(tmp_path, inner)

        first = embeddings.embed_documents(["alpha", "beta"])
        second = embeddings.embed_documents(["alpha", "beta"])

        assert first == second
        assert inner.documents_embedded == 2

    def test_duplicates_inside_one_batch_are_embedded_once(self, tmp_path):
        """
        Ingestion sends a whole file in one call. Two identical chunks in that
        call both missed the cache and were both embedded before either was
        written back.
        """
        inner = CountingEmbeddings()
        embeddings = cached(tmp_path, inner)

        result = embeddings.embed_documents(["same", "same", "same"])

        assert inner.documents_embedded == 1
        assert result[0] == result[1] == result[2]

    def test_a_partially_cached_batch_only_embeds_what_is_missing(self, tmp_path):
        inner = CountingEmbeddings()
        embeddings = cached(tmp_path, inner)
        embeddings.embed_documents(["alpha"])

        embeddings.embed_documents(["alpha", "beta"])

        assert inner.documents_embedded == 2

    def test_results_come_back_in_the_order_they_were_asked_for(self, tmp_path):
        """
        A cache that returns hits first and misses after would silently
        misalign vectors with their documents.
        """
        inner = CountingEmbeddings()
        embeddings = cached(tmp_path, inner)
        embeddings.embed_documents(["bbb"])

        result = embeddings.embed_documents(["aa", "bbb", "cccc"])

        assert result == [inner._vector(t) for t in ["aa", "bbb", "cccc"]]

    def test_a_repeated_query_is_embedded_once(self, tmp_path):
        inner = CountingEmbeddings()
        embeddings = cached(tmp_path, inner)

        embeddings.embed_query("how much revenue?")
        embeddings.embed_query("how much revenue?")

        assert inner.queries_embedded == 1

    def test_an_empty_batch_touches_neither_the_store_nor_the_backend(self, tmp_path):
        inner = CountingEmbeddings()
        embeddings = cached(tmp_path, inner)

        assert embeddings.embed_documents([]) == []
        assert inner.documents_embedded == 0


class TestIsolationBetweenModels:
    def test_two_models_sharing_a_file_do_not_share_vectors(self, tmp_path):
        """
        The failure this prevents is invisible at runtime: a 384-dimension
        vector served to a 3072-dimension index retrieves nonsense rather than
        raising.
        """
        store = EmbeddingStore(tmp_path / "shared.sqlite")
        first = CachingEmbeddings(CountingEmbeddings(4), store, model_name="model-a")
        second = CachingEmbeddings(CountingEmbeddings(8), store, model_name="model-b")

        first.embed_documents(["alpha"])
        result = second.embed_documents(["alpha"])

        assert len(result[0]) == 8


class TestPersistenceAndVisibility:
    def test_vectors_survive_a_new_process_reading_the_same_file(self, tmp_path):
        path = tmp_path / "cache.sqlite"
        inner = CountingEmbeddings()
        CachingEmbeddings(inner, EmbeddingStore(path), "m").embed_documents(["alpha"])

        reopened = CachingEmbeddings(inner, EmbeddingStore(path), "m")
        reopened.embed_documents(["alpha"])

        assert inner.documents_embedded == 1

    def test_the_hit_rate_is_observable(self, tmp_path):
        """A cache whose hit rate nobody can see is a cache nobody can justify."""
        embeddings = cached(tmp_path)

        embeddings.embed_documents(["alpha", "beta"])
        embeddings.embed_documents(["alpha", "beta"])

        assert embeddings.hits == 2
        assert embeddings.misses == 2
        assert embeddings.hit_rate == 0.5

    def test_the_hit_rate_of_an_unused_cache_is_zero_not_an_error(self, tmp_path):
        assert cached(tmp_path).hit_rate == 0.0

    def test_wrap_with_cache_creates_the_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "cache.sqlite"

        embeddings = wrap_with_cache(CountingEmbeddings(), str(path), "m")
        embeddings.embed_query("alpha")

        assert path.exists()

    def test_a_backend_without_an_explicit_name_is_named_after_itself(self, tmp_path):
        embeddings = wrap_with_cache(CountingEmbeddings(), str(tmp_path / "c.sqlite"), None)

        assert embeddings.model_name == "CountingEmbeddings"


class TestLargeBatches:
    def test_a_batch_larger_than_the_sqlite_parameter_limit_still_works(self, tmp_path):
        """
        `SELECT ... WHERE key IN (?, ?, ...)` has a bound-parameter ceiling.
        A directory ingestion overshoots it easily, and the failure would be an
        OperationalError from deep inside the cache.
        """
        inner = CountingEmbeddings()
        embeddings = cached(tmp_path, inner)
        texts = [f"chunk number {i}" for i in range(1200)]

        first = embeddings.embed_documents(texts)
        second = embeddings.embed_documents(texts)

        assert first == second
        assert inner.documents_embedded == 1200
