"""
Tests for the in-process vector store.

It exists so latency can be benchmarked and the pipeline exercised end to end
without leaving an index on disk. That only holds if it behaves like the Chroma
wrapper it stands in for, so these tests pin the parts of that contract the
agent actually depends on: ordering, `k`, and the score's polarity.
"""

import pytest
from langchain.schema import Document

from src.retrieval.hashing_embeddings import HashingEmbeddings
from src.retrieval.memory_store import InMemoryVectorStore, cosine_distance


@pytest.fixture
def store():
    return InMemoryVectorStore(HashingEmbeddings(dimensions=64))


DOCUMENTS = [
    Document(page_content="revenue grew twenty two percent", metadata={"source": "a.md"}),
    Document(page_content="net promoter score rose to forty one", metadata={"source": "b.md"}),
    Document(page_content="the platform ships four modules", metadata={"source": "c.md"}),
]


class TestDistance:
    def test_identical_vectors_are_at_distance_zero(self):
        assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors_are_at_distance_one(self):
        assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)

    def test_opposite_vectors_are_at_distance_two(self):
        assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)

    def test_a_zero_vector_does_not_divide_by_zero(self):
        """An empty chunk embeds to zeros; it must not crash the search."""
        assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


class TestSearch:
    def test_the_closest_document_comes_back_first(self, store):
        store.add_documents(DOCUMENTS)

        results = store.similarity_search("revenue grew", k=3)

        assert results[0].metadata["source"] == "a.md"

    def test_k_bounds_the_number_of_results(self, store):
        store.add_documents(DOCUMENTS)

        assert len(store.similarity_search("revenue", k=2)) == 2

    def test_scores_are_distances_so_lower_is_closer(self, store):
        """
        Chroma's convention. Getting the polarity wrong here would invert
        relevance pruning, which drops the *best* chunks rather than the worst.
        """
        store.add_documents(DOCUMENTS)

        scored = store.similarity_search_with_score("revenue grew", k=3)

        assert [s for _, s in scored] == sorted(s for _, s in scored)

    def test_searching_an_empty_store_returns_nothing_rather_than_raising(self, store):
        assert store.similarity_search("anything", k=5) == []

    def test_adding_no_documents_is_a_no_op(self, store):
        assert store.add_documents([]) == []
        assert store.count() == 0


class TestRetrieverInterface:
    def test_the_retriever_honours_the_configured_k(self, store):
        store.add_documents(DOCUMENTS)

        retriever = store.as_retriever(search_kwargs={"k": 1})

        assert len(retriever.invoke("revenue")) == 1

    def test_the_older_langchain_entry_point_also_works(self, store):
        store.add_documents(DOCUMENTS)

        retriever = store.as_retriever(search_kwargs={"k": 2})

        assert len(retriever.get_relevant_documents("revenue")) == 2


class TestLifecycle:
    def test_documents_are_counted(self, store):
        store.add_documents(DOCUMENTS)

        assert store.count() == 3

    def test_delete_empties_the_store(self, store):
        store.add_documents(DOCUMENTS)

        store.delete()

        assert store.count() == 0
        assert store.similarity_search("revenue", k=3) == []

    def test_every_added_document_gets_a_distinct_id(self, store):
        ids = store.add_documents(DOCUMENTS)

        assert len(set(ids)) == 3


class TestHashingEmbeddings:
    def test_the_same_text_always_gives_the_same_vector(self):
        embeddings = HashingEmbeddings(dimensions=32)

        assert embeddings.embed_query("revenue") == embeddings.embed_query("revenue")

    def test_vectors_have_the_requested_width(self):
        assert len(HashingEmbeddings(dimensions=17).embed_query("revenue")) == 17

    def test_vectors_are_normalised(self):
        vector = HashingEmbeddings(dimensions=32).embed_query("revenue grew sharply")

        assert sum(v * v for v in vector) == pytest.approx(1.0)

    def test_text_with_no_tokens_yields_a_zero_vector_rather_than_an_error(self):
        assert HashingEmbeddings(dimensions=8).embed_query("!!! ???") == [0.0] * 8

    def test_a_non_positive_width_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            HashingEmbeddings(dimensions=0)
