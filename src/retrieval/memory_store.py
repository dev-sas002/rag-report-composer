"""
An in-process vector store implementing the slice of the Chroma interface the
agent actually uses.

Brute-force cosine similarity over a list. That is the right algorithm here and
the wrong one in production, which is the point: it makes the cost of *not*
having an index visible. `scripts/benchmark.py` measures both this and Chroma
at the same corpus sizes, and the gap between them is the argument for the
index rather than an assertion that one is needed.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple
from uuid import uuid4

from langchain.schema import Document


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Distance in [0, 2]. Lower is closer, matching Chroma's convention."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


class InMemoryVectorStore:
    """Holds documents and their vectors in a list. Nothing is persisted."""

    def __init__(self, embeddings: Any) -> None:
        self.embeddings = embeddings
        self._documents: List[Document] = []
        self._vectors: List[List[float]] = []
        self._ids: List[str] = []

    # -- the VectorStoreBackend protocol ---------------------------------

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []
        vectors = self.embeddings.embed_documents([d.page_content for d in documents])
        ids = [str(uuid4()) for _ in documents]
        self._documents.extend(documents)
        self._vectors.extend(vectors)
        self._ids.extend(ids)
        return ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k)]

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        if not self._documents:
            return []
        query_vector = self.embeddings.embed_query(query)
        scored = [
            (document, cosine_distance(query_vector, vector))
            for document, vector in zip(self._documents, self._vectors)
        ]
        scored.sort(key=lambda row: row[1])
        return scored[:k]

    def as_retriever(self, *_args: Any, **kwargs: Any) -> "_MemoryRetriever":
        k = (kwargs.get("search_kwargs") or {}).get("k", 4)
        return _MemoryRetriever(self, k)

    # -- the slice of the chroma client the VectorStore wrapper uses ------

    def count(self) -> int:
        return len(self._documents)

    def delete(self) -> None:
        self._documents.clear()
        self._vectors.clear()
        self._ids.clear()


class _MemoryRetriever:
    """LangChain-shaped retriever over `InMemoryVectorStore`."""

    def __init__(self, store: InMemoryVectorStore, k: int) -> None:
        self.store = store
        self.k = k

    def invoke(self, query: str, config: Optional[dict] = None) -> List[Document]:
        return self.store.similarity_search(query, k=self.k)

    # LangChain's older retriever entry point, kept so either call site works.
    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.store.similarity_search(query, k=self.k)
