"""
Vector store management.

A thin wrapper over whichever backend `VECTOR_BACKEND` selects from the
provider registry — persistent ChromaDB by default, an in-process store for
benchmarks and tests. The wrapper is what the rest of the project talks to, so
swapping the backend does not reach any caller.
"""

import os
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

import chromadb
from langchain.schema import Document

from ..config import get_settings
from ..observability.logger import get_logger
from ..providers import create_vector_store_backend
from .embeddings import get_embeddings

logger = get_logger(__name__)

# Backends that keep their vectors in a Chroma collection on disk, and so need
# a chromadb client and support the collection-level operations below. Anything
# else is served entirely by the backend object itself.
PERSISTENT_BACKENDS = {"chroma"}


def chroma_client_settings() -> chromadb.config.Settings:
    """
    Chroma client configuration, with telemetry off unless asked for.

    ChromaDB posts usage events to a third party by default. A tool for
    analysing a company's own documents should not phone home without being
    asked, and in this environment the attempt also fails noisily on every
    command — a posthog version mismatch prints four lines of
    `Failed to send telemetry event` before any output the user wanted.

    Reading the environment variable rather than hard-coding `False` keeps the
    opt-in available to an operator who wants it.
    """
    enabled = os.environ.get("ANONYMIZED_TELEMETRY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    return chromadb.config.Settings(anonymized_telemetry=enabled)


# Output width of the embedding backends this project ships with. Used only to
# detect a persisted collection that was built with a different one; an
# unlisted model simply skips the check.
KNOWN_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
}


@runtime_checkable
class VectorStoreBackend(Protocol):
    """Minimal protocol for Chroma-like vector stores used by the agent."""

    def add_documents(self, documents: List[Document]) -> List[str]:  # pragma: no cover - protocol
        ...

    def similarity_search(
        self, query: str, k: int
    ) -> List[Document]:  # pragma: no cover - protocol
        ...

    def similarity_search_with_score(  # pragma: no cover - protocol
        self, query: str, k: int
    ) -> List[tuple[Document, float]]: ...

    def as_retriever(self, *args, **kwargs):  # pragma: no cover - protocol
        ...


class VectorStore:
    """
    Vector store wrapper for ChromaDB.
    Provides high-level interface for document storage and retrieval.
    """

    def __init__(
        self,
        client: Optional[chromadb.PersistentClient] = None,
        vectorstore: Optional[VectorStoreBackend] = None,
    ):
        """
        Initialize the vector store.

        The client and underlying vector store can be injected for tests to avoid
        touching the real Chroma persistence layer.
        """
        self.settings = get_settings()
        self.embeddings = get_embeddings()
        self.backend_name = (self.settings.vector_backend or "chroma").strip().lower()

        # Only the persistent backends need a chromadb client. Creating one for
        # the in-memory backend would put a directory on disk for a store whose
        # entire point is not touching it.
        if client is None and self.backend_name in PERSISTENT_BACKENDS:
            persist_dir = Path(self.settings.chroma_persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(persist_dir), settings=chroma_client_settings()
            )

        self.client: Optional[chromadb.PersistentClient] = client

        # Initialize or use provided vector store implementation
        self.vectorstore: VectorStoreBackend
        if vectorstore is not None:
            self.vectorstore = vectorstore
        else:
            self._initialize_vectorstore()

    def _build_backend(self) -> VectorStoreBackend:
        return create_vector_store_backend(
            self.settings, embeddings=self.embeddings, client=self.client
        )

    def _initialize_vectorstore(self) -> None:
        """Initialize or load the configured backend."""
        try:
            self.vectorstore = self._build_backend()

            if self.client is None:
                logger.info("vectorstore_initialized", backend=self.backend_name)
                return

            # Check if collection exists and has documents
            collection = self.client.get_collection(self.settings.chroma_collection_name)
            doc_count = collection.count()

            logger.info(
                "vectorstore_initialized",
                backend=self.backend_name,
                collection_name=self.settings.chroma_collection_name,
                document_count=doc_count,
            )

            if doc_count:
                self._warn_on_dimension_mismatch(collection)
        except Exception as e:
            logger.warning("vectorstore_init_warning", error=str(e))
            # Create new collection
            self.vectorstore = self._build_backend()
            logger.info("vectorstore_created", collection_name=self.settings.chroma_collection_name)

    def _expected_embedding_dimension(self) -> Optional[int]:
        """Vector width the configured embedding backend produces, if known."""
        return KNOWN_EMBEDDING_DIMENSIONS.get(self.settings.embedding_model_name)

    def _warn_on_dimension_mismatch(self, collection) -> None:
        """
        Say plainly when the persisted vectors were built by another backend.

        Switching between the OpenAI and local embedders against the same
        persistence directory leaves Chroma holding vectors of one width and
        being queried with another. Chroma does raise, but at query time and
        with a message that names neither setting, so the cause reads as a
        Chroma bug rather than a configuration change.
        """
        expected = self._expected_embedding_dimension()
        if expected is None:
            return

        try:
            sample = collection.peek(limit=1)
            embeddings = sample.get("embeddings") if sample else None
            if embeddings is None or len(embeddings) == 0:
                return
            stored = len(embeddings[0])
        except Exception as e:  # pragma: no cover - backend-specific
            logger.debug("embedding_dimension_check_skipped", error=str(e))
            return

        if stored != expected:
            logger.error(
                "embedding_dimension_mismatch",
                collection_name=self.settings.chroma_collection_name,
                stored_dimensions=stored,
                configured_dimensions=expected,
                persist_directory=self.settings.chroma_persist_directory,
                remedy="clear the collection and re-ingest, or restore the "
                "embedding backend that built it",
            )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("no_documents_to_add")
            return []

        try:
            ids = self.vectorstore.add_documents(documents)
            logger.info("documents_added", count=len(documents), ids_count=len(ids))
            return ids
        except Exception as e:
            logger.error("documents_add_failed", error=str(e), count=len(documents))
            raise

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return (defaults to settings.top_k_results)

        Returns:
            List of similar documents
        """
        if k is None:
            k = self.settings.top_k_results

        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(
                "similarity_search_complete",
                query_length=len(query),
                k=k,
                results_found=len(results),
            )
            return results
        except Exception as e:
            logger.error("similarity_search_failed", error=str(e), query_length=len(query))
            raise

    def similarity_search_with_score(
        self, query: str, k: Optional[int] = None
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        if k is None:
            k = self.settings.top_k_results

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(
                "similarity_search_with_score_complete",
                query_length=len(query),
                k=k,
                results_found=len(results),
            )
            return results
        except Exception as e:
            logger.error(
                "similarity_search_with_score_failed",
                error=str(e),
                query_length=len(query),
            )
            raise

    def get_retriever(self, k: Optional[int] = None):
        """
        Get a retriever interface for the vector store.

        Args:
            k: Number of results to return

        Returns:
            VectorStoreRetriever instance
        """
        if k is None:
            k = self.settings.top_k_results

        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            if self.client is None:
                # A backend with no client owns its own storage; ask it.
                self.vectorstore.delete()
            else:
                self.client.delete_collection(self.settings.chroma_collection_name)
                # Reinitialize
                self._initialize_vectorstore()
            logger.info("collection_deleted", collection_name=self.settings.chroma_collection_name)
        except Exception as e:
            logger.error("collection_delete_failed", error=str(e))
            raise

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection metadata
        """
        try:
            if self.client is None:
                return {
                    "name": self.settings.chroma_collection_name,
                    "count": self.vectorstore.count(),
                    "metadata": {"backend": self.backend_name},
                }
            collection = self.client.get_collection(self.settings.chroma_collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata or {},
            }
        except Exception as e:
            logger.error("get_collection_info_failed", error=str(e))
            return {"name": self.settings.chroma_collection_name, "count": 0, "metadata": {}}
