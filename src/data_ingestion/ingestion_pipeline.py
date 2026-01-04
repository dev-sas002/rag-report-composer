"""
Data ingestion pipeline for loading company documents into the vector store.
Handles document loading, splitting, and storage.
"""

from typing import List, Optional, Protocol

from langchain.schema import Document

from ..observability.logger import get_logger
from ..retrieval.vector_store import VectorStore
from ..utils.document_loader import DocumentLoader
from ..utils.text_splitter import get_text_splitter

logger = get_logger(__name__)


class TextSplitter(Protocol):
    """Protocol for text splitters used by the ingestion pipeline."""

    def split_documents(
        self, documents: List[Document]
    ) -> List[Document]:  # pragma: no cover - protocol
        ...


class IngestionPipeline:
    """
    Pipeline for ingesting company data into the vector store.
    Handles the full workflow from raw documents to stored embeddings.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        text_splitter: Optional[TextSplitter] = None,
        document_loader: Optional[DocumentLoader] = None,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            vector_store: VectorStore instance (creates new if not provided)
            text_splitter: Text splitter implementation (defaults to configured splitter)
            document_loader: DocumentLoader instance (defaults to concrete loader)
        """
        self.vector_store = vector_store or VectorStore()
        self.text_splitter = text_splitter or get_text_splitter()
        self.document_loader = document_loader or DocumentLoader()

    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single file into the vector store.

        Args:
            file_path: Path to the file

        Returns:
            Number of chunks added
        """
        logger.info("ingestion_started", file_path=file_path, type="file")

        try:
            # Load documents
            documents = self.document_loader.load_file(file_path)

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Add metadata
            for chunk in chunks:
                chunk.metadata["source_file"] = file_path
                chunk.metadata["ingestion_type"] = "file"

            # Store in vector store
            ids = self.vector_store.add_documents(chunks)

            logger.info(
                "ingestion_complete",
                file_path=file_path,
                num_chunks=len(chunks),
                num_ids=len(ids),
            )

            return len(chunks)

        except Exception as e:
            logger.error("ingestion_failed", file_path=file_path, error=str(e))
            raise

    def ingest_directory(self, directory_path: str, glob_pattern: str = "**/*") -> int:
        """
        Ingest all supported files from a directory.

        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file matching

        Returns:
            Total number of chunks added
        """
        logger.info(
            "ingestion_started",
            directory_path=directory_path,
            type="directory",
            glob_pattern=glob_pattern,
        )

        try:
            # Load all documents
            documents = self.document_loader.load_directory(directory_path, glob_pattern)

            if not documents:
                logger.warning("no_documents_found", directory_path=directory_path)
                return 0

            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Add metadata
            for chunk in chunks:
                chunk.metadata["source_directory"] = directory_path
                chunk.metadata["ingestion_type"] = "directory"

            # Store in vector store
            ids = self.vector_store.add_documents(chunks)

            logger.info(
                "ingestion_complete",
                directory_path=directory_path,
                num_documents=len(documents),
                num_chunks=len(chunks),
                num_ids=len(ids),
            )

            return len(chunks)

        except Exception as e:
            logger.error("ingestion_failed", directory_path=directory_path, error=str(e))
            raise

    def ingest_documents(self, documents: List[Document]) -> int:
        """
        Ingest pre-loaded documents.

        Args:
            documents: List of Document objects

        Returns:
            Number of chunks added
        """
        logger.info("ingestion_started", type="documents", num_documents=len(documents))

        try:
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)

            # Add metadata
            for chunk in chunks:
                chunk.metadata["ingestion_type"] = "direct"

            # Store in vector store
            ids = self.vector_store.add_documents(chunks)

            logger.info(
                "ingestion_complete",
                num_documents=len(documents),
                num_chunks=len(chunks),
                num_ids=len(ids),
            )

            return len(chunks)

        except Exception as e:
            logger.error("ingestion_failed", error=str(e), num_documents=len(documents))
            raise

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the ingested data.

        Returns:
            Dictionary with collection statistics
        """
        return self.vector_store.get_collection_info()

    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        logger.warning("clearing_collection")
        self.vector_store.delete_collection()
        logger.info("collection_cleared")
