"""
Document loader for the file formats the pipeline can index: PDF, DOCX, TXT, MD.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from ..observability.logger import get_logger

logger = get_logger(__name__)

#: Kept alongside the citation label, so the full path is still recoverable.
SOURCE_PATH_KEY = "source_path"


def label_sources(documents: List[Document]) -> List[Document]:
    """
    Replace each document's `source` with the file name it came from.

    LangChain's loaders record the path they were handed, so a report ingested
    from an absolute path cited
    `/Users/someone/projects/rag/data/raw_data/annual_report_2025.md` — one
    developer's home directory, in an API response and in the report text. The
    file name is what a citation is for. The full path moves to `source_path`,
    so nothing is lost, it is just no longer the label.
    """
    for document in documents:
        source = document.metadata.get("source")
        if not source:
            continue
        document.metadata[SOURCE_PATH_KEY] = source
        document.metadata["source"] = Path(source).name
    return documents


class DocxLoader:
    """
    Reads a .docx file with python-docx: paragraphs in order, then table rows.

    This replaces LangChain's `UnstructuredWordDocumentLoader`, which needs the
    `unstructured` package. That package was never in requirements.txt, so
    every Word upload failed with an ImportError while python-docx sat
    installed and unused. Legacy binary `.doc` is not supported by either
    library without LibreOffice, so it is no longer advertised.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load(self) -> List[Document]:
        import docx

        word_document = docx.Document(self.file_path)
        blocks = [p.text for p in word_document.paragraphs if p.text.strip()]
        for table in word_document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    blocks.append(" | ".join(cells))
        return [Document(page_content="\n\n".join(blocks), metadata={"source": self.file_path})]


class DocumentLoader:
    """Loads documents from various file formats."""

    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".docx": DocxLoader,
    }

    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """
        Load a single file.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects
        """
        path = Path(file_path)

        if not path.exists():
            logger.error("file_not_found", file_path=file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()

        if extension not in cls.SUPPORTED_EXTENSIONS:
            logger.warning("unsupported_file_type", file_path=file_path, extension=extension)
            raise ValueError(f"Unsupported file type: {extension}")

        loader_class = cls.SUPPORTED_EXTENSIONS[extension]

        try:
            loader = loader_class(str(path))
            documents = label_sources(loader.load())
            logger.info(
                "file_loaded",
                file_path=file_path,
                num_documents=len(documents),
                extension=extension,
            )
            return documents
        except Exception as e:
            logger.error("file_load_failed", file_path=file_path, error=str(e))
            raise

    @classmethod
    def load_directory(cls, directory_path: str, glob_pattern: str = "**/*") -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file matching

        Returns:
            List of Document objects
        """
        path = Path(directory_path)

        if not path.exists() or not path.is_dir():
            logger.error("directory_not_found", directory_path=directory_path)
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        all_documents = []

        for extension, loader_class in cls.SUPPORTED_EXTENSIONS.items():
            try:
                loader = DirectoryLoader(
                    str(path),
                    glob=f"{glob_pattern}{extension}",
                    loader_cls=loader_class,
                    show_progress=True,
                    use_multithreading=True,
                )
                documents = label_sources(loader.load())
                all_documents.extend(documents)
                logger.info(
                    "directory_files_loaded",
                    extension=extension,
                    num_documents=len(documents),
                )
            except Exception as e:
                logger.warning(
                    "directory_load_partial_failure",
                    extension=extension,
                    error=str(e),
                )

        logger.info(
            "directory_load_complete",
            directory_path=directory_path,
            total_documents=len(all_documents),
        )

        return all_documents
