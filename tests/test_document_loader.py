"""
Tests for document loading.

The loader is the only place the project touches the filesystem on the
ingestion side, so the cases that matter are the unhappy ones: a path that is
not there, a format nothing can read, and a directory holding a mixture.
"""

from unittest.mock import Mock, patch

import pytest
from langchain.schema import Document

from src.utils.document_loader import DocumentLoader, label_sources


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "annual_report.md").write_text(
        "# Annual report\n\nRevenue grew twenty two percent.\n", encoding="utf-8"
    )
    (tmp_path / "notes.txt").write_text("A plain text note.\n", encoding="utf-8")
    (tmp_path / "spreadsheet.xlsx").write_bytes(b"not really a spreadsheet")
    return tmp_path


class TestSingleFile:
    def test_a_markdown_file_is_loaded_with_its_name_as_the_source(self, corpus):
        documents = DocumentLoader.load_file(str(corpus / "annual_report.md"))

        assert len(documents) == 1
        assert "Revenue grew twenty two percent." in documents[0].page_content
        # The label is the file name; the full path stays available under
        # source_path. A citation naming one developer's home directory is not
        # a citation anyone else can follow.
        assert documents[0].metadata["source"] == "annual_report.md"
        assert documents[0].metadata["source_path"] == str(corpus / "annual_report.md")

    def test_a_plain_text_file_is_loaded(self, corpus):
        documents = DocumentLoader.load_file(str(corpus / "notes.txt"))

        assert documents[0].page_content.strip() == "A plain text note."

    def test_a_missing_file_raises_rather_than_returning_nothing(self, corpus):
        with pytest.raises(FileNotFoundError):
            DocumentLoader.load_file(str(corpus / "absent.md"))

    def test_an_unsupported_extension_is_named_in_the_error(self, corpus):
        with pytest.raises(ValueError, match=r"\.xlsx"):
            DocumentLoader.load_file(str(corpus / "spreadsheet.xlsx"))

    def test_the_extension_check_is_case_insensitive(self, tmp_path):
        path = tmp_path / "REPORT.MD"
        path.write_text("Uppercase extension.\n", encoding="utf-8")

        assert DocumentLoader.load_file(str(path))[0].page_content.strip() == (
            "Uppercase extension."
        )

    def test_a_loader_failure_propagates_instead_of_being_swallowed(self, corpus):
        failing = Mock(side_effect=OSError("disk gone"))
        with patch.dict(DocumentLoader.SUPPORTED_EXTENSIONS, {".md": failing}):
            with pytest.raises(OSError, match="disk gone"):
                DocumentLoader.load_file(str(corpus / "annual_report.md"))


class TestDirectory:
    def test_supported_files_are_loaded_and_unsupported_ones_ignored(self, corpus):
        documents = DocumentLoader.load_directory(str(corpus))

        sources = {doc.metadata["source"] for doc in documents}
        assert "annual_report.md" in sources
        assert "notes.txt" in sources
        assert not any("spreadsheet.xlsx" in source for source in sources)

    def test_a_path_that_is_not_a_directory_raises(self, corpus):
        with pytest.raises(NotADirectoryError):
            DocumentLoader.load_directory(str(corpus / "annual_report.md"))

    def test_a_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            DocumentLoader.load_directory(str(tmp_path / "nowhere"))

    def test_an_empty_directory_yields_no_documents(self, tmp_path):
        assert DocumentLoader.load_directory(str(tmp_path)) == []

    def test_one_broken_format_does_not_lose_the_others(self, corpus):
        """
        Directory loading is deliberately per-extension and tolerant: a PDF
        that a parser chokes on must not discard the Markdown that loaded
        fine. It is logged as a partial failure rather than raised.
        """
        broken = Mock(side_effect=RuntimeError("pdf parser exploded"))
        with patch.dict(DocumentLoader.SUPPORTED_EXTENSIONS, {".pdf": broken}):
            documents = DocumentLoader.load_directory(str(corpus))

        assert any("annual_report.md" in doc.metadata["source"] for doc in documents)


class TestSourceLabelling:
    """
    A citation has to name something the reader can find. An absolute path from
    the machine that did the ingestion is neither portable nor meaningful, and
    it ends up in the report text and the API response, not just a log.
    """

    def test_an_absolute_path_is_reduced_to_a_file_name(self):
        documents = [
            Document(
                page_content="text",
                metadata={"source": "/Users/someone/projects/data/annual_report.md"},
            )
        ]

        label_sources(documents)

        assert documents[0].metadata["source"] == "annual_report.md"

    def test_the_full_path_is_kept_rather_than_discarded(self):
        original = "/srv/corpus/2025/annual_report.md"
        documents = [Document(page_content="text", metadata={"source": original})]

        label_sources(documents)

        assert documents[0].metadata["source_path"] == original

    def test_a_document_with_no_source_is_left_alone(self):
        documents = [Document(page_content="text", metadata={})]

        label_sources(documents)

        assert "source" not in documents[0].metadata
        assert "source_path" not in documents[0].metadata
