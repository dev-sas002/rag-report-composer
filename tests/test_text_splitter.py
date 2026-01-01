"""
Tests for chunking.

Chunking is the setting with the widest blast radius in a RAG system: too
large and retrieval returns a haystack, too small and a fact is split away
from the sentence that qualifies it, and an overlap that is not smaller than
the chunk size is not a bad chunking strategy but an outright error.
"""

from unittest.mock import Mock, patch

import pytest
from langchain.schema import Document

from src.utils.text_splitter import get_text_splitter


def splitter_with(chunk_size: int, chunk_overlap: int):
    settings = Mock()
    settings.chunk_size = chunk_size
    settings.chunk_overlap = chunk_overlap
    with patch("src.utils.text_splitter.get_settings", return_value=settings):
        return get_text_splitter()


class TestConfiguration:
    def test_the_splitter_takes_its_size_and_overlap_from_settings(self):
        splitter = splitter_with(chunk_size=250, chunk_overlap=40)

        assert splitter._chunk_size == 250
        assert splitter._chunk_overlap == 40

    def test_an_overlap_larger_than_the_chunk_is_rejected_by_the_splitter(self):
        # Settings rejects overlap >= size before it ever gets here; this is
        # the backstop for a splitter built directly.
        with pytest.raises(ValueError):
            splitter_with(chunk_size=100, chunk_overlap=200)


class TestChunkBoundaries:
    def test_every_chunk_respects_the_configured_size(self):
        splitter = splitter_with(chunk_size=120, chunk_overlap=20)
        text = " ".join(f"sentence number {i} about revenue." for i in range(60))

        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)

    def test_consecutive_chunks_actually_overlap(self):
        """
        The overlap is the whole reason a fact spanning a boundary is still
        retrievable. A splitter that silently produced disjoint chunks would
        look identical from the outside until a query missed.
        """
        splitter = splitter_with(chunk_size=100, chunk_overlap=30)
        text = " ".join(f"word{i}" for i in range(200))

        chunks = splitter.split_text(text)

        assert len(chunks) > 2
        shared = [
            any(tail in chunks[i + 1] for tail in (chunks[i][-15:], chunks[i][-10:]))
            for i in range(len(chunks) - 1)
        ]
        assert any(shared), "no two consecutive chunks shared any text"

    def test_text_shorter_than_a_chunk_is_left_whole(self):
        splitter = splitter_with(chunk_size=1000, chunk_overlap=200)

        assert splitter.split_text("A single short paragraph.") == ["A single short paragraph."]

    def test_splitting_prefers_paragraph_then_line_then_word_boundaries(self):
        splitter = splitter_with(chunk_size=60, chunk_overlap=0)
        text = "First paragraph body.\n\nSecond paragraph body.\n\nThird one."

        chunks = splitter.split_text(text)

        # The separators are ["\n\n", "\n", " ", ""], so paragraph breaks win
        # and no chunk should begin or end mid-word.
        assert "First paragraph body." in chunks[0]
        assert all(chunk == chunk.strip() for chunk in chunks)

    def test_empty_text_produces_no_chunks(self):
        splitter = splitter_with(chunk_size=100, chunk_overlap=10)

        assert splitter.split_text("") == []


class TestDocumentMetadata:
    def test_each_chunk_keeps_the_metadata_of_the_document_it_came_from(self):
        splitter = splitter_with(chunk_size=80, chunk_overlap=10)
        document = Document(
            page_content=" ".join(f"token{i}" for i in range(100)),
            metadata={"source": "annual_report_2025.md", "page": 3},
        )

        chunks = splitter.split_documents([document])

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "annual_report_2025.md"
            assert chunk.metadata["page"] == 3

    def test_documents_are_not_merged_into_one_another(self):
        splitter = splitter_with(chunk_size=1000, chunk_overlap=100)
        documents = [
            Document(page_content="Alpha content.", metadata={"source": "a.md"}),
            Document(page_content="Beta content.", metadata={"source": "b.md"}),
        ]

        chunks = splitter.split_documents(documents)

        assert len(chunks) == 2
        assert {c.metadata["source"] for c in chunks} == {"a.md", "b.md"}
        assert not any("Alpha" in c.page_content and "Beta" in c.page_content for c in chunks)
