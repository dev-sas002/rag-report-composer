"""Tests for the ReportWriter and report content builders."""

from src.report_writer import (
    ReportWriter,
    _build_markdown_report_content,
    _build_safe_filename,
    _build_text_report_content,
)


def test_safe_filename_sanitizes_query():
    filename = _build_safe_filename("report", "revenue: Q4/2024?", "txt")
    assert filename.startswith("report_")
    assert filename.endswith(".txt")
    # Ensure special characters have been replaced
    assert "revenue__Q4_2024_" in filename


def test_text_report_content_includes_sections():
    content = _build_text_report_content(
        query="What are revenues?",
        report="Detailed report body",
        summary="Summary content",
        sources=["a.txt", "b.txt"],
        metadata={"tokens_used": 123, "cost_usd": "$0.01"},
    )

    assert "COMPANY DATA REPORT" in content
    assert "EXECUTIVE SUMMARY" in content
    assert "DETAILED REPORT" in content
    assert "SOURCES" in content
    assert "METADATA" in content
    assert "a.txt" in content
    assert "tokens_used: 123" in content


def test_markdown_report_content_includes_sections():
    content = _build_markdown_report_content(
        query="What are revenues?",
        report="Detailed report body",
        summary="Summary content",
        sources=["a.txt"],
        metadata={"tokens_used": 123},
    )

    assert "# Company Data Report" in content
    assert "## Executive Summary" in content
    assert "## Detailed Report" in content
    assert "## Sources" in content
    assert "## Metadata" in content
    assert "a.txt" in content


def test_report_writer_saves_text_file(tmp_path, monkeypatch):
    writer = ReportWriter(output_dir=str(tmp_path))

    file_path = writer.save_report(
        query="test query",
        report="report body",
        summary="summary body",
        sources=["source1"],
        metadata={"key": "value"},
    )

    assert file_path.exists()
    assert file_path.suffix == ".txt"
    content = file_path.read_text(encoding="utf-8")
    assert "test query" in content
    assert "report body" in content
    assert "summary body" in content


def test_report_writer_saves_markdown_file(tmp_path):
    writer = ReportWriter(output_dir=str(tmp_path))

    file_path = writer.save_report_markdown(
        query="test query",
        report="report body",
        summary="summary body",
        sources=["source1"],
        metadata={"key": "value"},
    )

    assert file_path.exists()
    assert file_path.suffix == ".md"
    content = file_path.read_text(encoding="utf-8")
    assert "# Company Data Report" in content
    assert "test query" in content
    assert "report body" in content
