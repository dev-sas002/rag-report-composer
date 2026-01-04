"""
Report writer for saving generated reports to files.
"""

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .config import get_settings
from .observability.logger import get_logger

logger = get_logger(__name__)


def _build_safe_filename(prefix: str, query: str, extension: str) -> str:
    """Build a filesystem-safe filename for a report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else "_" for c in query[:50])
    return f"{prefix}_{timestamp}_{safe_query}.{extension}"


def _build_text_report_content(
    query: str,
    report: str,
    summary: Optional[str] = None,
    sources: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Construct the plain-text report content."""
    content_parts = [
        "=" * 80,
        "COMPANY DATA REPORT",
        "=" * 80,
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nQuery: {query}",
        "\n" + "=" * 80,
    ]

    if summary:
        content_parts.extend(
            [
                "\n\nEXECUTIVE SUMMARY",
                "-" * 80,
                summary,
            ]
        )

    content_parts.extend(
        [
            "\n\nDETAILED REPORT",
            "-" * 80,
            report,
        ]
    )

    if sources:
        content_parts.extend(
            [
                "\n\nSOURCES",
                "-" * 80,
            ]
        )
        for i, source in enumerate(sources, 1):
            content_parts.append(f"{i}. {source}")

    if metadata:
        content_parts.extend(
            [
                "\n\nMETADATA",
                "-" * 80,
            ]
        )
        for key, value in metadata.items():
            content_parts.append(f"{key}: {value}")

    content_parts.append("\n" + "=" * 80)
    return "\n".join(content_parts)


def _build_markdown_report_content(
    query: str,
    report: str,
    summary: Optional[str] = None,
    sources: Optional[Iterable[str]] = None,
    metadata: Optional[dict] = None,
) -> str:
    """Construct the markdown report content."""
    content_parts = [
        "# Company Data Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Query:** {query}",
        "\n---",
    ]

    if summary:
        content_parts.extend(
            [
                "\n## Executive Summary\n",
                summary,
            ]
        )

    content_parts.extend(
        [
            "\n## Detailed Report\n",
            report,
        ]
    )

    if sources:
        content_parts.append("\n## Sources\n")
        for i, source in enumerate(sources, 1):
            content_parts.append(f"{i}. {source}")

    if metadata:
        content_parts.append("\n## Metadata\n")
        for key, value in metadata.items():
            content_parts.append(f"- **{key}:** {value}")

    return "\n".join(content_parts)


class ReportWriter:
    """Handles writing reports to files."""

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the report writer.

        Args:
            output_dir: Output directory for reports (uses settings default if not provided)
        """
        settings = get_settings()
        self.output_dir = Path(output_dir or settings.report_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_report(
        self,
        query: str,
        report: str,
        summary: Optional[str] = None,
        sources: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Path:
        """
        Save report to a file.

        Args:
            query: Original query
            report: Generated report content
            summary: Executive summary (optional)
            sources: List of source documents (optional)
            metadata: Additional metadata (optional)

        Returns:
            Path to the saved report file
        """
        filename = _build_safe_filename("report", query, "txt")
        file_path = self.output_dir / filename

        content = _build_text_report_content(
            query=query,
            report=report,
            summary=summary,
            sources=sources,
            metadata=metadata,
        )

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(
                "report_saved", file_path=str(file_path), size_bytes=file_path.stat().st_size
            )

            return file_path

        except Exception as e:
            logger.error("report_save_failed", file_path=str(file_path), error=str(e))
            raise

    def save_report_markdown(
        self,
        query: str,
        report: str,
        summary: Optional[str] = None,
        sources: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Path:
        """
        Save report as markdown file.

        Args:
            query: Original query
            report: Generated report content
            summary: Executive summary (optional)
            sources: List of source documents (optional)
            metadata: Additional metadata (optional)

        Returns:
            Path to the saved report file
        """
        filename = _build_safe_filename("report", query, "md")
        file_path = self.output_dir / filename

        content = _build_markdown_report_content(
            query=query,
            report=report,
            summary=summary,
            sources=sources,
            metadata=metadata,
        )

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(
                "report_saved_markdown",
                file_path=str(file_path),
                size_bytes=file_path.stat().st_size,
            )

            return file_path

        except Exception as e:
            logger.error("report_save_markdown_failed", file_path=str(file_path), error=str(e))
            raise
