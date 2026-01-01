"""
Text splitter configuration for document chunking.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import get_settings


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Get configured text splitter.

    Returns:
        RecursiveCharacterTextSplitter instance
    """
    settings = get_settings()

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
