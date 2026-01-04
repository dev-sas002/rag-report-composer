"""
The backends that ship with the project.

Every factory imports its dependency inside the function body. That is not
style: `sentence_transformers` pulls in torch, and importing it to register a
provider nobody selected would add seconds to the start-up of a process running
entirely on OpenAI — and would make the offline path impossible to test on a
machine without torch installed.
"""

from __future__ import annotations

from typing import Any

from ..observability.logger import get_logger
from .registry import register_embeddings, register_llm, register_vector_store

logger = get_logger(__name__)


# -- chat models ---------------------------------------------------------


def openai_llm(settings: Any) -> Any:
    """OpenAI chat completion, with a timeout and a retry budget."""
    from langchain_openai import ChatOpenAI

    # A chat call with no timeout can hang for as long as the socket stays
    # open, which in a request handler means the request hangs with it. Both
    # values are configurable rather than baked in.
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.7,
        timeout=settings.openai_request_timeout,
        max_retries=settings.openai_max_retries,
    )


def local_llm(settings: Any) -> Any:
    """The offline extractive writer. Quotes the sources; never invents."""
    from ..agent.local_llm import ExtractiveReportWriter

    logger.warning(
        "using_offline_writer",
        reason="no OpenAI backend selected",
        note="reports will be extractive, not generated",
    )
    return ExtractiveReportWriter()


# -- embeddings ----------------------------------------------------------


def openai_embeddings(settings: Any) -> Any:
    """OpenAI embeddings, with per-call cost tracking."""
    from ..retrieval.embeddings import TrackedOpenAIEmbeddings

    # Ingestion issues one embedding call per batch, so a single stalled
    # request stalls the whole run.
    return TrackedOpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
        timeout=settings.openai_request_timeout,
        max_retries=settings.openai_max_retries,
    )


def local_embeddings(settings: Any) -> Any:
    """A sentence-transformer running in-process. Real vectors, no API."""
    from ..retrieval.local_embeddings import LocalEmbeddings

    return LocalEmbeddings(settings.local_embedding_model)


def hashing_embeddings(settings: Any) -> Any:
    """
    Deterministic hashed bag-of-words vectors. No model, no download.

    Retrieval quality is poor — it matches on shared words and nothing else —
    so this is not a backend to serve from. It exists because benchmarking
    vector-store latency, and reproducing a ranking bug, should not require
    loading 90MB of weights or waiting for a download.
    """
    from ..retrieval.hashing_embeddings import HashingEmbeddings

    return HashingEmbeddings(dimensions=getattr(settings, "hashing_dimensions", 256))


# -- vector stores -------------------------------------------------------


def chroma_backend(settings: Any, *, embeddings: Any, client: Any = None) -> Any:
    """Persistent ChromaDB, the default."""
    from langchain_community.vectorstores import Chroma

    return Chroma(
        client=client,
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
    )


def memory_backend(settings: Any, *, embeddings: Any, client: Any = None) -> Any:
    """
    An in-process store with no persistence.

    Useful for exactly two things, both of which the project does: measuring
    query latency against a synthetic corpus without leaving a multi-gigabyte
    index on disk, and running the evaluation harness end to end in a test.
    """
    from ..retrieval.memory_store import InMemoryVectorStore

    return InMemoryVectorStore(embeddings)


def register_builtin_providers() -> None:
    """
    Register everything that ships with the project.

    Idempotent, because it runs at import of `src.providers` and a module can
    be imported more than once under a different name during testing.
    """
    for name, factory in (("openai", openai_llm), ("local", local_llm)):
        register_llm(name, factory, replace=True)

    for name, factory in (
        ("openai", openai_embeddings),
        ("local", local_embeddings),
        ("hashing", hashing_embeddings),
    ):
        register_embeddings(name, factory, replace=True)

    for name, factory in (("chroma", chroma_backend), ("memory", memory_backend)):
        register_vector_store(name, factory, replace=True)
