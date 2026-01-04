"""
Embedding backend selection.

The choice of *which* embedder lives in `src/providers`; this module owns the
OpenAI implementation and the assembly — provider, then cache — that the rest
of the project consumes through `get_embeddings()`.
"""

from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ..config import get_settings
from ..observability.cost_tracker import get_cost_tracker
from ..observability.logger import get_logger
from ..providers import create_embeddings, resolve_name
from .embedding_cache import wrap_with_cache

logger = get_logger(__name__)


class TrackedOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAI embeddings with cost tracking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cost_tracker = get_cost_tracker()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents, recording the tokens they cost."""
        total_tokens = sum(self._cost_tracker.count_tokens(text) for text in texts)

        embeddings = super().embed_documents(texts)

        self._cost_tracker.track_call(
            model=self.model,
            input_tokens=total_tokens,
            output_tokens=0,
            operation="embed_documents",
            metadata={"num_texts": len(texts)},
        )

        logger.debug("documents_embedded", num_texts=len(texts), total_tokens=total_tokens)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query, recording the tokens it cost."""
        tokens = self._cost_tracker.count_tokens(text)

        embedding = super().embed_query(text)

        self._cost_tracker.track_call(
            model=self.model,
            input_tokens=tokens,
            output_tokens=0,
            operation="embed_query",
            metadata={"query_length": len(text)},
        )

        logger.debug("query_embedded", tokens=tokens)

        return embedding


def get_embeddings(settings: Optional[object] = None) -> Embeddings:
    """
    Return the embeddings backend for the current configuration.

    Two decisions, in order. Which backend: a registered provider named by
    `EMBEDDING_PROVIDER`, or — under `auto` — OpenAI when a key is configured
    and the local sentence-transformer when one is not. Whether to cache it:
    yes unless `ENABLE_EMBEDDING_CACHE` is false, because embedding is a pure
    function of (model, text) and re-running it is pure waste.

    Callers only ever see the LangChain `Embeddings` interface and never branch
    on which backend is behind it.
    """
    settings = settings or get_settings()
    backend_name = resolve_name(getattr(settings, "embedding_provider", "auto"), settings)

    logger.info(
        "embeddings_backend_selected",
        backend=backend_name,
        model=settings.embedding_model_name,
    )

    embeddings = create_embeddings(settings)

    if not getattr(settings, "enable_embedding_cache", False):
        return embeddings

    return wrap_with_cache(
        embeddings,
        settings.embedding_cache_path,
        model_name=settings.embedding_model_name,
    )
