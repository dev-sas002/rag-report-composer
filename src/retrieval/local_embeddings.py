"""
Offline embeddings, used when no OpenAI key is configured.

Runs a small sentence-transformer locally instead of calling an API. This is
not a stub returning random vectors: it produces real semantic embeddings, so
retrieval genuinely works — a query about "revenue growth" finds a passage that
says "sales increased" without sharing a word with it.

`sentence-transformers` was already a dependency of this project, so the local
path costs nothing extra to ship. The model is ~90MB and is baked into the
Docker image at build time so the first request does not pay for a download.

The trade against `text-embedding-3-large` is real and worth stating: 384
dimensions against 3072, and noticeably weaker on long or specialised text. The
evaluation harness reports retrieval quality for whichever backend is active,
so the gap is measured rather than assumed.
"""

from __future__ import annotations

import threading
from typing import List

from langchain_core.embeddings import Embeddings

from ..observability.logger import get_logger

logger = get_logger(__name__)

DEFAULT_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Loading the model takes a few seconds and it is safe to share, so it is
# created once per process behind a lock rather than per VectorStore.
_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


def _load_model(name: str):
    with _model_lock:
        if name not in _model_cache:
            from sentence_transformers import SentenceTransformer

            logger.info("loading_local_embedding_model", model=name)
            _model_cache[name] = SentenceTransformer(name)
        return _model_cache[name]


class LocalEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by a local model."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        # Deferred so that constructing this class — which happens at import
        # time in some code paths — does not block on loading weights.
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        logger.debug("documents_embedded_locally", num_texts=len(texts))
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        vector = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return vector.tolist()
