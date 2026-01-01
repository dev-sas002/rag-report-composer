"""
A content-addressed cache in front of any embedding backend.

Embeddings are a pure function of (model, text). The pipeline nonetheless
recomputed them constantly:

* **Re-ingestion re-embeds everything.** Adding one document to a corpus, or
  re-running ingestion after changing an unrelated setting, paid to embed every
  chunk again. On OpenAI that is real money; on the local model it is minutes
  of CPU. Overlapping chunks make it worse — with `CHUNK_OVERLAP=200`, a fifth
  of the corpus is embedded twice even on a first run of two adjacent files
  that share boilerplate.
* **Repeated queries re-embed the query.** An evaluation run, a dashboard
  polling the same question, a user pressing retry: each one paid again.

Keying on a hash of the model name and the exact text means the cache can never
serve a vector produced by a different model — the failure that would otherwise
be invisible, because a wrong-but-plausible vector does not raise, it just
retrieves the wrong chunks.

SQLite rather than a pickle or a directory of files: it is in the standard
library, it survives a crash mid-write, and it handles two processes ingesting
at once, which a naive file cache does not.
"""

from __future__ import annotations

import array
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.embeddings import Embeddings

from ..observability.logger import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key    TEXT PRIMARY KEY,
    vector BLOB NOT NULL
)
"""


def cache_key(model: str, text: str) -> str:
    """A stable key for one (model, text) pair."""
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


class EmbeddingStore:
    """Persistent key-to-vector storage. Float32, so a 3072-dim vector is 12KB."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.execute(_SCHEMA)
        self._connection.commit()

    def get_many(self, keys: Sequence[str]) -> Dict[str, List[float]]:
        if not keys:
            return {}
        found: Dict[str, List[float]] = {}
        with self._lock:
            # Chunked so a large ingestion batch does not exceed SQLite's
            # limit on the number of bound parameters in one statement.
            for start in range(0, len(keys), 500):
                window = keys[start : start + 500]
                placeholders = ",".join("?" * len(window))
                rows = self._connection.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                    window,
                ).fetchall()
                for key, blob in rows:
                    values = array.array("f")
                    values.frombytes(blob)
                    found[key] = list(values)
        return found

    def put_many(self, items: Dict[str, Sequence[float]]) -> None:
        if not items:
            return
        rows = [(key, array.array("f", vector).tobytes()) for key, vector in items.items()]
        with self._lock:
            self._connection.executemany(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)", rows
            )
            self._connection.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class CachingEmbeddings(Embeddings):
    """
    Wraps an embeddings backend and serves repeats from storage.

    Transparent: it is an `Embeddings`, so the vector store, the ingestion
    pipeline and LangChain cannot tell the difference. `hits` and `misses` are
    exposed because a cache whose hit rate nobody can see is a cache nobody can
    justify.
    """

    def __init__(
        self,
        inner: Embeddings,
        store: EmbeddingStore,
        model_name: Optional[str] = None,
    ) -> None:
        self.inner = inner
        self.store = store
        self.model_name = model_name or _describe(inner)
        self.hits = 0
        self.misses = 0

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        keys = [cache_key(self.model_name, text) for text in texts]
        cached = self.store.get_many(keys)

        # Deduplicate within the batch too: two identical chunks in one call
        # were previously embedded twice before either reached the cache.
        missing_order: List[str] = []
        missing_texts: List[str] = []
        for key, text in zip(keys, texts):
            if key not in cached and key not in missing_order:
                missing_order.append(key)
                missing_texts.append(text)

        if missing_texts:
            fresh = self.inner.embed_documents(missing_texts)
            new = dict(zip(missing_order, fresh))
            self.store.put_many(new)
            cached.update(new)

        self.hits += len(texts) - len(missing_texts)
        self.misses += len(missing_texts)

        logger.debug(
            "embedding_cache_batch",
            texts=len(texts),
            hits=len(texts) - len(missing_texts),
            misses=len(missing_texts),
        )
        return [cached[key] for key in keys]

    def embed_query(self, text: str) -> List[float]:
        key = cache_key(self.model_name, text)
        cached = self.store.get_many([key])
        if key in cached:
            self.hits += 1
            return cached[key]

        self.misses += 1
        vector = self.inner.embed_query(text)
        self.store.put_many({key: vector})
        return vector

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _describe(embeddings: Any) -> str:
    """
    Name the backend well enough that two different ones cannot share a key.

    Falls back to the class name, which is still distinct per backend — the
    risk this guards against is an OpenAI vector being served to a local model,
    not two builds of the same model disagreeing.
    """
    for attribute in ("model", "model_name"):
        value = getattr(embeddings, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(embeddings).__name__


def wrap_with_cache(
    embeddings: Embeddings, path: str, model_name: Optional[str] = None
) -> CachingEmbeddings:
    """Put a cache at `path` in front of `embeddings`."""
    return CachingEmbeddings(embeddings, EmbeddingStore(Path(path)), model_name)
