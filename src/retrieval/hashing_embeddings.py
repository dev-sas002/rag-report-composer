"""
Embeddings with no model behind them.

A hashed bag of words, L2-normalised. It has no semantics: "revenue rose" and
"sales increased" get orthogonal vectors, which is precisely the thing a real
embedder is for. So this is never a serving backend.

It exists because two useful activities were blocked on loading 90MB of
weights: measuring how vector-store query latency grows with corpus size, and
reproducing a ranking or plumbing bug in a test. Neither depends on the vectors
meaning anything — only on them being the right shape, deterministic, and
cheap.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

from langchain_core.embeddings import Embeddings

DEFAULT_DIMENSIONS = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbeddings(Embeddings):
    """Deterministic, dependency-free vectors for benchmarking and tests."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def _vector(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            # The sign bit spreads tokens that collide on the same index, so a
            # collision cancels out rather than compounding.
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)
