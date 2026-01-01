"""Interchangeable backends for the chat model, the embedder and the vector store."""

from .builtin import register_builtin_providers
from .registry import (
    AUTO,
    ProviderNotRegisteredError,
    create_embeddings,
    create_llm,
    create_vector_store_backend,
    embedding_registry,
    llm_registry,
    register_embeddings,
    register_llm,
    register_vector_store,
    resolve_name,
    vector_store_registry,
)

register_builtin_providers()

__all__ = [
    "AUTO",
    "ProviderNotRegisteredError",
    "create_embeddings",
    "create_llm",
    "create_vector_store_backend",
    "embedding_registry",
    "llm_registry",
    "register_builtin_providers",
    "register_embeddings",
    "register_llm",
    "register_vector_store",
    "resolve_name",
    "vector_store_registry",
]
