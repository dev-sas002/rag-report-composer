"""
The seam: a registry of interchangeable backends.

Three things in this pipeline are genuinely replaceable — the chat model that
writes the report, the embedder that turns text into vectors, and the store
those vectors live in. Everything else is arithmetic over their outputs.

Before this module, replacing any of them meant editing an `if
settings.use_openai:` branch buried in the module that used it, and the three
branches lived in three different files. A new backend — Anthropic, Ollama, a
local llama.cpp server, pgvector, Qdrant — meant touching every one.

Now a backend is a name and a factory:

    from src.providers import register_llm

    def make_ollama(settings):
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model="llama3")

    register_llm("ollama", make_ollama)

and `LLM_PROVIDER=ollama` selects it. The factories are lazy — a registration
must not import its backend at module scope, or registering a provider would
cost the import time of every provider.

This module holds no knowledge of any particular backend; `builtin.py` is where
the ones that ship with the project are registered.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Protocol

#: A factory takes the settings object and returns a ready backend.
Factory = Callable[[Any], Any]

#: Provider name meaning "decide from the rest of the configuration".
AUTO = "auto"


class ProviderNotRegisteredError(KeyError):
    """Raised when configuration names a backend nothing registered."""

    def __init__(self, kind: str, name: str, available: List[str]) -> None:
        super().__init__(
            f"no {kind} provider named {name!r}; registered: {', '.join(available) or 'none'}"
        )
        self.kind = kind
        self.name = name
        self.available = available


class SettingsLike(Protocol):  # pragma: no cover - structural type only
    """The slice of Settings a factory is allowed to depend on."""

    use_openai: bool


class _Registry:
    """A named collection of factories for one kind of backend."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: Dict[str, Factory] = {}

    def register(self, name: str, factory: Factory, *, replace: bool = False) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError(f"{self.kind} provider name must not be empty")
        if key == AUTO:
            raise ValueError(f"{AUTO!r} is reserved and cannot be registered")
        if key in self._factories and not replace:
            raise ValueError(
                f"{self.kind} provider {key!r} is already registered; "
                f"pass replace=True to override it"
            )
        self._factories[key] = factory

    def unregister(self, name: str) -> None:
        self._factories.pop(name.strip().lower(), None)

    def names(self) -> List[str]:
        return sorted(self._factories)

    def create(self, name: str, settings: Any, **kwargs: Any) -> Any:
        key = name.strip().lower()
        try:
            factory = self._factories[key]
        except KeyError:
            raise ProviderNotRegisteredError(self.kind, key, self.names()) from None
        return factory(settings, **kwargs)


llm_registry = _Registry("llm")
embedding_registry = _Registry("embedding")
vector_store_registry = _Registry("vector store")


def register_llm(name: str, factory: Factory, *, replace: bool = False) -> None:
    """Register a chat model backend under `name`."""
    llm_registry.register(name, factory, replace=replace)


def register_embeddings(name: str, factory: Factory, *, replace: bool = False) -> None:
    """Register an embedding backend under `name`."""
    embedding_registry.register(name, factory, replace=replace)


def register_vector_store(name: str, factory: Factory, *, replace: bool = False) -> None:
    """Register a vector store backend under `name`."""
    vector_store_registry.register(name, factory, replace=replace)


def resolve_name(configured: str, settings: Any) -> str:
    """
    Turn a possibly-`auto` provider name into a concrete one.

    `auto` follows `settings.use_openai`, which is itself derived from
    `PROVIDER_MODE` and the presence of a key. Keeping that in one function is
    what stops "which backend am I actually using?" from having three answers.
    """
    name = (configured or AUTO).strip().lower()
    if name != AUTO:
        return name
    return "openai" if getattr(settings, "use_openai", False) else "local"


def create_llm(settings: Any) -> Any:
    """Build the chat model the configuration asks for."""
    return llm_registry.create(
        resolve_name(getattr(settings, "llm_provider", AUTO), settings), settings
    )


def create_embeddings(settings: Any) -> Any:
    """Build the embedding backend the configuration asks for."""
    return embedding_registry.create(
        resolve_name(getattr(settings, "embedding_provider", AUTO), settings), settings
    )


def create_vector_store_backend(settings: Any, **kwargs: Any) -> Any:
    """
    Build the vector store backend the configuration asks for.

    Unlike the other two, this one takes keyword arguments: a store needs the
    embedding function and, for the persistent backends, a client.
    """
    name = getattr(settings, "vector_backend", "chroma") or "chroma"
    return vector_store_registry.create(name, settings, **kwargs)
