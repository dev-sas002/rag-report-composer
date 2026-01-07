"""
Tests for the backend registry.

The registry is the extension seam: a future developer adds Anthropic, Ollama
or pgvector by registering a factory, not by editing an `if settings.use_openai`
branch. What has to hold is that a name selects exactly one factory, that a
wrong name fails immediately and says what was available, and that registering
a provider does not import it.
"""

import sys

import pytest

from src.providers import (
    ProviderNotRegisteredError,
    create_embeddings,
    create_llm,
    create_vector_store_backend,
    embedding_registry,
    llm_registry,
    register_builtin_providers,
    register_embeddings,
    register_llm,
    resolve_name,
    vector_store_registry,
)


class Stub:
    """A settings-shaped object with only the fields the registry reads."""

    def __init__(self, **kwargs):
        self.use_openai = False
        self.llm_provider = "auto"
        self.embedding_provider = "auto"
        self.vector_backend = "chroma"
        self.__dict__.update(kwargs)


@pytest.fixture(autouse=True)
def restore_registries():
    """A test that registers a provider must not leak it into the next one."""
    yield
    for registry in (llm_registry, embedding_registry, vector_store_registry):
        for name in list(registry.names()):
            registry.unregister(name)
    register_builtin_providers()


class TestAutoResolution:
    def test_auto_follows_the_key(self):
        assert resolve_name("auto", Stub(use_openai=True)) == "openai"
        assert resolve_name("auto", Stub(use_openai=False)) == "local"

    def test_an_explicit_name_wins_over_the_key(self):
        assert resolve_name("hashing", Stub(use_openai=True)) == "hashing"

    def test_names_are_matched_case_and_space_insensitively(self):
        assert resolve_name("  OpenAI ", Stub()) == "openai"

    def test_an_empty_name_falls_back_to_auto(self):
        assert resolve_name("", Stub(use_openai=True)) == "openai"


class TestRegistration:
    def test_a_registered_factory_is_selected_by_name(self):
        register_llm("fake", lambda settings: "the-fake-model")

        assert create_llm(Stub(llm_provider="fake")) == "the-fake-model"

    def test_the_factory_receives_the_settings(self):
        seen = {}
        register_embeddings("spy", lambda settings: seen.setdefault("settings", settings))
        settings = Stub(embedding_provider="spy")

        create_embeddings(settings)

        assert seen["settings"] is settings

    def test_registering_over_an_existing_name_is_refused_by_default(self):
        """
        Silently replacing a provider would make "which backend am I running?"
        depend on module import order, which is not a question anyone should
        have to debug.
        """
        register_llm("dup", lambda settings: 1)

        with pytest.raises(ValueError, match="already registered"):
            register_llm("dup", lambda settings: 2)

    def test_replacement_is_available_when_asked_for_explicitly(self):
        register_llm("dup", lambda settings: 1)
        register_llm("dup", lambda settings: 2, replace=True)

        assert create_llm(Stub(llm_provider="dup")) == 2

    def test_auto_cannot_be_registered_as_a_backend_name(self):
        with pytest.raises(ValueError, match="reserved"):
            register_llm("auto", lambda settings: 1)

    def test_an_empty_name_is_refused(self):
        with pytest.raises(ValueError, match="must not be empty"):
            register_llm("   ", lambda settings: 1)


class TestUnknownProviders:
    def test_an_unknown_name_names_itself_and_the_alternatives(self):
        with pytest.raises(ProviderNotRegisteredError) as caught:
            create_llm(Stub(llm_provider="anthropic"))

        message = str(caught.value)
        assert "anthropic" in message
        assert "openai" in message and "local" in message

    def test_the_error_carries_the_kind_so_a_caller_can_act_on_it(self):
        with pytest.raises(ProviderNotRegisteredError) as caught:
            create_embeddings(Stub(embedding_provider="word2vec"))

        assert caught.value.kind == "embedding"
        assert caught.value.name == "word2vec"


class TestBuiltins:
    def test_the_shipped_backends_are_all_registered(self):
        assert llm_registry.names() == ["local", "openai"]
        assert embedding_registry.names() == ["hashing", "local", "openai"]
        assert vector_store_registry.names() == ["chroma", "memory"]

    def test_registering_the_builtins_does_not_import_torch(self):
        """
        Factories import lazily on purpose. If registration imported every
        backend, a process running entirely on OpenAI would pay the import cost
        of sentence-transformers — and the offline path could not be tested on
        a machine without torch at all.
        """
        already_loaded = "torch" in sys.modules

        register_builtin_providers()

        assert ("torch" in sys.modules) is already_loaded

    def test_the_memory_backend_is_built_with_the_embeddings_it_is_given(self):
        from src.retrieval.hashing_embeddings import HashingEmbeddings
        from src.retrieval.memory_store import InMemoryVectorStore

        embeddings = HashingEmbeddings(dimensions=8)

        store = create_vector_store_backend(
            Stub(vector_backend="memory"), embeddings=embeddings, client=None
        )

        assert isinstance(store, InMemoryVectorStore)
        assert store.embeddings is embeddings
