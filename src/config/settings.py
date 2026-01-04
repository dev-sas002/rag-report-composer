"""
Application settings and configuration management.
Uses Pydantic for validation and environment variable loading.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    #
    # Optional on purpose. This used to be required, so importing settings —
    # which every module does — raised before the process could start, and the
    # project could not be run, demoed or tested without a billable key.
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key. Absent -> offline providers."
    )
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI Model")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large", description="OpenAI Embedding Model"
    )
    openai_request_timeout: float = Field(
        default=60.0,
        gt=0,
        description="Per-request timeout in seconds for OpenAI chat and embedding calls",
    )
    openai_max_retries: int = Field(
        default=3, ge=0, description="Retries for transient OpenAI API failures"
    )

    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", description="ChromaDB persistence directory"
    )
    chroma_collection_name: str = Field(
        default="company_data", description="ChromaDB collection name"
    )

    # Application Configuration
    app_name: str = Field(default="RAG Company Report Generator", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Environment")

    # Vector Store Configuration
    chunk_size: int = Field(default=1000, gt=0, description="Text chunk size for splitting")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")
    top_k_results: int = Field(default=5, gt=0, description="Number of top results to retrieve")
    max_context_chars: int = Field(
        default=12000,
        gt=0,
        description="Upper bound on the retrieved context assembled into a prompt",
    )
    context_relevance_margin: float = Field(
        default=0.35,
        ge=0.0,
        description=(
            "Drop retrieved chunks whose distance exceeds the best hit by more "
            "than this fraction. 0 keeps everything retrieval returned."
        ),
    )
    vector_backend: str = Field(
        default="chroma", description="Registered vector store backend: chroma | memory"
    )
    hashing_dimensions: int = Field(
        default=256,
        gt=0,
        description="Width of the model-free hashing embedder used for benchmarks and tests",
    )

    # Embedding cache
    enable_embedding_cache: bool = Field(
        default=True,
        description="Serve repeated (model, text) embeddings from disk instead of re-embedding",
    )
    embedding_cache_path: str = Field(
        default="./data/embedding_cache.sqlite", description="SQLite file backing the cache"
    )

    # Report Configuration
    max_report_length: int = Field(default=2000, description="Maximum report length in tokens")
    report_output_dir: str = Field(default="./reports", description="Report output directory")

    # Provider selection. "auto" follows the presence of an API key; "openai"
    # and "local" force a choice and are what CI and the demo use.
    provider_mode: str = Field(default="auto", description="auto | openai | local")
    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model used when running offline",
    )
    # Named backends from src/providers. "auto" follows `use_openai`; anything
    # else selects a registered provider by name, including one a downstream
    # project registered itself.
    llm_provider: str = Field(default="auto", description="auto | openai | local | <registered>")
    embedding_provider: str = Field(
        default="auto", description="auto | openai | local | hashing | <registered>"
    )

    @model_validator(mode="after")
    def _validate_configuration(self) -> "Settings":
        """
        Reject configurations that would only fail much later, and obscurely.

        A chunk overlap that is not smaller than the chunk size makes
        RecursiveCharacterTextSplitter raise at ingestion time, with a message
        that does not name the setting responsible. An unrecognised
        PROVIDER_MODE used to fall through to the key check silently, so a
        typo in `PROVIDER_MODE=lokal` quietly ran against OpenAI.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        if self.provider_mode not in {"auto", "openai", "local"}:
            raise ValueError(
                "PROVIDER_MODE must be one of auto, openai, local " f"(got {self.provider_mode!r})"
            )
        return self

    @property
    def resolved_embedding_provider(self) -> str:
        """`embedding_provider` with `auto` already decided."""
        name = (self.embedding_provider or "auto").strip().lower()
        if name != "auto":
            return name
        return "openai" if self.use_openai else "local"

    @property
    def resolved_llm_provider(self) -> str:
        """`llm_provider` with `auto` already decided."""
        name = (self.llm_provider or "auto").strip().lower()
        if name != "auto":
            return name
        return "openai" if self.use_openai else "local"

    @property
    def embedding_model_name(self) -> str:
        """
        The embedding model actually in use, whichever backend is active.

        This is the embedding cache's namespace, so it has to distinguish every
        backend that produces different vectors. Deriving it from `use_openai`
        alone would give the hashing embedder the sentence-transformer's name
        and let one serve the other's vectors — a failure that does not raise,
        it just retrieves the wrong chunks.
        """
        provider = self.resolved_embedding_provider
        if provider == "openai":
            return self.openai_embedding_model
        if provider == "hashing":
            return f"hashing-{self.hashing_dimensions}"
        return self.local_embedding_model

    @property
    def use_openai(self) -> bool:
        """True when live OpenAI calls should be made."""
        if self.provider_mode == "local":
            return False
        if self.provider_mode == "openai":
            return True
        return bool(self.openai_api_key)

    # Cost Tracking
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    cost_log_file: str = Field(
        default="./logs/cost_tracking.json", description="Cost tracking log file"
    )

    # Logging Configuration
    log_dir: str = Field(default="./logs", description="Log directory")
    log_file: str = Field(default="app.log", description="Log file name")
    log_rotation_size_mb: int = Field(default=10, description="Log rotation size in MB")
    log_backup_count: int = Field(default=5, description="Number of log backups to keep")

    def get_absolute_path(self, path: str) -> Path:
        """Convert relative path to absolute path."""
        path_obj = Path(path)
        if not path_obj.is_absolute():
            return Path.cwd() / path_obj
        return path_obj

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.chroma_persist_directory,
            self.report_output_dir,
            self.log_dir,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
