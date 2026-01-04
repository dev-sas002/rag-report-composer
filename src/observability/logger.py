"""
Structured logging configuration using structlog.
Provides consistent, JSON-formatted logs for production environments.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from ..config import get_settings


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log entries."""
    settings = get_settings()
    event_dict["app_name"] = settings.app_name
    event_dict["app_version"] = settings.app_version
    event_dict["environment"] = settings.environment
    return event_dict


def _get_log_level(level_name: str) -> int:
    """
    Resolve a logging level name to its numeric value.

    Falls back to INFO if an invalid level is provided to keep the
    application running in misconfigured environments.
    """
    try:
        return getattr(logging, level_name.upper())
    except AttributeError:
        return logging.INFO


def _has_file_handler_for(log_file: Path) -> bool:
    """Check whether a file handler for the given log file is already registered."""
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, "baseFilename", None
        ) == str(log_file):
            return True
    return False


def _add_file_handler(log_file: Path, level: int) -> None:
    """
    Attach a file handler for the given log file.

    Any exceptions are allowed to bubble up to the caller so they can
    decide how to degrade gracefully.
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    logging.root.addHandler(file_handler)


def setup_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()

    # Configure standard library logging for stdout first so that we always
    # have at least one working sink, even if file-based logging fails.
    log_level = _get_log_level(settings.log_level)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Configure file-based logging, but degrade gracefully if the directory
    # or file cannot be created (e.g. read-only filesystem).
    log_dir = Path(settings.log_dir)
    log_file = log_dir / settings.log_file

    try:
        log_dir.mkdir(parents=True, exist_ok=True)

        if not _has_file_handler_for(log_file):
            _add_file_handler(log_file, log_level)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logging.getLogger(__name__).warning(
            "file_logging_disabled",
            extra={
                "log_file": str(log_file),
                "error": str(exc),
            },
        )

    # Configure structlog processors
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_app_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add JSON renderer for production, ConsoleRenderer for development
    if settings.environment == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _silence_third_party_noise()


# Loggers whose output is never about this application. Chroma's product
# telemetry is disabled by `vector_store.chroma_client_settings`, but its
# posthog client tries to send anyway and logs the resulting SDK version
# mismatch at ERROR — four lines of `Failed to send telemetry event` ahead of
# any output the user asked for, about a feature that is switched off.
THIRD_PARTY_NOISE = ("chromadb.telemetry.product.posthog",)


def _silence_third_party_noise() -> None:
    for name in THIRD_PARTY_NOISE:
        logging.getLogger(name).setLevel(logging.CRITICAL)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)
