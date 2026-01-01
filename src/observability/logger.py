"""
Structured logging configuration using structlog.
Console output for development, JSON for production, and a rotating log file.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
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


#: Marks the handlers `setup_logging` installed, so calling it twice (the CLI
#: and the API both do, and tests import both) does not double every line.
_HANDLER_TAG = "_report_generator_handler"


def _shared_processors() -> list[Processor]:
    return [
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


def _install_handler(handler: logging.Handler, renderer: Processor, level: int) -> None:
    handler.setLevel(level)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer, foreign_pre_chain=_shared_processors()
        )
    )
    setattr(handler, _HANDLER_TAG, True)
    logging.root.addHandler(handler)


def setup_logging() -> None:
    """
    Configure structured logging for the application.

    Rendering happens per handler rather than once for all of them. The console
    gets colour when it is a terminal; the log file never does. Rendering once
    with colours on wrote ANSI escape codes into `logs/app.log`, which made it
    unreadable in anything but a terminal and ungreppable even there.
    """
    settings = get_settings()
    log_level = _get_log_level(settings.log_level)
    production = settings.environment == "production"

    for handler in [h for h in logging.root.handlers if getattr(h, _HANDLER_TAG, False)]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.root.setLevel(log_level)

    console_renderer: Processor = (
        structlog.processors.JSONRenderer()
        if production
        else structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    )
    _install_handler(logging.StreamHandler(sys.stdout), console_renderer, log_level)

    # The file handler rotates on LOG_ROTATION_SIZE_MB / LOG_BACKUP_COUNT. A
    # plain FileHandler ignored both settings and grew without bound.
    log_file = Path(settings.log_dir) / settings.log_file
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=settings.log_rotation_size_mb * 1024 * 1024,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
    except OSError as exc:  # pragma: no cover - read-only filesystem
        logging.getLogger(__name__).warning("file_logging_disabled: %s (%s)", log_file, exc)
    else:
        file_renderer: Processor = (
            structlog.processors.JSONRenderer()
            if production
            else structlog.dev.ConsoleRenderer(colors=False)
        )
        _install_handler(file_handler, file_renderer, log_level)

    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
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
