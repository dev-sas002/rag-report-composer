"""Tests for the logging setup: rotation settings and a colour-free log file."""

import logging
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace

import pytest
import structlog

from src.observability import logger as logger_module


@pytest.fixture
def isolated_logging(tmp_path, monkeypatch):
    """Run setup_logging against a temp directory, then undo everything it did."""
    settings = SimpleNamespace(
        app_name="test-app",
        app_version="0.0.0",
        environment="development",
        log_level="INFO",
        log_dir=str(tmp_path / "logs"),
        log_file="app.log",
        log_rotation_size_mb=2,
        log_backup_count=3,
    )
    monkeypatch.setattr(logger_module, "get_settings", lambda: settings)
    saved_config = structlog.get_config()
    saved_level = logging.root.level
    yield settings, tmp_path / "logs" / "app.log"
    for handler in [
        h for h in logging.root.handlers if getattr(h, logger_module._HANDLER_TAG, False)
    ]:
        logging.root.removeHandler(handler)
        handler.close()
    logging.root.setLevel(saved_level)
    structlog.configure(**saved_config)


def _installed_handlers():
    return [h for h in logging.root.handlers if getattr(h, logger_module._HANDLER_TAG, False)]


def test_the_log_file_rotates_on_the_configured_size_and_count(isolated_logging):
    _settings, log_file = isolated_logging

    logger_module.setup_logging()

    rotating = [h for h in _installed_handlers() if isinstance(h, RotatingFileHandler)]
    assert len(rotating) == 1
    assert rotating[0].maxBytes == 2 * 1024 * 1024
    assert rotating[0].backupCount == 3
    assert rotating[0].baseFilename == str(log_file)


def test_calling_setup_twice_does_not_duplicate_handlers(isolated_logging):
    logger_module.setup_logging()
    logger_module.setup_logging()

    assert len(_installed_handlers()) == 2  # one console, one file


def test_the_log_file_never_contains_terminal_colour_codes(isolated_logging):
    _settings, log_file = isolated_logging

    logger_module.setup_logging()
    structlog.get_logger("test.logger").warning("disk_nearly_full", free_mb=12)
    for handler in _installed_handlers():
        handler.flush()

    written = log_file.read_text(encoding="utf-8")
    assert "disk_nearly_full" in written
    assert "free_mb=12" in written
    assert "\x1b[" not in written


def test_production_writes_json_lines(isolated_logging):
    settings, log_file = isolated_logging
    settings.environment = "production"

    logger_module.setup_logging()
    structlog.get_logger("test.logger").info("report_generated", tokens=7)
    for handler in _installed_handlers():
        handler.flush()

    line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    assert '"event": "report_generated"' in line
    assert '"tokens": 7' in line
