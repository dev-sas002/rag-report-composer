"""Lightweight tests for the setup verification script."""

from unittest.mock import Mock, patch

from scripts import verify_setup


def test_check_environment_reports_local_mode_without_a_key(capsys):
    """
    A missing key is a supported mode, not a failure.

    This previously asserted the check returned False, which stopped being
    correct once the app gained offline providers: refusing to start is exactly
    the behaviour that made the project impossible to evaluate.
    """
    with patch("src.config.get_settings") as mock_get_settings:
        settings = Mock()
        settings.openai_api_key = "your_openai_api_key_here"
        settings.openai_model = "gpt-4-turbo-preview"
        settings.openai_embedding_model = "text-embedding-3-large"
        settings.local_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        settings.environment = "test"
        mock_get_settings.return_value = settings

        ok = verify_setup.check_environment()

    out = capsys.readouterr().out
    assert "Mode: local" in out
    assert ok is True


def test_check_environment_reports_openai_mode_with_a_key(capsys):
    with patch("src.config.get_settings") as mock_get_settings:
        settings = Mock()
        settings.openai_api_key = "sk-real-looking-key"
        settings.openai_model = "gpt-4-turbo-preview"
        settings.openai_embedding_model = "text-embedding-3-large"
        settings.environment = "test"
        mock_get_settings.return_value = settings

        ok = verify_setup.check_environment()

    out = capsys.readouterr().out
    assert "Mode: OpenAI" in out
    assert ok is True


def test_check_dependencies_handles_missing_package(capsys, monkeypatch):
    """check_dependencies should surface missing packages cleanly."""

    def fake_import(name):
        if name == "langchain":
            raise ImportError()
        return None

    monkeypatch.setattr("builtins.__import__", fake_import)

    ok = verify_setup.check_dependencies()

    out = capsys.readouterr().out
    assert "Missing packages" in out
    assert ok is False
