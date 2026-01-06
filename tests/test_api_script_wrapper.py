"""Tests that wrap the scripts/test_api helpers with mocked requests."""

from unittest.mock import Mock

from scripts import test_api as api_script


def test_print_section_formats_title(capsys):
    api_script.print_section("Section Title")
    out = capsys.readouterr().out
    assert "Section Title" in out


def test_test_root_uses_requests_get(monkeypatch):
    fake_response = Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {"message": "ok", "version": "1.0.0"}

    monkeypatch.setattr("scripts.test_api.requests.get", lambda url: fake_response)

    ok = api_script.test_root()
    assert ok is True
