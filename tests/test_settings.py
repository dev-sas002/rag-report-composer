"""
Tests for configuration.

Settings decide which provider runs, how documents are chunked and how much
context a prompt is allowed to carry, so a bad value here is not a typo — it
is a silent change of behaviour or a live API call nobody asked for.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import Settings

# The example file sits at the repository root, wherever pytest was invoked from.
ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


def settings(**overrides) -> Settings:
    # _env_file=None so a developer's real .env cannot influence the result.
    return Settings(_env_file=None, **overrides)


class TestProviderSelection:
    def test_a_key_selects_openai_in_auto_mode(self):
        assert settings(openai_api_key="sk-test", provider_mode="auto").use_openai is True

    def test_no_key_selects_the_local_providers_in_auto_mode(self):
        assert settings(openai_api_key=None, provider_mode="auto").use_openai is False

    def test_local_mode_overrides_a_present_key(self):
        """
        The point of the override: a developer with a key exported in their
        shell must still be able to run the offline path deliberately.
        """
        assert settings(openai_api_key="sk-test", provider_mode="local").use_openai is False

    def test_openai_mode_is_honoured_without_a_key_so_the_failure_is_explicit(self):
        assert settings(openai_api_key=None, provider_mode="openai").use_openai is True

    def test_an_unrecognised_provider_mode_is_rejected_rather_than_ignored(self):
        with pytest.raises(ValidationError, match="PROVIDER_MODE"):
            settings(provider_mode="lokal")


class TestChunkingValidation:
    def test_an_overlap_smaller_than_the_chunk_is_accepted(self):
        assert settings(chunk_size=500, chunk_overlap=100).chunk_overlap == 100

    def test_an_overlap_equal_to_the_chunk_size_is_rejected(self):
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            settings(chunk_size=500, chunk_overlap=500)

    def test_an_overlap_larger_than_the_chunk_size_is_rejected(self):
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            settings(chunk_size=500, chunk_overlap=900)

    def test_a_zero_chunk_size_is_rejected(self):
        with pytest.raises(ValidationError):
            settings(chunk_size=0)

    def test_top_k_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            settings(top_k_results=0)


class TestApiCallLimits:
    def test_timeout_and_retries_have_defaults_so_calls_cannot_hang_forever(self):
        s = settings()
        assert s.openai_request_timeout > 0
        assert s.openai_max_retries >= 1

    def test_a_non_positive_timeout_is_rejected(self):
        with pytest.raises(ValidationError):
            settings(openai_request_timeout=0)

    def test_context_assembly_is_bounded_by_default(self):
        assert settings().max_context_chars > 0


class TestPaths:
    def test_a_relative_path_is_resolved_against_the_working_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        resolved = settings().get_absolute_path("./reports")

        assert resolved.is_absolute()
        assert resolved == tmp_path / "reports"

    def test_an_absolute_path_is_left_alone(self, tmp_path):
        assert settings().get_absolute_path(str(tmp_path)) == tmp_path

    def test_ensure_directories_creates_everything_the_app_writes_to(self, tmp_path):
        s = settings(
            chroma_persist_directory=str(tmp_path / "chroma"),
            report_output_dir=str(tmp_path / "reports"),
            log_dir=str(tmp_path / "logs"),
        )

        s.ensure_directories()

        assert (tmp_path / "chroma").is_dir()
        assert (tmp_path / "reports").is_dir()
        assert (tmp_path / "logs").is_dir()


class TestEnvironmentLoading:
    def test_values_come_from_the_environment_case_insensitively(self, monkeypatch):
        monkeypatch.setenv("TOP_K_RESULTS", "9")
        monkeypatch.setenv("provider_mode", "local")

        s = Settings(_env_file=None)

        assert s.top_k_results == 9
        assert s.use_openai is False

    def test_an_unknown_variable_does_not_break_startup(self, monkeypatch):
        monkeypatch.setenv("SOME_UNRELATED_VARIABLE", "x")

        assert Settings(_env_file=None).app_version

    def test_the_api_key_is_optional_so_the_app_starts_without_one(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        assert Settings(_env_file=None).openai_api_key is None


class TestTheExampleEnvironmentFile:
    """
    `.env.example` is the documentation everyone actually reads, and
    documentation that drifts from the code is worse than none. These keep it
    honest in both directions.
    """

    @staticmethod
    def _example_keys() -> set:
        keys = set()
        for line in ENV_EXAMPLE.read_text().splitlines():
            line = line.split("#")[0].strip()
            if "=" in line:
                keys.add(line.split("=", 1)[0].strip().lower())
        return keys

    # Read by scripts/start_api.sh and src/api/main.py's __main__ block, not by
    # Settings, so it is legitimately in the file and not a field.
    NON_SETTING_KEYS = {"port"}

    def test_every_setting_appears_in_the_example(self):
        missing = set(Settings.model_fields) - self._example_keys()

        assert not missing, f".env.example does not document: {sorted(missing)}"

    def test_the_example_names_no_setting_that_does_not_exist(self):
        unknown = self._example_keys() - set(Settings.model_fields) - self.NON_SETTING_KEYS

        assert not unknown, f".env.example names unknown settings: {sorted(unknown)}"

    def test_the_example_loads_without_raising(self):
        """A file whose defaults the validator rejects is a broken example."""
        values = {}
        for line in ENV_EXAMPLE.read_text().splitlines():
            line = line.split("#")[0].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().lower()
            if key in Settings.model_fields and value.strip():
                values[key] = value.strip()

        assert Settings(**values).chunk_overlap < Settings(**values).chunk_size
