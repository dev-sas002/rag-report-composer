"""Tests for cost tracking functionality."""

from unittest.mock import Mock, patch

import pytest

from src.observability.cost_tracker import CostTracker


@pytest.fixture
def cost_tracker(tmp_path):
    """Create a cost tracker instance for testing."""
    with patch("src.observability.cost_tracker.get_settings") as mock:
        settings = Mock()
        settings.enable_cost_tracking = True
        settings.cost_log_file = str(tmp_path / "cost_tracking.json")
        mock.return_value = settings
        tracker = CostTracker()
        yield tracker


def test_token_counting(cost_tracker):
    """Test token counting functionality."""
    text = "This is a test sentence."
    tokens = cost_tracker.count_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)


def test_cost_calculation(cost_tracker):
    """Test cost calculation for different models."""
    model = "gpt-4-turbo-preview"
    input_tokens = 1000
    output_tokens = 500

    cost = cost_tracker.calculate_cost(model, input_tokens, output_tokens)

    assert cost > 0
    assert isinstance(cost, float)
    # GPT-4 turbo: $10 per 1M input tokens, $30 per 1M output tokens
    expected_cost = (1000 / 1_000_000 * 10) + (500 / 1_000_000 * 30)
    assert abs(cost - expected_cost) < 0.0001


def test_session_tracking(cost_tracker):
    """Test session cost tracking."""
    cost_tracker.track_call("gpt-4-turbo-preview", 1000, 500, "test_operation")

    summary = cost_tracker.get_session_summary()

    assert summary["total_cost_usd"] > 0
    assert summary["total_input_tokens"] == 1000
    assert summary["total_output_tokens"] == 500
    assert "gpt-4-turbo-preview" in summary["by_model"]


def test_unknown_model_cost(cost_tracker):
    """Test handling of unknown model."""
    cost = cost_tracker.calculate_cost("unknown-model", 1000, 500)
    assert cost == 0.0


def test_disabled_cost_tracking_avoids_updates(cost_tracker):
    """When cost tracking is disabled, calls should not update session state."""
    cost_tracker.settings.enable_cost_tracking = False

    cost = cost_tracker.track_call("gpt-4-turbo-preview", 1000, 500, "disabled_operation")

    assert cost == 0.0
    summary = cost_tracker.get_session_summary()
    assert summary["total_cost_usd"] == 0.0
    assert summary["total_input_tokens"] == 0
    assert summary["total_output_tokens"] == 0
    assert summary["by_model"] == {}


def test_log_write_failure_is_non_fatal(cost_tracker, monkeypatch):
    """Failures writing the cost log file should not raise and should disable further writes."""

    def failing_open(*args, **kwargs):
        raise OSError("simulated write failure")

    # Force log writes to fail
    monkeypatch.setattr("src.observability.cost_tracker.open", failing_open, raising=False)

    # First call should not raise even though logging fails
    cost = cost_tracker.track_call("gpt-4-turbo-preview", 1000, 0, "test_operation")
    assert cost > 0

    # Subsequent calls should also succeed and not attempt further writes
    cost_tracker.track_call("gpt-4-turbo-preview", 500, 0, "second_operation")

    summary = cost_tracker.get_session_summary()
    assert summary["total_input_tokens"] == 1500


def test_log_session_summary_emits_event(cost_tracker, caplog):
    """Session summary logging should use the structured logger."""
    cost_tracker.track_call("gpt-4-turbo-preview", 1000, 500, "summary_test")

    with caplog.at_level("INFO"):
        cost_tracker.log_session_summary()

    # We cannot rely on exact structlog formatting, but the event name should appear.
    assert any("session_cost_summary" in message for message in caplog.text.splitlines())
