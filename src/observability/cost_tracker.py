"""
Cost tracking for LLM API calls.
Tracks token usage and estimated costs for OpenAI API calls.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import tiktoken

from ..config import get_settings
from .logger import get_logger

logger = get_logger(__name__)


class CostTracker:
    """
    Tracks and logs costs for LLM API calls.
    Supports multiple OpenAI models with different pricing.
    """

    # OpenAI pricing (per 1M tokens) - as of 2024
    PRICING = {
        "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
        # Local backends are listed explicitly at zero rather than falling
        # through to the unknown-model path, so a keyless run does not emit a
        # pricing warning on every call for something that is genuinely free.
        "local-extractive-writer": {"input": 0.0, "output": 0.0},
        "sentence-transformers/all-MiniLM-L6-v2": {"input": 0.0, "output": 0.0},
    }

    def __init__(self):
        """Initialize the cost tracker."""
        self.settings = get_settings()
        self.cost_log_file = Path(self.settings.cost_log_file)

        # Guard file-system interaction so that cost tracking never prevents
        # the application from starting (e.g. in read-only environments).
        self._file_logging_enabled = True
        try:
            self.cost_log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:  # pragma: no cover - defensive, environment-specific
            logger.error("cost_log_dir_create_failed", error=str(e), path=str(self.cost_log_file))
            self._file_logging_enabled = False

        self.session_costs: Dict[str, float] = {}
        self.session_tokens: Dict[str, Dict[str, int]] = {}

        # Initialize encoding for token counting. tiktoken fetches the BPE
        # file on first use, so this must not be fatal: an air-gapped or
        # offline run has no key either, and should still start.
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:  # pragma: no cover - depends on network/cache
            logger.warning("tiktoken_encoding_unavailable", error=str(e))
            self.encoding = None

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text for a specific model.

        Args:
            text: Text to count tokens for
            model: Model name (optional)

        Returns:
            Number of tokens
        """
        try:
            if model and "gpt" in model:
                encoding = tiktoken.encoding_for_model(model)
            else:
                encoding = self.encoding
            if encoding is None:
                raise RuntimeError("no tiktoken encoding available")
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning("token_count_failed", error=str(e), model=model)
            # Fallback: rough estimation
            return len(text) // 4

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        """
        Calculate cost for a model call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        if model not in self.PRICING:
            logger.warning("unknown_model_pricing", model=model)
            return 0.0

        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def _build_log_entry(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        operation: str,
        metadata: Optional[Dict],
        cost: float,
    ) -> Dict:
        """Create a serialisable log entry for a tracked call."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "metadata": metadata or {},
        }

    def _write_log_entry(self, log_entry: Dict) -> None:
        """
        Persist a log entry to disk if file logging is enabled.

        Any failure disables further file writes for the lifetime of the
        process but does not raise, so API calls can continue unaffected.
        """
        if not self._file_logging_enabled:
            return

        try:
            with open(self.cost_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:  # pragma: no cover - depends on filesystem failures
            self._file_logging_enabled = False
            logger.error("cost_log_write_failed", error=str(e), path=str(self.cost_log_file))

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        operation: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> float:
        """
        Track an API call and log its cost.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            operation: Operation name/description
            metadata: Additional metadata to log

        Returns:
            Cost of the call in USD
        """
        if not self.settings.enable_cost_tracking:
            return 0.0

        cost = self.calculate_cost(model, input_tokens, output_tokens)

        # Update session tracking
        if model not in self.session_costs:
            self.session_costs[model] = 0.0
            self.session_tokens[model] = {"input": 0, "output": 0}

        self.session_costs[model] += cost
        self.session_tokens[model]["input"] += input_tokens
        self.session_tokens[model]["output"] += output_tokens

        # Log the call
        log_entry = self._build_log_entry(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            operation=operation,
            metadata=metadata,
            cost=cost,
        )

        logger.info(
            "api_call_tracked",
            model=model,
            operation=operation,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        # Append to cost log file (non-fatal on failure)
        self._write_log_entry(log_entry)

        return cost

    def get_session_summary(self) -> Dict:
        """
        Get summary of costs for current session.

        Returns:
            Dictionary with cost and token summaries
        """
        total_cost = sum(self.session_costs.values())
        total_input_tokens = sum(t["input"] for t in self.session_tokens.values())
        total_output_tokens = sum(t["output"] for t in self.session_tokens.values())

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "by_model": {
                model: {
                    "cost_usd": round(cost, 6),
                    "tokens": self.session_tokens[model],
                }
                for model, cost in self.session_costs.items()
            },
        }

    def log_session_summary(self) -> None:
        """Log the session summary."""
        summary = self.get_session_summary()
        logger.info("session_cost_summary", **summary)


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """
    Get the global cost tracker instance.

    Returns:
        CostTracker instance
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker
