"""LangGraph agent module for RAG-based report generation.

Exports are resolved lazily. Importing `ReportGenerationGraph` eagerly pulls in
langgraph, langchain-openai, chromadb and torch, which meant that any module in
this package — including the dependency-free offline writer — could only be
imported with the full stack installed. `__getattr__` keeps the public names
while letting light modules be imported on their own.
"""

from typing import TYPE_CHECKING

__all__ = ["ReportGenerationGraph", "AgentState"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .graph import ReportGenerationGraph
    from .state import AgentState


def __getattr__(name: str):
    if name == "ReportGenerationGraph":
        from .graph import ReportGenerationGraph

        return ReportGenerationGraph
    if name == "AgentState":
        from .state import AgentState

        return AgentState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
