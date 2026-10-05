"""AgentBase: shared interface for all review agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AgentBase(ABC):
    """Abstract base class that every agent must implement."""

    def __init__(self, name: str) -> None:
        self.name = name

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def analyze(self, code: str) -> Any:
        """Run analysis on the submitted code and return a result object."""

    @abstractmethod
    def report(self) -> str:
        """Return a human-readable summary of the last analysis."""

    @abstractmethod
    def confidence_score(self) -> float:
        """Return a float in [0.0, 1.0] expressing confidence in the last result."""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"
