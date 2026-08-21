"""Abstract base class for summarization strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SummarizerStrategy(ABC):
    """Interface all summarization strategies must implement."""

    @abstractmethod
    def summarize(self, text: str) -> str:
        """Return a concise summary of the given text."""
        raise NotImplementedError