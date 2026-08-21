"""Data models for the summarization layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SummaryResult:
    """Holds the summarization output and metadata."""

    source_name: str
    summary: str
    original_chars: int
    summary_chars: int
    chunks_processed: int

    @property
    def compression_ratio(self) -> float:
        """Ratio of summary length to original length."""
        if self.original_chars == 0:
            return 0.0
        return round(self.summary_chars / self.original_chars, 3)