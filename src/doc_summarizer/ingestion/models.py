"""Data models for the ingestion layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    """Represents a document after text extraction."""

    source_path: Path
    text: str

    @property
    def char_count(self) -> int:
        """Number of characters in the extracted text."""
        return len(self.text)