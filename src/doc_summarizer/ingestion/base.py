"""Abstract base class for all document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from doc_summarizer.ingestion.models import LoadedDocument


class DocumentLoader(ABC):
    """Abstract interface every concrete loader must implement."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @abstractmethod
    def load(self) -> LoadedDocument:
        """Read the file and return a LoadedDocument."""
        raise NotImplementedError