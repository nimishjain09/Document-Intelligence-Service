"""Helper to discover supported documents in a directory."""

from __future__ import annotations

from pathlib import Path

from doc_summarizer.ingestion.factory import LoaderFactory


def discover_documents(folder: Path) -> list[Path]:
    """Return all supported document paths in a folder (non-recursive)."""
    supported = set(LoaderFactory.supported_extensions())
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in supported
    ]