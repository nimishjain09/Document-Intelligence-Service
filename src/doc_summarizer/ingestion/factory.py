"""Factory that selects the correct DocumentLoader by file extension."""

from __future__ import annotations

from pathlib import Path

from doc_summarizer.core.exceptions import UnsupportedFileTypeError
from doc_summarizer.ingestion.base import DocumentLoader
from doc_summarizer.ingestion.loaders import DocxLoader, PdfLoader, TxtLoader


class LoaderFactory:
    """Creates the appropriate loader instance for a given file."""

    _registry: dict[str, type[DocumentLoader]] = {
        ".txt": TxtLoader,
        ".pdf": PdfLoader,
        ".docx": DocxLoader,
    }

    @classmethod
    def get_loader(cls, path: Path) -> DocumentLoader:
        """Return a loader instance matching the file's extension."""
        extension = path.suffix.lower()
        loader_cls = cls._registry.get(extension)
        if loader_cls is None:
            raise UnsupportedFileTypeError(
                f"No loader registered for '{extension}' "
                f"(supported: {', '.join(cls._registry)})"
            )
        return loader_cls(path)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return the list of supported file extensions."""
        return list(cls._registry)