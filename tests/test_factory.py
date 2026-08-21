"""Unit tests for the loader factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_summarizer.core.exceptions import UnsupportedFileTypeError
from doc_summarizer.ingestion.factory import LoaderFactory
from doc_summarizer.ingestion.loaders import DocxLoader, PdfLoader, TxtLoader


def test_txt_returns_txt_loader() -> None:
    loader = LoaderFactory.get_loader(Path("sample.txt"))
    assert isinstance(loader, TxtLoader)


def test_pdf_returns_pdf_loader() -> None:
    loader = LoaderFactory.get_loader(Path("sample.pdf"))
    assert isinstance(loader, PdfLoader)


def test_docx_returns_docx_loader() -> None:
    loader = LoaderFactory.get_loader(Path("sample.docx"))
    assert isinstance(loader, DocxLoader)


def test_unsupported_extension_raises() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        LoaderFactory.get_loader(Path("sample.csv"))


def test_supported_extensions_listed() -> None:
    exts = LoaderFactory.supported_extensions()
    assert ".txt" in exts
    assert ".pdf" in exts
    assert ".docx" in exts