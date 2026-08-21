"""Unit tests for the text chunking utility."""

from __future__ import annotations

from doc_summarizer.summarization.chunking import chunk_text


def test_empty_text_returns_empty_list() -> None:
    assert chunk_text("", 100) == []


def test_short_text_single_chunk() -> None:
    text = "Short sentence."
    assert chunk_text(text, 100) == [text]


def test_long_text_splits_into_multiple_chunks() -> None:
    text = "\n".join([f"Paragraph number {i}." for i in range(50)])
    chunks = chunk_text(text, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_oversized_paragraph_is_hard_split() -> None:
    text = "x" * 250
    chunks = chunk_text(text, 100)
    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)