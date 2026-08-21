"""Unit tests for the async orchestrator using a mock summarizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_summarizer.core.orchestrator import SummarizationOrchestrator
from doc_summarizer.summarization.base import SummarizerStrategy


class MockSummarizer(SummarizerStrategy):
    """Fake summarizer that returns a fixed string (no model)."""

    def summarize(self, text: str) -> str:
        return "MOCK SUMMARY"


@pytest.mark.asyncio
async def test_process_single_txt_file(tmp_path: Path) -> None:
    doc = tmp_path / "a.txt"
    doc.write_text("Hello world, this is a test document.", encoding="utf-8")

    orchestrator = SummarizationOrchestrator(
        summarizer=MockSummarizer(), max_concurrency=2
    )
    results = await orchestrator.process_many([doc])

    assert len(results) == 1
    assert results[0].summary == "MOCK SUMMARY"
    assert results[0].source_name == "a.txt"


@pytest.mark.asyncio
async def test_process_multiple_files_concurrently(tmp_path: Path) -> None:
    docs = []
    for i in range(3):
        p = tmp_path / f"doc{i}.txt"
        p.write_text(f"Content of document {i}.", encoding="utf-8")
        docs.append(p)

    orchestrator = SummarizationOrchestrator(
        summarizer=MockSummarizer(), max_concurrency=2
    )
    results = await orchestrator.process_many(docs)

    assert len(results) == 3
    assert all(r.summary == "MOCK SUMMARY" for r in results)


@pytest.mark.asyncio
async def test_unsupported_file_is_skipped(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("Valid content.", encoding="utf-8")
    bad = tmp_path / "bad.csv"
    bad.write_text("unsupported", encoding="utf-8")

    orchestrator = SummarizationOrchestrator(
        summarizer=MockSummarizer(), max_concurrency=2
    )
    results = await orchestrator.process_many([good, bad])

    # Only the .txt succeeds; the .csv is skipped, not fatal.
    assert len(results) == 1
    assert results[0].source_name == "good.txt"