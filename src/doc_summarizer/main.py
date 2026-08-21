"""Entry point for the Modular Async Document Summarization Service."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from doc_summarizer.config.logging_config import (
    configure_logging,
    correlation_id,
    get_logger,
)
from doc_summarizer.config.settings import get_settings
from doc_summarizer.core.discovery import discover_documents
from doc_summarizer.core.orchestrator import SummarizationOrchestrator
from doc_summarizer.summarization.factory import SummarizerFactory


async def _main_async() -> None:
    """Async entry: discover, summarize concurrently, and report."""
    settings = get_settings()
    configure_logging(settings.log_level)
    correlation_id.set(str(uuid.uuid4())[:8])
    logger = get_logger(__name__)

    if len(sys.argv) < 2:
        logger.info("Usage: uv run doc-summarizer <file-or-folder>")
        return

    target = Path(sys.argv[1])

    # Accept a single file or a whole folder.
    if target.is_dir():
        file_paths = discover_documents(target)
    else:
        file_paths = [target]

    if not file_paths:
        logger.info("No supported documents found at '%s'.", target)
        return

    # Build the summarizer once, share it across all documents.
    summarizer = SummarizerFactory.create(settings)
    orchestrator = SummarizationOrchestrator(
        summarizer=summarizer,
        max_concurrency=settings.max_concurrency,
    )

    results = await orchestrator.process_many(file_paths)

    logger.info("Finished: %d succeeded, %d failed.",
                len(results), len(file_paths) - len(results))

    for result in results:
        print(f"\n----- {result.source_name} -----")
        print(result.summary)
        print("-" * (12 + len(result.source_name)))


def run() -> None:
    """Synchronous wrapper that launches the async event loop."""
    asyncio.run(_main_async())


if __name__ == "__main__":
    run()