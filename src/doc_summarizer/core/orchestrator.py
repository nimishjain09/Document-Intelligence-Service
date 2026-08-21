"""Async orchestrator for concurrent document summarization."""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path

from doc_summarizer.config.logging_config import correlation_id, get_logger
from doc_summarizer.core.exceptions import SummarizerError
from doc_summarizer.ingestion.factory import LoaderFactory
from doc_summarizer.summarization.base import SummarizerStrategy
from doc_summarizer.summarization.models import SummaryResult

logger = get_logger(__name__)

# Maximum seconds to wait for a single document before giving up.
DOCUMENT_TIMEOUT_SECONDS = 120


class SummarizationOrchestrator:
    """Coordinates concurrent summarization of multiple documents."""

    def __init__(
        self,
        summarizer: SummarizerStrategy,
        max_concurrency: int,
    ) -> None:
        self._summarizer = summarizer
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # Serialize actual model inference to avoid MPS/thread deadlocks.
        self._model_lock = threading.Lock()

    async def _process_one(self, file_path: Path) -> SummaryResult | None:
        """Process a single document with error isolation and a timeout."""
        doc_id = str(uuid.uuid4())[:8]
        correlation_id.set(doc_id)

        async with self._semaphore:
            try:
                # Ingestion is lightweight and safe to run inline.
                document = LoaderFactory.get_loader(file_path).load()

                def _summarize_with_context() -> str:
                    # Re-set the correlation ID inside the worker thread,
                    # since ContextVars do not auto-propagate to executors.
                    correlation_id.set(doc_id)
                    # Only one thread runs the model at a time (MPS-safe).
                    with self._model_lock:
                        return self._summarizer.summarize(document.text)

                loop = asyncio.get_running_loop()

                # Offload blocking inference to a thread, with a timeout
                # so a stuck document can never hang the whole batch.
                summary_text = await asyncio.wait_for(
                    loop.run_in_executor(None, _summarize_with_context),
                    timeout=DOCUMENT_TIMEOUT_SECONDS,
                )

                result = SummaryResult(
                    source_name=document.source_path.name,
                    summary=summary_text,
                    original_chars=document.char_count,
                    summary_chars=len(summary_text),
                    chunks_processed=1,
                )
                logger.info(
                    "Completed '%s' (compression %.3f).",
                    result.source_name,
                    result.compression_ratio,
                )
                return result

            except asyncio.TimeoutError:
                logger.error(
                    "Timed out after %ds on '%s'; skipping.",
                    DOCUMENT_TIMEOUT_SECONDS,
                    file_path.name,
                )
                return None

            except SummarizerError as exc:
                logger.error("Skipped '%s': %s", file_path.name, exc)
                return None

    async def process_many(self, file_paths: list[Path]) -> list[SummaryResult]:
        """Summarize many documents concurrently, isolating failures."""
        logger.info("Processing %d document(s) concurrently.", len(file_paths))
        tasks = [self._process_one(path) for path in file_paths]
        results = await asyncio.gather(*tasks)
        # Filter out failed / timed-out (None) documents.
        return [r for r in results if r is not None]