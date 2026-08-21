"""Command-line interface for the document summarization service."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
from doc_summarizer.summarization.models import SummaryResult


def _build_parser() -> argparse.ArgumentParser:
    """Define CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="doc-summarizer",
        description="Summarize documents (PDF, TXT, DOCX) concurrently.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a document file or a folder of documents.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional directory to write summary files into.",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=None,
        help="Override MAX_CONCURRENCY for this run.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress logs and show only the summaries.",
    )
    return parser


def _write_outputs(results: list[SummaryResult], output_dir: Path) -> None:
    """Write each summary to a .summary.txt file in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        out_path = output_dir / f"{result.source_name}.summary.txt"
        out_path.write_text(result.summary, encoding="utf-8")


def _print_results(results: list[SummaryResult], fmt: str) -> None:
    """Print results to stdout in the chosen format."""
    if fmt == "json":
        payload = [
            {
                "source": r.source_name,
                "summary": r.summary,
                "original_chars": r.original_chars,
                "summary_chars": r.summary_chars,
                "compression_ratio": r.compression_ratio,
            }
            for r in results
        ]
        print(json.dumps(payload, indent=2))
    else:
        for r in results:
            print(f"\n----- {r.source_name} -----")
            print(r.summary)
            print("-" * (12 + len(r.source_name)))


async def _run_async(args: argparse.Namespace) -> None:
    """Core async CLI logic."""
    settings = get_settings()

    # In quiet mode, only show warnings and errors.
    log_level = "WARNING" if args.quiet else settings.log_level
    configure_logging(log_level)
    correlation_id.set(str(uuid.uuid4())[:8])
    logger = get_logger(__name__)

    target: Path = args.path

    if not target.exists():
        logger.error("Path does not exist: '%s'", target)
        return

    if target.is_dir():
        file_paths = discover_documents(target)
    else:
        file_paths = [target]

    if not file_paths:
        logger.warning(
            "No supported documents found at '%s'. Supported: %s",
            target,
            SummarizerFactory  # placeholder; real list below
            and __import__("doc_summarizer.ingestion.factory", fromlist=["LoaderFactory"]).LoaderFactory.supported_extensions(),
        )
        return

    concurrency = args.concurrency or settings.max_concurrency
    summarizer = SummarizerFactory.create(settings)
    orchestrator = SummarizationOrchestrator(
        summarizer=summarizer,
        max_concurrency=concurrency,
    )

    results = await orchestrator.process_many(file_paths)

    if not args.quiet:
        logger.info(
            "Finished: %d succeeded, %d failed.",
            len(results),
            len(file_paths) - len(results),
        )

    _print_results(results, args.format)

    if args.output is not None:
        _write_outputs(results, args.output)
        if not args.quiet:
            logger.info("Wrote %d summary file(s) to '%s'.", len(results), args.output)


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()