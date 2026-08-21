"""Async FastAPI interface for the summarization service."""

from __future__ import annotations

import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from doc_summarizer.config.logging_config import (
    configure_logging,
    correlation_id,
    get_logger,
)
from doc_summarizer.config.settings import get_settings
from doc_summarizer.core.orchestrator import SummarizationOrchestrator
from doc_summarizer.ingestion.factory import LoaderFactory
from doc_summarizer.summarization.factory import SummarizerFactory

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

# Build the summarizer once at startup (model loads a single time).
_summarizer = SummarizerFactory.create(settings)
_orchestrator = SummarizationOrchestrator(
    summarizer=_summarizer,
    max_concurrency=settings.max_concurrency,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Log clean startup and shutdown lifecycle events."""
    logger.info("API starting up with model: %s", settings.model_name)
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Document Summarization Service",
    version="1.0.0",
    lifespan=lifespan,
)


class SummaryResponse(BaseModel):
    """API response schema for a summary."""

    source: str
    summary: str
    original_chars: int
    summary_chars: int
    compression_ratio: float


def _validate_files(files: List[UploadFile]) -> None:
    """Validate uploaded files, raising HTTP 400 on problems."""
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    supported = set(LoaderFactory.supported_extensions())
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in supported:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{suffix or 'unknown'}' "
                    f"for '{upload.filename}'. Supported: {sorted(supported)}"
                ),
            )


@app.get("/")
async def root() -> RedirectResponse:
    """Redirect the root path to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "model": settings.model_name}


@app.post("/summarize-one", response_model=SummaryResponse)
async def summarize_one(file: UploadFile = File(...)) -> SummaryResponse:
    """Summarize a SINGLE uploaded document."""
    _validate_files([file])
    correlation_id.set(str(uuid.uuid4())[:8])

    suffix = Path(file.filename or "file.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)

    try:
        results = await _orchestrator.process_many([temp_path])
    finally:
        temp_path.unlink(missing_ok=True)

    if not results:
        raise HTTPException(
            status_code=422,
            detail=f"Could not summarize '{file.filename}'.",
        )

    result = results[0]
    return SummaryResponse(
        source=file.filename or result.source_name,
        summary=result.summary,
        original_chars=result.original_chars,
        summary_chars=result.summary_chars,
        compression_ratio=result.compression_ratio,
    )


@app.post("/summarize", response_model=List[SummaryResponse])
async def summarize(files: List[UploadFile] = File(...)) -> List[SummaryResponse]:
    """Summarize MULTIPLE uploaded documents."""
    _validate_files(files)
    correlation_id.set(str(uuid.uuid4())[:8])

    temp_paths: List[Path] = []
    for upload in files:
        suffix = Path(upload.filename or "file.txt").suffix or ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            temp_paths.append(Path(tmp.name))

    try:
        results = await _orchestrator.process_many(temp_paths)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)

    response = []
    for original, result in zip(files, results):
        response.append(
            SummaryResponse(
                source=original.filename or result.source_name,
                summary=result.summary,
                original_chars=result.original_chars,
                summary_chars=result.summary_chars,
                compression_ratio=result.compression_ratio,
            )
        )
    return response