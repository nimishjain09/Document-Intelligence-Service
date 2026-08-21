"""Concrete summarization strategies using Hugging Face pipelines."""

from __future__ import annotations

from transformers import pipeline

from doc_summarizer.config.logging_config import get_logger
from doc_summarizer.core.exceptions import SummarizationError
from doc_summarizer.summarization.base import SummarizerStrategy
from doc_summarizer.summarization.chunking import chunk_text
from doc_summarizer.summarization.retry import with_retry

logger = get_logger(__name__)


class HuggingFaceSummarizer(SummarizerStrategy):
    """Summarizer backed by a Hugging Face summarization pipeline."""

    def __init__(
        self,
        model_name: str,
        chunk_size: int,
        max_length: int = 250,
        min_length: int = 80,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.min_length = min_length
        logger.info("Loading summarization pipeline: %s", model_name)
        try:
            self._pipeline = pipeline("summarization", model=model_name)
        except Exception as exc:
            raise SummarizationError(
                f"Failed to load model '{model_name}': {exc}"
            ) from exc

        self.chunk_size = self._validate_chunk_size(chunk_size)

    def _validate_chunk_size(self, chunk_size: int) -> int:
        """Validate CHUNK_SIZE against the model's max input token limit."""
        model_max_tokens = getattr(
            self._pipeline.tokenizer, "model_max_length", 1024
        )
        if model_max_tokens > 100_000:
            model_max_tokens = 1024

        safe_char_limit = model_max_tokens * 4

        if chunk_size > safe_char_limit:
            logger.warning(
                "CHUNK_SIZE=%d exceeds model limit (~%d chars for %d tokens). "
                "Capping to %d to avoid truncation.",
                chunk_size,
                safe_char_limit,
                model_max_tokens,
                safe_char_limit,
            )
            return safe_char_limit

        logger.info(
            "CHUNK_SIZE=%d is within model limit (~%d chars / %d tokens).",
            chunk_size,
            safe_char_limit,
            model_max_tokens,
        )
        return chunk_size

    @with_retry(max_attempts=3, base_delay=0.5)
    def _summarize_chunk(self, chunk: str) -> str:
        """Summarize a single chunk with balanced length control."""
        input_tokens = len(self._pipeline.tokenizer.encode(chunk, truncation=True))

        # Only shrink for very short inputs; otherwise use full max_length
        # so summaries are complete and not truncated.
        if input_tokens < 60:
            dynamic_max = max(20, int(input_tokens * 0.8))
            dynamic_min = max(10, int(dynamic_max * 0.5))
        else:
            # Allow up to the configured max_length, capped by input length.
            dynamic_max = min(self.max_length, max(self.min_length, input_tokens))
            dynamic_min = min(self.min_length, int(dynamic_max * 0.4))

        try:
            result = self._pipeline(
                chunk,
                max_length=dynamic_max,
                min_length=dynamic_min,
                do_sample=False,
                truncation=True,
            )
            return result[0]["summary_text"].strip()
        except Exception as exc:
            raise SummarizationError(f"Chunk summarization failed: {exc}") from exc

    def summarize(self, text: str) -> str:
        """Chunk, summarize each part, then combine into a final summary."""
        chunks = chunk_text(text, self.chunk_size)
        if not chunks:
            raise SummarizationError("No text provided to summarize.")

        logger.info("Summarizing %d chunk(s).", len(chunks))
        partial_summaries = [self._summarize_chunk(c) for c in chunks]

        # Single chunk: return its summary directly (no lossy second pass).
        if len(partial_summaries) == 1:
            return partial_summaries[0]

        # Multiple chunks: JOIN the summaries instead of re-summarizing,
        # so no content is lost to a second compression pass.
        logger.info("Combining %d chunk summaries.", len(partial_summaries))
        return " ".join(partial_summaries)