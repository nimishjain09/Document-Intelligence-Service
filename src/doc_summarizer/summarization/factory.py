"""Factory that builds a summarization strategy from settings."""

from __future__ import annotations

from doc_summarizer.config.logging_config import get_logger
from doc_summarizer.config.settings import Settings
from doc_summarizer.summarization.base import SummarizerStrategy
from doc_summarizer.summarization.strategies import HuggingFaceSummarizer

logger = get_logger(__name__)


class SummarizerFactory:
    """Creates summarization strategies based on configuration."""

    @staticmethod
    def create(settings: Settings) -> SummarizerStrategy:
        """Return a configured summarization strategy.

        Chooses the quantized strategy when `settings.quantize` is True,
        otherwise the standard Hugging Face pipeline strategy.
        """
        if settings.quantize:
            # Import lazily so standard runs don't pay the import cost.
            from doc_summarizer.summarization.quantized import QuantizedSummarizer

            logger.info("Using QuantizedSummarizer (INT8, CPU).")
            return QuantizedSummarizer(
                model_name=settings.model_name,
                chunk_size=settings.chunk_size,
                max_length=settings.max_output_tokens,
            )

        logger.info("Using HuggingFaceSummarizer (standard).")
        return HuggingFaceSummarizer(
            model_name=settings.model_name,
            chunk_size=settings.chunk_size,
            max_length=settings.max_output_tokens,
        )