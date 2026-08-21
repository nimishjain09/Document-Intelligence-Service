"""Quantized summarization strategy for optimized CPU inference."""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from doc_summarizer.config.logging_config import get_logger
from doc_summarizer.core.exceptions import SummarizationError
from doc_summarizer.summarization.base import SummarizerStrategy
from doc_summarizer.summarization.chunking import chunk_text
import warnings
warnings.filterwarnings("ignore", message=".*reduce_range.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

logger = get_logger(__name__)


def _select_quantization_engine() -> None:
    """Select an available quantization backend engine.

    Apple Silicon (ARM) uses QNNPACK; x86 uses FBGEMM. If neither is
    available, quantization is not supported on this platform.
    """
    supported = torch.backends.quantized.supported_engines
    if "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
        logger.info("Quantization engine set to QNNPACK (ARM).")
    elif "fbgemm" in supported:
        torch.backends.quantized.engine = "fbgemm"
        logger.info("Quantization engine set to FBGEMM (x86).")
    else:
        raise SummarizationError(
            "No quantization engine available on this platform "
            f"(supported: {supported}). Disable QUANTIZE to run standard mode."
        )


class QuantizedSummarizer(SummarizerStrategy):
    """Summarizer using dynamic INT8 quantization on CPU."""

    def __init__(
        self,
        model_name: str,
        chunk_size: int,
        max_length: int = 200,
        min_length: int = 50,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.min_length = min_length

        logger.info("Loading and quantizing model: %s", model_name)
        try:
            # Select the correct quantization backend for this CPU.
            _select_quantization_engine()

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Dynamic INT8 quantization on Linear layers (CPU only).
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

            self._pipeline = pipeline(
                "summarization",
                model=quantized_model,
                tokenizer=tokenizer,
                device=-1,  # quantized models must run on CPU
            )
            logger.info("Quantization complete (INT8, CPU).")
            self.chunk_size = self._validate_chunk_size(chunk_size)
        except SummarizationError:
            raise
        except Exception as exc:
            raise SummarizationError(
                f"Failed to quantize model '{model_name}': {exc}"
            ) from exc
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

    def _summarize_chunk(self, chunk: str) -> str:
        """Summarize a single chunk with balanced length control."""
        input_tokens = len(self._pipeline.tokenizer.encode(chunk, truncation=True))

        # Only shrink for very short inputs; otherwise use full max_length
        # so summaries are complete and not truncated.
        if input_tokens < 60:
            dynamic_max = max(20, int(input_tokens * 0.8))
            dynamic_min = max(10, int(dynamic_max * 0.5))
        else:
            # Cap max_length just below the input length, but never below 80,
            # so the model can produce a complete, detailed summary.
            dynamic_max = min(self.max_length, max(80, int(input_tokens * 0.7)))
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