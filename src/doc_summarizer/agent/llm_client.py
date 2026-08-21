"""Generative LLM client using a local Ollama model."""

from __future__ import annotations

import ollama

from doc_summarizer.config.logging_config import get_logger
from doc_summarizer.core.exceptions import SummarizerError

logger = get_logger(__name__)


class OllamaLLM:
    """Wraps a local Ollama model for generative text tasks."""

    def __init__(self, model_name: str = "llama3.2") -> None:
        self.model_name = model_name
        logger.info("Using Ollama LLM: %s", model_name)

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate a response from the LLM for the given prompt."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
            return response["message"]["content"].strip()
        except Exception as exc:
            raise SummarizerError(
                f"LLM generation failed (is Ollama running?): {exc}"
            ) from exc