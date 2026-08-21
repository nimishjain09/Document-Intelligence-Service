"""Text embedding using sentence-transformers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from doc_summarizer.config.logging_config import get_logger

logger = get_logger(__name__)


class Embedder:
    """Wraps a sentence-transformer model to embed text chunks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for a list of texts as a numpy array."""
        return self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text and return its vector."""
        return self.embed([text])[0]