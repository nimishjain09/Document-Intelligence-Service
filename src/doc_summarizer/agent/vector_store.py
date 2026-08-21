"""FAISS-based vector store for document chunks."""

from __future__ import annotations

import faiss
import numpy as np

from doc_summarizer.config.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Stores chunk embeddings in a FAISS index for similarity search."""

    def __init__(self) -> None:
        self._index = None
        self._chunks: list[str] = []

    def build(self, chunks: list[str], embeddings: np.ndarray) -> None:
        """Build the index from chunks and their embeddings."""
        self._chunks = chunks
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings.astype("float32"))
        logger.info("Vector store built with %d chunks.", len(chunks))

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[str]:
        """Return the top_k most similar chunks to the query vector."""
        if self._index is None:
            return []
        query = np.array([query_vector]).astype("float32")
        distances, indices = self._index.search(query, top_k)
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]