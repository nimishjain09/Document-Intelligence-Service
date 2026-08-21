"""Unit tests for the Q&A agent components (no models loaded)."""

from __future__ import annotations

import numpy as np

from doc_summarizer.agent.vector_store import VectorStore


def test_vector_store_build_and_search() -> None:
    chunks = ["cats are animals", "python is a language", "the sky is blue"]
    # Simple fake 3-dim embeddings.
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype="float32",
    )
    store = VectorStore()
    store.build(chunks, embeddings)

    # Query close to the second embedding.
    query = np.array([0.0, 0.9, 0.1], dtype="float32")
    results = store.search(query, top_k=1)

    assert results == ["python is a language"]


def test_empty_store_returns_empty() -> None:
    store = VectorStore()
    query = np.array([1.0, 0.0, 0.0], dtype="float32")
    assert store.search(query) == []