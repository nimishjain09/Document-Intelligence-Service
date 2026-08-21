"""Document Q&A agent using retrieval + a Hugging Face QA model."""

from __future__ import annotations

from pathlib import Path

from transformers import pipeline

from doc_summarizer.agent.embedder import Embedder
from doc_summarizer.agent.vector_store import VectorStore
from doc_summarizer.config.logging_config import get_logger
from doc_summarizer.core.exceptions import SummarizerError
from doc_summarizer.ingestion.factory import LoaderFactory
from doc_summarizer.summarization.chunking import chunk_text

logger = get_logger(__name__)


class DocumentQAAgent:
    """Answers questions about a document using retrieval-augmented QA."""

    def __init__(
        self,
        chunk_size: int = 500,
        qa_model: str = "distilbert-base-cased-distilled-squad",
    ) -> None:
        self.chunk_size = chunk_size
        self._embedder = Embedder()
        self._store = VectorStore()
        logger.info("Loading QA model: %s", qa_model)
        self._qa = pipeline("question-answering", model=qa_model)
        self._ready = False

    def index_document(self, file_path: Path) -> int:
        """Load, chunk, embed, and index a document. Returns chunk count."""
        document = LoaderFactory.get_loader(file_path).load()
        chunks = chunk_text(document.text, self.chunk_size)
        if not chunks:
            raise SummarizerError("Document has no text to index.")

        embeddings = self._embedder.embed(chunks)
        self._store.build(chunks, embeddings)
        self._ready = True
        logger.info("Indexed document with %d chunks.", len(chunks))
        return len(chunks)

    def ask(self, question: str, top_k: int = 3) -> dict:
        """Answer a question using the indexed document."""
        if not self._ready:
            raise SummarizerError("No document indexed. Call index_document first.")

        query_vector = self._embedder.embed_one(question)
        relevant_chunks = self._store.search(query_vector, top_k=top_k)
        context = " ".join(relevant_chunks)

        result = self._qa(question=question, context=context)

        return {
            "question": question,
            "answer": result["answer"],
            "confidence": round(float(result["score"]), 3),
            "context_used": context[:300],
        }