"""Generative Q&A agent: retrieval + LLM for conversational answers."""

from __future__ import annotations

from pathlib import Path

from doc_summarizer.agent.embedder import Embedder
from doc_summarizer.agent.llm_client import OllamaLLM
from doc_summarizer.agent.vector_store import VectorStore
from doc_summarizer.config.logging_config import get_logger
from doc_summarizer.core.exceptions import SummarizerError
from doc_summarizer.ingestion.factory import LoaderFactory
from doc_summarizer.summarization.chunking import chunk_text

logger = get_logger(__name__)


class GenerativeQAAgent:
    """Answers questions using retrieval + a generative LLM (Ollama)."""

    def __init__(
        self,
        chunk_size: int = 400,
        llm_model: str = "llama3.2",
    ) -> None:
        self.chunk_size = chunk_size
        self._embedder = Embedder()
        self._store = VectorStore()
        self._llm = OllamaLLM(llm_model)
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

    def ask(self, question: str, top_k: int = 4) -> dict:
        """Answer with the LLM and have it self-rate its confidence."""
        if not self._ready:
            raise SummarizerError("No document indexed. Call index_document first.")

        query_vector = self._embedder.embed_one(question)
        relevant_chunks = self._store.search(query_vector, top_k=top_k)
        context = "\n\n".join(relevant_chunks)

        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the "
            "context below. Then rate your confidence from 0.0 to 1.0.\n"
            "Be honest and critical: use 1.0 ONLY if the context explicitly and "
            "completely answers the question. Use lower values if the answer is "
            "partial, inferred, or uncertain.\n\n"
            "Respond in EXACTLY this format:\n"
            "ANSWER: <your answer>\n"
            "CONFIDENCE: <number between 0.0 and 1.0>\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
        )

        raw = self._llm.generate(prompt)
        answer, confidence = self._parse_response(raw)

        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "context_used": context[:400],
        }

    def _parse_response(self, raw: str) -> tuple[str, float]:
        """Parse the LLM's ANSWER/CONFIDENCE formatted response."""
        answer = raw.strip()
        confidence = 0.5

        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("ANSWER:"):
                answer = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError):
                    confidence = 0.5

        return answer, confidence