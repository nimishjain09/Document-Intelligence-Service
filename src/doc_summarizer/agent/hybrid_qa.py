"""Hybrid Q&A agent: runs both agents and returns the higher-confidence answer."""

from __future__ import annotations

from pathlib import Path

from doc_summarizer.agent.generative_qa import GenerativeQAAgent
from doc_summarizer.agent.qa_agent import DocumentQAAgent
from doc_summarizer.config.logging_config import get_logger

logger = get_logger(__name__)


class HybridQAAgent:
    """Runs extractive and generative agents, returns the more confident one."""

    def __init__(
        self,
        chunk_size: int = 400,
        llm_model: str = "llama3.2",
    ) -> None:
        self._extractive = DocumentQAAgent(chunk_size=chunk_size)
        self._generative = GenerativeQAAgent(
            chunk_size=chunk_size, llm_model=llm_model
        )
        self._ready = False

    def index_document(self, file_path: Path) -> int:
        """Index the document in both agents."""
        n = self._extractive.index_document(file_path)
        self._generative.index_document(file_path)
        self._ready = True
        logger.info("Hybrid agent indexed document (%d chunks).", n)
        return n

    def ask(self, question: str) -> dict:
        """Run both agents and return whichever has higher confidence."""
        extractive = self._extractive.ask(question)
        generative = self._generative.ask(question)

        ext_conf = extractive["confidence"]
        # Discount the LLM's self-reported confidence (LLMs over-report).
        gen_conf = generative["confidence"] * 0.75

        logger.info(
            "Extractive=%.3f, Generative=%.3f (raw %.3f)",
            ext_conf, gen_conf, generative["confidence"],
        )

        # Prefer extractive for confident factual answers.
        if ext_conf >= gen_conf:
            logger.info("Selected EXTRACTIVE answer.")
            return {
                "question": question,
                "answer": extractive["answer"],
                "confidence": round(ext_conf, 3),
                "mode": "extractive",
                "context_used": extractive["context_used"],
            }

        logger.info("Selected GENERATIVE answer.")
        return {
            "question": question,
            "answer": generative["answer"],
            "confidence": round(gen_conf, 3),
            "mode": "generative",
            "context_used": generative["context_used"],
        }