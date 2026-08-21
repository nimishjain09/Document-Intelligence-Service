"""Utility to split long text into model-sized chunks."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            for i in range(0, len(para), chunk_size):
                chunks.append(para[i : i + chunk_size])
            continue
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks