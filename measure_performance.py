"""Measure and record real latency numbers for the report."""

import time
from pathlib import Path

from doc_summarizer.config.settings import get_settings
from doc_summarizer.summarization.factory import SummarizerFactory
from doc_summarizer.ingestion.factory import LoaderFactory

DOC = Path("docs/long_article.txt")

# --- Measure summarization ---
settings = get_settings()
document = LoaderFactory.get_loader(DOC).load()

start_load = time.perf_counter()
summarizer = SummarizerFactory.create(settings)
load_time = time.perf_counter() - start_load

start = time.perf_counter()
summary = summarizer.summarize(document.text)
summ_time = time.perf_counter() - start

print("=== SUMMARIZATION MEASUREMENTS ===")
print(f"Document size:        {len(document.text)} chars")
print(f"Model load time:      {load_time:.2f} s")
print(f"Summarization time:   {summ_time:.2f} s")
print(f"Summary length:       {len(summary)} chars")
print(f"Compression ratio:    {round(len(summary)/len(document.text), 3)}")
print(f"\nSummary:\n{summary}")