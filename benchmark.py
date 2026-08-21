"""Benchmark standard (FP32) vs. quantized (INT8) summarization."""

from __future__ import annotations

import time

from doc_summarizer.config.logging_config import configure_logging
from doc_summarizer.config.settings import Settings
from doc_summarizer.summarization.factory import SummarizerFactory

# Suppress info logs so the benchmark table is clean.
configure_logging("WARNING")

# A realistic sample document (repeated to make it longer).
_PARAGRAPH = (
    "Artificial intelligence is rapidly transforming business process services "
    "across industries. Organizations are adopting large language models to "
    "automate document-heavy workflows such as contract review and claims "
    "processing. Modular asynchronous pipelines process thousands of documents "
    "concurrently while maintaining low latency. Deploying at enterprise scale "
    "requires attention to cost, inference optimization, and reliability."
)
SAMPLE = " ".join([_PARAGRAPH] * 3)


def benchmark_mode(label: str, quantize: bool) -> dict:
    """Run one mode and return timing + compression metrics."""
    settings = Settings(quantize=quantize)

    start_load = time.perf_counter()
    summarizer = SummarizerFactory.create(settings)
    load_time = time.perf_counter() - start_load

    start_infer = time.perf_counter()
    summary = summarizer.summarize(SAMPLE)
    infer_time = time.perf_counter() - start_infer

    return {
        "label": label,
        "load_time": round(load_time, 2),
        "infer_time": round(infer_time, 2),
        "compression": round(len(summary) / len(SAMPLE), 3),
    }


def print_report(results: list) -> None:
    """Print a comparison table of all benchmark runs."""
    print("\n" + "=" * 60)
    print("  PERFORMANCE vs. COST BENCHMARK")
    print("=" * 60)
    print(f"{'Mode':<20}{'Load (s)':<12}{'Infer (s)':<12}{'Compression':<12}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['label']:<20}"
            f"{r['load_time']:<12}"
            f"{r['infer_time']:<12}"
            f"{r['compression']:<12}"
        )
    print("=" * 60)

    if len(results) == 2:
        delta = results[0]["infer_time"] - results[1]["infer_time"]
        faster = "quantized" if delta > 0 else "standard"
        print(f"\nInference: {faster} was faster by {abs(delta):.2f}s.")


def main() -> None:
    """Run both modes and print the comparison report."""
    results = [
        benchmark_mode("STANDARD (FP32)", quantize=False),
        benchmark_mode("QUANTIZED (INT8)", quantize=True),
    ]
    print_report(results)


if __name__ == "__main__":
    main()