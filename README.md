# Document Intelligence Service

A production-grade Python service that ingests documents (PDF, TXT, DOCX),
generates concise summaries using Hugging Face models, and answers questions
about them using a local LLM with Retrieval-Augmented Generation (RAG).

## Features

- Multi-format ingestion (PDF, TXT, DOCX) via the Factory pattern
- Abstractive summarization using facebook/bart-large-cnn
- Document Q&A using RAG with a local Ollama LLM (llama3.2)
- Async concurrent processing with a semaphore and thread-safe model access
- Structured JSON logging with per-document correlation IDs
- Automatic chunk-size validation against model token limits
- Optional INT8 quantization for optimized CPU inference
- Three interfaces: CLI, FastAPI REST API, and Streamlit web UI
- Full unit test suite and Docker deployment

## Requirements

- Python 3.11
- uv (dependency manager)
- Ollama (for Q&A generative LLM)
- Docker (optional)

## Installation

    uv sync
    ollama pull llama3.2

## Configuration (.env)

| Variable          | Default                 | Description                     |
|-------------------|-------------------------|---------------------------------|
| MODEL_NAME        | facebook/bart-large-cnn | Summarization model             |
| MAX_CONCURRENCY   | 4                       | Max documents processed at once |
| CHUNK_SIZE        | 4000                    | Max characters per chunk        |
| MAX_OUTPUT_TOKENS | 256                     | Max summary length in tokens    |
| LOG_LEVEL         | INFO                    | Logging level                   |
| QUANTIZE          | false                   | Enable INT8 quantization        |

## Usage: CLI

    uv run doc-summarizer path/to/document.txt
    uv run doc-summarizer docs --format json
    uv run doc-summarizer docs --output summaries --quiet

## Usage: REST API

    uv run uvicorn doc_summarizer.api:app --port 8000

Endpoints: GET /health, GET /docs, POST /summarize-one, POST /summarize

## Usage: Web UI (Streamlit)

    uv run streamlit run streamlit_app.py

Open [http://localhost:8501%20to%20upload%20documents,%20view%20summaries,%20and%20ask](http://localhost:8501%20to%20upload%20documents,%20view%20summaries,%20and%20ask)
questions. Ensure Ollama is running for Q&A.

## Document Q&A (RAG)

The Q&A agent embeds the question, retrieves relevant chunks from a FAISS
vector index, and uses a local Ollama LLM (llama3.2) to generate grounded
answers with self-rated confidence.

## Quantization

    QUANTIZE=true uv run doc-summarizer docs/article.txt

INT8 quantization reduces memory (~4x) and can speed up CPU inference. QNNPACK
(ARM) or FBGEMM (x86) backend is selected automatically.

## Optimization Findings

| Mode             | Load (s) | Inference (s) | Compression |
|------------------|----------|---------------|-------------|
| Standard (FP32)  | 4.85     | 4.90          | 0.234       |
| Quantized (INT8) | 6.11     | 4.05          | 0.138       |

## Testing

    uv run pytest -v

## Docker Deployment

    docker build -t doc-summarizer .
    docker run -p 8000:8000 doc-summarizer

Access at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Design Patterns

- Factory: LoaderFactory selects the loader by file extension.
- Strategy: SummarizerStrategy allows swapping implementations.
- Singleton (cached): get_settings() returns a cached Settings instance.
- RAG: retrieval-augmented generation for Q&A.

## Project Structure

    doc_summarizer/
    ├── Dockerfile
    ├── pyproject.toml
    ├── README.md
    ├── benchmark.py
    ├── streamlit_app.py
    ├── src/doc_summarizer/
    │   ├── cli.py
    │   ├── api.py
    │   ├── config/
    │   ├── core/
    │   ├── ingestion/
    │   ├── summarization/
    │   └── agent/
    └── tests/

## License

MIT