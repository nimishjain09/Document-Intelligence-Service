# Use a slim Python 3.11 base image for a smaller footprint
FROM python:3.11-slim

# Install system build dependencies (needed by some ML wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast dependency manager)
RUN pip install --no-cache-dir uv

# Set the working directory
WORKDIR /app

# Copy dependency manifests first (better build caching)
COPY pyproject.toml uv.lock ./

# Copy the application source
COPY src ./src

# Install project dependencies (production only, no dev tools)
RUN uv sync --frozen --no-dev

# Pre-download the model at build time so the container starts fast
RUN uv run python -c "from transformers import pipeline; pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')"

# Expose the API port
EXPOSE 8000

# Environment defaults (override at runtime with -e)
ENV MODEL_NAME=sshleifer/distilbart-cnn-12-6 \
    MAX_CONCURRENCY=2 \
    CHUNK_SIZE=1000 \
    LOG_LEVEL=INFO \
    QUANTIZE=false

# Run the FastAPI server, binding to all interfaces
CMD ["uv", "run", "uvicorn", "doc_summarizer.api:app", "--host", "0.0.0.0", "--port", "8000"]