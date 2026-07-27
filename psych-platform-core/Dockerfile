# syntax=docker/dockerfile:1
FROM python:3.11-slim

# libgomp1 — required by PyTorch/sentence-transformers (cross-encoder reranker
# + BGE embeddings) at runtime for OpenMP-based CPU inference.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dir lives under /app so it survives the later chown to a non-root user.
ENV HF_HOME=/app/.cache/huggingface \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding + reranker models at build time. Without this,
# the first RAG request after every container start pays a ~15-30s cold-start
# tax downloading ~500MB from HuggingFace Hub — and fails outright if the
# runtime environment has no outbound internet access (common in locked-down
# prod VPCs).
RUN python -c "\
from langchain_huggingface import HuggingFaceEmbeddings; \
HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5', model_kwargs={'device': 'cpu'}); \
from sentence_transformers import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app ./app
COPY scripts ./scripts
COPY migrations ./migrations
COPY alembic.ini ./
COPY start_server.py ./

RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/health || exit 1

# One worker per container — the in-process ML models (BGE embeddings +
# cross-encoder) are loaded once per worker process. Scale horizontally via
# replica count (the orchestrator), not --workers, to keep per-container
# memory predictable. See DEPLOYMENT_PLAN.md.
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
