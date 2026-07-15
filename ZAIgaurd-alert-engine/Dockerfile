# ============================================================
# ZaiGuard Alert Engine — Production Dockerfile
# Base: Python 3.10 Slim Linux
# ============================================================

FROM python:3.10-slim as builder

# Set Python build and runtime behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS build requirements (required if compiling wheels like asyncpg or torch dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the sentence-transformers embedding model (all-MiniLM-L6-v2)
# During docker build, this saves the ~90MB weights directly into the container image
# so startup is instantaneous and does not depend on HuggingFace network reachability.
ENV EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2" \
    EMBEDDING_CACHE_DIR="/app/.model_cache"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/.model_cache')"

# Copy application source code
COPY . .

# Expose HTTP port for uvicorn
EXPOSE 8000

# Healthcheck against the FastAPI /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the asynchronous ASGI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
