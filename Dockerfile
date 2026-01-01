# Multi-stage Dockerfile for RAG Company Report Generator
# Stage 1: Builder
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/data /app/logs /app/reports && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Directories the app writes to. data/ is a volume in compose, so this only
# matters for a bare `docker run`, but an image that cannot start without one
# is an image with a hidden prerequisite.
RUN mkdir -p data/chroma_db data/raw_data data/hf logs reports

# Warm the local embedding model into the image. Without this the first request
# after a cold start pays for a ~90MB download, which reads as the app hanging.
ENV HF_HOME=/app/data/hf \
    SENTENCE_TRANSFORMERS_HOME=/app/data/hf
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || \
    echo "model warm-up skipped; it will download on first use"

# The API's port inside the container. The compose file publishes it on 8350
# of the host; 8000 is only ever seen from inside the network namespace, where
# nothing else is competing for it.
EXPOSE 8000

# Health check
# The old check ran `sys.exit(0)`, which reported healthy even when the app was
# down. This one actually reaches the API.
HEALTHCHECK --interval=20s --timeout=8s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

# Default command
# Serve the API by default. The previous default printed a help message and
# exited, so `docker compose up` produced a container that did nothing.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

