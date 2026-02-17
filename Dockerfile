# Multi-stage build for Contextual RAG Pipeline
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Install Python dependencies
RUN uv pip install -r requirements.txt

# ============================================
# Final production image
# ============================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies for ChromaDB and Ollama client
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY README.md .
COPY Makefile .

# Set environment variables
# PYTHONUNBUFFERED: Force stdout/stderr to be unbuffered (real-time logs in Docker)
# PYTHONDONTWRITEBYTECODE: Don't create .pyc files (keeps container clean)
# PATH: Use virtual environment Python binaries
# PYTHONPATH: Add /app to Python import path
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}"

# Create directories for persistent storage
RUN mkdir -p chroma_db cache_db logs benchmarks

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Start FastAPI server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
