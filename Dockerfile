# POaaS Docker Image
# Minimal-Edit Prompt Optimization as a Service
#
# Build: docker build -t poaas:latest .
# Run:   docker run -p 8001:8001 -p 8002:8002 -p 8003:8003 -p 8004:8004 poaas:latest
#
# For GPU support with vLLM:
#   docker run --gpus all -p 8000:8000 -p 8001:8001 ... poaas:latest

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create directories for runs and data
RUN mkdir -p runs eval/data sample_indices results

# Download datasets (optional - can be done at runtime)
# RUN python scripts/download_datasets.py

# Expose ports
# 8000: vLLM server (if running locally)
# 8001: POaaS Orchestrator
# 8002: Cleaner Worker
# 8003: Paraphraser Worker
# 8004: Fact-Adder Worker
EXPOSE 8000 8001 8002 8003 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Default command: start all services
CMD ["python", "scripts/start_services.py"]

# Alternative: run specific service
# CMD ["python", "orchestrator/app.py", "--port", "8001"]

