# =============================================================================
# REMAP-Net: Multi-stage Docker Build
# =============================================================================
# Build:  docker build -t remap-net .
# Run:    docker run --gpus all -it remap-net
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base – CUDA + Python + system dependencies
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    curl \
    git \
    build-essential \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 2: Dependencies – install Python packages
# ---------------------------------------------------------------------------
FROM base AS dependencies

WORKDIR /tmp/build

# Copy only requirements first for Docker layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support first, then remaining dependencies
RUN pip install --no-cache-dir \
    torch>=2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3: Runtime – copy project and set up workspace
# ---------------------------------------------------------------------------
FROM dependencies AS runtime

# Set working directory
WORKDIR /workspace

# Copy project files
COPY setup.py pyproject.toml requirements.txt ./
COPY remap_net/ ./remap_net/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Copy remaining project files (configs, scripts, tests)
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Expose ports for TensorBoard and MLflow
EXPOSE 6006 5000

# Default command
CMD ["python", "-c", "import remap_net; print(f'REMAP-Net v{remap_net.__version__} ready.')"]
