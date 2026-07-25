# ---------------------------------------------------------------
#  Library RAG Chatbot - Production Docker Image
#  GPU  : NVIDIA (CUDA 12.8 + cuDNN 9) - matches RTX 5050 env
#  Entry: chainlit run RAG/chainlit_app.py
# ---------------------------------------------------------------

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# -- System environment ---------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Required for the NVIDIA Container Runtime to expose the GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# HuggingFace model cache (Jina / Qwen models download here on first run).
# Persisted via the hf_cache named volume declared in docker-compose.yml.
ENV HF_HOME=/app/.cache/huggingface

# -- System packages ------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        curl \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -- Python dependencies --------------------------------------
# Step 1: PyTorch with CUDA 12.8 wheels (separate layer for cache efficiency)
RUN python3.11 -m pip install --no-cache-dir \
        torch==2.7.1 \
        torchvision \
        --index-url https://download.pytorch.org/whl/cu128

# Step 2: Everything else (torch filtered out to prevent CPU override)
COPY requirements.txt .
RUN grep -v "^torch" requirements.txt \
    | python3.11 -m pip install --no-cache-dir -r /dev/stdin

# -- App source code ------------------------------------------
COPY RAG/ ./RAG/

# -- Cover images ---------------------------------------------
COPY Cleaned_data/cover_images/ ./Cleaned_data/cover_images/

# -- Vector databases -----------------------------------------
# NOT baked in � mounted as named volumes via docker-compose.yml.
# Populate volumes once with init-db.ps1 before first run.

# -- Port -----------------------------------------------------
EXPOSE 8000

# -- Entrypoint -----------------------------------------------
# Pass your API keys at runtime - never bake them into the image:
#   docker compose up   (recommended)
#   docker run --gpus all --env-file .env -p 8000:8000 library-rag
CMD ["python", "-m", "chainlit", "run", "RAG/chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
