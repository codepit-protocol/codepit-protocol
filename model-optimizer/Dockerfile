# Modal Function image for the CodePit V2 model optimizer agent.
#
# Used by `agents/model-optimizer/modal_app.py` and consumed in production
# by `engine/src/v2/runtime/modal-worker-pool.ts`. Pins the same Python
# version as `pyproject.toml` (>=3.10) and installs the kit's `[optimize]`
# extras so the container has transformers + onnx + optimum on first run.
#
# Build is intentionally minimal: Modal layers its own runtime helpers on
# top, and the container just needs the Python kit + dependencies.

FROM python:3.11-slim

# System deps for onnxruntime, transformers tokenizers, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Copy the kit source into the image so the Modal Function can import it
# without a separate registry pull. The Modal Image build context is the
# `agents/model-optimizer/` directory (set in `modal_app.py`).
COPY pyproject.toml /app/pyproject.toml
COPY codepit_optimizer /app/codepit_optimizer

# Install the kit + its `optimize` extras (transformers, optimum, onnx*).
RUN pip install --upgrade pip \
 && pip install ".[optimize]"

# Default command runs one supervised loop. Railway managed-worker services
# override this with `python -m codepit_optimizer.managed_worker`.
CMD ["codepit-model-optimizer", "run-forever"]
