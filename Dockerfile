# One CUDA image for T4 smoke and RTX 3090 full trains (sm_75 + sm_86).
# Base carries CUDA 12.6 toolkit; torch 2.11.0 cu126 wheels (verified on the
# wheel index; confirm-or-repin on first build) bundle the CUDA runtime.
# Host driver must support CUDA 12.6.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    CONFIG=configs/qlora-tiny.yaml

WORKDIR /work

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.11 python3.11-venv make git \
 && rm -rf /var/lib/apt/lists/* \
 && python3.11 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

COPY pyproject.toml README.md Makefile requirements-fast.txt ./
COPY src/ src/
COPY configs/ configs/

# cu126 first so the torch CUDA build is explicit, not pip-default luck;
# PyPI fallback serves everything the torch index does not host.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple ".[train]"
# Fused-kernel path; failure falls back to the pure-torch Mamba path
# (documented T4 behavior) and is recorded in the build log.
RUN pip install --no-cache-dir -r requirements-fast.txt --no-build-isolation \
 || echo "WARNING: fast kernels not installed; using pure-torch Mamba fallback"

# Smoke:  docker run --gpus all <image> python -m cyberft.cli -c configs/qlora-tiny.yaml train --smoke
# Full:    docker run --gpus all <image> python -m cyberft.cli -c configs/qlora-tiny.yaml data \
#            && python -m cyberft.cli -c configs/qlora-tiny.yaml train \
#            && python -m cyberft.cli -c configs/qlora-tiny.yaml eval
# Export:  docker run --gpus all <image> python -m cyberft.cli -c configs/qlora-tiny.yaml export
# RunPod:  pull ghcr.io/sourcepirate/falcon-h1-tiny:tiny, same CLI commands.
CMD ["python", "-m", "cyberft.cli", "train", "-c", "configs/qlora-tiny.yaml", "--smoke"]
