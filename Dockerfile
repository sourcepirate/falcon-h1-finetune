# One CUDA image for T4 smoke and RTX 3090 full trains (sm_75 + sm_86).
# Base carries CUDA 12.6 toolkit; torch 2.11.0 cu126 wheels bundle the CUDA runtime.
# Host driver must support CUDA 12.6.
#
# llama.cpp ships inside the image because `push` publishes a GGUF: the
# converter is pure Python (needs gguf-py + sentencepiece), while k-quants use
# the compiled llama-quantize copied onto PATH. Q8_0/f16/bf16/f32 are written
# by the converter directly, so Q8_0 pushes need no binary at all.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    WANDB_DISABLE_CODE=true \
    GOGGLE_CONFIG_PATH=/config/job.yaml \
    CONFIG=configs/qlora-tiny.yaml

WORKDIR /work

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.11 python3.11-venv make git ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python3.11 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

COPY pyproject.toml README.md Makefile requirements-fast.txt image.contract.json ./
COPY src/ src/
COPY configs/ configs/
COPY docker/ docker/

# cu126 first so the torch CUDA build is explicit, not pip-default luck;
# PyPI fallback serves everything the torch index does not host.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple ".[train]"
# Fused-kernel path; failure falls back to the pure-torch Mamba path
# (documented T4 behavior) and is recorded in the build log.
RUN pip install --no-cache-dir -r requirements-fast.txt --no-build-isolation \
 || echo "WARNING: fast kernels not installed; using pure-torch Mamba fallback"
# Conversion deps only: llama.cpp's requirements file pins transformers==4.57.6,
# which would downgrade the 5.20.2 runtime, so it is deliberately not used.
RUN pip install --no-cache-dir "sentencepiece>=0.1.98,<0.3.0" "protobuf>=4.21,<5"


FROM runtime AS llama-builder

# Pin the same revision as configs/*.yaml export.llama_cpp_rev.
ARG LLAMA_CPP_REV=9fba51d2c9a1b5cb5f0c47986c031334e2d7663a

RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake build-essential \
 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && git -C /opt/llama.cpp checkout "${LLAMA_CPP_REV}" \
 && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
      -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /opt/llama.cpp/build --target llama-quantize -j"$(nproc)"


FROM runtime AS final

ARG LLAMA_CPP_REV=9fba51d2c9a1b5cb5f0c47986c031334e2d7663a

# Keep only what conversion needs; the build tree and compiler stay behind.
COPY --from=llama-builder /opt/llama.cpp/convert_hf_to_gguf.py /work/vendor/llama.cpp/
COPY --from=llama-builder /opt/llama.cpp/conversion /work/vendor/llama.cpp/conversion
COPY --from=llama-builder /opt/llama.cpp/gguf-py /work/vendor/llama.cpp/gguf-py
COPY --from=llama-builder /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize
RUN chmod +x /usr/local/bin/llama-quantize \
 && test -x /usr/local/bin/llama-quantize \
 && cd /work/vendor/llama.cpp \
 && python -c "from conversion import TEXT_MODEL_MAP; assert 'FalconH1ForCausalLM' in TEXT_MODEL_MAP, 'converter lacks Falcon-H1'" \
 && echo "llama.cpp converter + quantizer ready" 

# Smoke:  docker run --gpus all <image> train -c configs/qlora-tiny.yaml --smoke
# Full:    docker run --gpus all <image> train -c configs/qlora-tiny.yaml --wandb
#          (trains, reports to Goggle, merges, writes the model card + GGUF, uploads)
# Export:  docker run --gpus all <image> export -c configs/qlora-tiny.yaml
# Repair:  docker run --gpus all -e HF_TOKEN <image> push -c configs/qlora-tiny.yaml --from-hub
# RunPod:  pull ghcr.io/sourcepirate/falcon-h1-tiny:tiny, same CLI commands.
# Goggle:  register image.contract.json, submit a run; the contract command already
#          passes --wandb so a mis-wired job fails at init instead of completing blind.
ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["train", "-c", "configs/qlora-tiny.yaml", "--smoke"]
