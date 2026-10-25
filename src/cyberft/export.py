"""Merge the adapter and produce a quantized GGUF.

Every step asserts its output exists: shells routinely ignore non-zero exit codes, which
is how you end up publishing a directory that contains only the F16 intermediate.

The merge runs on CPU in bf16 and needs roughly 7 GB of host RAM, so it works on any
machine that can hold the full-precision weights.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from .config import Config
from .errors import DependencyError, NotBuiltError

log = logging.getLogger(__name__)


def merge(cfg: Config, adapter_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (adapter_dir / "adapter_config.json").exists():
        raise NotBuiltError(f"no adapter at {adapter_dir}")

    merged = Path(cfg.export.merged_dir)
    log.info("loading %s in bf16 (CPU) for merge", cfg.model.id)
    base = AutoModelForCausalLM.from_pretrained(cfg.model.id, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()

    merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(cfg.model.id).save_pretrained(merged)
    log.info("merged model at %s", merged)
    return merged


def convert(cfg: Config, merged: Path) -> Path:
    script = Path(cfg.export.llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not script.exists():
        raise DependencyError(f"{script} not found: clone llama.cpp into {cfg.export.llama_cpp_dir}")

    gguf_dir = Path(cfg.export.gguf_dir)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    out = gguf_dir / "model-f16.gguf"
    subprocess.run(
        [sys.executable, str(script), str(merged), "--outfile", str(out), "--outtype", "f16"],
        check=True,
    )
    if not out.exists():
        raise RuntimeError(f"conversion produced no file at {out}")
    return out


def quantize(cfg: Config, f16: Path) -> Path:
    binary = _find_binary(cfg, "llama-quantize")
    out = Path(cfg.export.gguf_dir) / f"model-{cfg.export.quant.lower()}.gguf"
    subprocess.run([str(binary), str(f16), str(out), cfg.export.quant], check=True)
    if not out.exists():
        raise RuntimeError(f"quantization produced no file at {out} (requested {cfg.export.quant})")
    log.info("quantized GGUF: %s (%.2f GB)", out, out.stat().st_size / 1e9)
    return out


def export_gguf(cfg: Config, *, adapter: str | Path | None = None, skip_merge: bool = False) -> Path:
    """Merge (unless cached) and return the quantized GGUF path."""
    adapter_dir = Path(adapter) if adapter else Path(cfg.train.output_dir)
    if skip_merge:
        merged = Path(cfg.export.merged_dir)
        if not merged.exists():
            raise NotBuiltError(f"--skip-merge given but {merged} does not exist")
    else:
        merged = merge(cfg, adapter_dir)
    return quantize(cfg, convert(cfg, merged))


def _find_binary(cfg: Config, name: str) -> Path:
    root = Path(cfg.export.llama_cpp_dir)
    for candidate in (root / "build" / "bin" / name, root / name):
        if candidate.exists():
            return candidate
    found = next(root.rglob(name), None)
    if found is None:
        raise DependencyError(f"{name} not found under {root}: build llama.cpp first")
    return found
