"""Falcon-H1 QLoRA construction - one dtype decision, no runtime casting.

Two Falcon-H1 specifics live here and must not be removed:
  * ``llm_int8_skip_modules=["out_proj"]`` - the Mamba kernel consumes that weight
    directly and cannot take a 4-bit packed tensor (TII fine-tuning guide).
  * LoRA never targets ``out_proj`` / ``conv1d`` - they bypass the LoRA forward.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import Config
from .errors import DependencyError

log = logging.getLogger(__name__)


def amp_dtype(cfg: Config) -> torch.dtype:
    """bf16 where it is real, fp16 otherwise. ``including_emulation=False`` is what makes
    this safe on a T4, where ``is_bf16_supported()`` would otherwise return True."""
    if cfg.model.dtype == "bf16":
        return torch.bfloat16
    if cfg.model.dtype == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def probe_trainable_dtypes(model: torch.nn.Module, dtype: torch.dtype) -> dict[str, Any]:
    """Inspect trainable tensors without raising. Returns a summary dict.

    torch >= 2.11 tracks a per-tensor ``grad_dtype``. PEFT's fp16/bf16->fp32
    upcast changes ``dtype`` without updating ``grad_dtype``, so gradients can
    arrive in a dtype the GradScaler and bitsandbytes' optimizer reject.
    This reports the state so a notebook can print it instead of guessing.
    """
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    mismatched = [
        name for name, p in trainable
        if p.dtype != dtype or getattr(p, "grad_dtype", p.dtype) != dtype
    ]
    return {
        "expected": str(dtype),
        "trainable_count": len(trainable),
        "trainable_params": sum(p.numel() for _, p in trainable),
        "mismatched_count": len(mismatched),
        "mismatched": mismatched[:20],
    }


def assert_trainable_dtypes(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Refuse to train when a trainable tensor (or its ``grad_dtype``) is off-dtype.

    Raises :class:`DependencyError` listing the offending parameter names.
    Callers that only want the measurement should use
    :func:`probe_trainable_dtypes`.
    """
    report = probe_trainable_dtypes(model, dtype)
    if report["mismatched_count"]:
        raise DependencyError(
            f"{report['mismatched_count']} trainable tensor(s) are not {report['expected']}: "
            f"{report['mismatched']}. This is the PEFT fp16/bf16->fp32 upcast vs "
            "torch>=2.11 grad_dtype mismatch; refusing to start instead of "
            "diverging mid-run."
        )


def build_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.id)
    if tokenizer.chat_template is None:
        raise RuntimeError(f"{cfg.model.id} ships no chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base(cfg: Config, dtype: torch.dtype) -> PreTrainedModel:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
        llm_int8_skip_modules=list(cfg.model.bnb_skip_modules),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.id,
        quantization_config=bnb,
        dtype=dtype,
        device_map={"": 0},
        attn_implementation=cfg.model.attn,
    )
    model.config.use_cache = False
    return model


def build_model(cfg: Config) -> PeftModel:
    """4-bit base + fresh LoRA adapters, all in one dtype."""
    base = prepare_model_for_kbit_training(
        _load_base(cfg, amp_dtype(cfg)),
        use_gradient_checkpointing=cfg.train.gradient_checkpointing,
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("trainable parameters: %s", f"{trainable:,}")
    report = probe_trainable_dtypes(model, amp_dtype(cfg))
    log.info(
        "dtype contract: expected=%s trainable=%d mismatched=%d",
        report["expected"], report["trainable_count"], report["mismatched_count"],
    )
    assert_trainable_dtypes(model, amp_dtype(cfg))
    return model


def attach_adapter(cfg: Config, adapter_dir: str | os.PathLike) -> PeftModel:
    """Trained adapter on a fresh 4-bit base, for evaluation and export."""
    base = _load_base(cfg, amp_dtype(cfg))
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    return model
