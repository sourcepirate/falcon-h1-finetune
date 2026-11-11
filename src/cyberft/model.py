"""Falcon-H1 QLoRA construction - one dtype decision, no runtime casting.

Two Falcon-H1 specifics live here and must not be removed:
  * ``llm_int8_skip_modules=["out_proj"]`` - the Mamba kernel consumes that weight
    directly and cannot take a 4-bit packed tensor (TII fine-tuning guide).
  * LoRA never targets ``out_proj`` / ``conv1d`` - they bypass the LoRA forward.
"""

from __future__ import annotations

import logging
import os

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

log = logging.getLogger(__name__)

# Fallback chat template for tokenizers that ship none (verified 2026-11-11:
# tiiuae/Falcon-H1-Tiny-R-90M tokenizer_config.json has no chat_template).
# Llama-3-style headers built from tokens the Tiny tokenizer owns
# (<|begin_of_text|>, <|start_header_id|>, <|end_header_id|>, <|eot_id|>).
TINY_FALLBACK_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{%- elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{%- elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{%- endif %}"
)


def amp_dtype(cfg: Config) -> torch.dtype:
    """bf16 where it is real, fp16 otherwise. ``including_emulation=False`` is what makes
    this safe on a T4, where ``is_bf16_supported()`` would otherwise return True."""
    if cfg.model.dtype == "bf16":
        return torch.bfloat16
    if cfg.model.dtype == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def build_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.id)
    if tokenizer.chat_template is None:
        if "tiny" in cfg.model.id.lower():
            log.info("no chat template shipped; using Tiny fallback template")
            tokenizer.chat_template = TINY_FALLBACK_CHAT_TEMPLATE
        else:
            raise RuntimeError(f"{cfg.model.id} ships no chat template")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base(cfg: Config, dtype: torch.dtype) -> PreTrainedModel:
    if cfg.model.quant == "none":
        return AutoModelForCausalLM.from_pretrained(
            cfg.model.id,
            dtype=dtype,
            device_map={"": 0},
            attn_implementation=cfg.model.attn,
        )
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
    """4-bit base + fresh LoRA adapters.

    The adapters are fp32: ``prepare_model_for_kbit_training`` upcasts the base
    to fp32 and PEFT creates LoRA weights to match, casting adapter inputs on
    the fly. This is the supported configuration, not a mismatch.
    """
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
    return model


def attach_adapter(cfg: Config, adapter_dir: str | os.PathLike) -> PeftModel:
    """Trained adapter on a fresh base (quantized per model.quant), for evaluation and export."""
    if cfg.model.quant == "none":
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.id,
            dtype=amp_dtype(cfg),
            device_map={"": 0},
            attn_implementation=cfg.model.attn,
        )
    else:
        base = _load_base(cfg, amp_dtype(cfg))
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    return model
