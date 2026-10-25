"""Train the LoRA adapter with the transformers Trainer. No subclasses, no casts."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from datasets import DatasetDict
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from .config import Config, warmup_steps
from .data import prepare_data
from .model import amp_dtype, build_model, build_tokenizer

log = logging.getLogger(__name__)

# Smoke mode: small enough to finish in ~2 minutes, large enough to exercise the
# collator, the label path and a few optimizer steps.
SMOKE_TRAIN_ROWS = 64
SMOKE_EVAL_ROWS = 8
SMOKE_STEPS = 5


def train_dir(cfg: Config, smoke: bool = False) -> Path:
    """Where the adapter, tokenizer and checkpoints for this run live."""
    return Path(cfg.train.output_dir) / ("smoke" if smoke else "")


def load_splits(cfg: Config, smoke: bool) -> DatasetDict:
    # prepare_data() is cached, so this builds the corpus once and reuses it afterwards.
    splits = prepare_data(cfg, smoke=smoke)
    return splits.remove_columns(["text"]) if "text" in splits["train"].column_names else splits


def build_args(cfg: Config, train_rows: int, smoke: bool) -> TrainingArguments:
    use_bf16 = amp_dtype(cfg) is torch.bfloat16
    return TrainingArguments(
        output_dir=str(train_dir(cfg, smoke)),
        per_device_train_batch_size=cfg.train.micro_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.train.grad_accum,
        num_train_epochs=1.0 if smoke else cfg.train.epochs,
        max_steps=SMOKE_STEPS if smoke else -1,
        learning_rate=cfg.train.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps(cfg, train_rows),
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if smoke else cfg.train.logging_steps,
        save_strategy="no" if smoke else "steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        eval_strategy="no" if smoke else "steps",
        eval_steps=None if smoke else cfg.train.eval_steps,
        report_to="none" if smoke else (list(cfg.train.report_to) or "none"),
        remove_unused_columns=True,
        dataloader_num_workers=cfg.train.num_workers,
        seed=cfg.train.seed,
    )


def train(cfg: Config, smoke: bool = False) -> Trainer:
    tokenizer = build_tokenizer(cfg)
    splits = load_splits(cfg, smoke)
    if smoke:
        splits = DatasetDict({
            "train": splits["train"].select(range(min(SMOKE_TRAIN_ROWS, len(splits["train"])))),
            "eval": splits["eval"].select(range(min(SMOKE_EVAL_ROWS, len(splits["eval"])))),
        })

    model = build_model(cfg)
    args = build_args(cfg, len(splits["train"]), smoke)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["eval"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    resume = None
    if cfg.train.resume and not smoke:
        from transformers.trainer_utils import get_last_checkpoint

        try:
            resume = get_last_checkpoint(args.output_dir)
        except (ValueError, FileNotFoundError):
            resume = None
        if resume:
            log.info("resuming from %s", resume)

    trainer.train(resume_from_checkpoint=resume)

    out_dir = train_dir(cfg, smoke)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    log.info("adapter + tokenizer saved to %s", out_dir)
    return trainer
