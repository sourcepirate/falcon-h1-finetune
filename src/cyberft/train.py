"""Train the LoRA adapter with the transformers Trainer. No subclasses, no casts."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, cast

from datasets import DatasetDict
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from .config import Config, warmup_steps
from .errors import ConfigError, NonFiniteLossError, NotBuiltError
from .gate import assert_generations
from .hub import push as push_model
from .loaders import prepare_data
from .model import build_model, build_tokenizer, resolve_device
from .tracking import NullTracker, WandbTracker, clean_metrics, final_summary, metric_name, start

log = logging.getLogger(__name__)

# Smoke mode: small enough to finish in ~2 minutes, large enough to exercise the
# collator, the label path and a few optimizer steps.
SMOKE_TRAIN_ROWS = 64
SMOKE_EVAL_ROWS = 8
SMOKE_STEPS = 5


class FiniteLossCallback(TrainerCallback):
    """Abort training on the first non-finite loss instead of saving a poisoned adapter."""

    def on_log(self, args: Any, state: Any, control: Any,
               logs: dict[str, float] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        for key in ("loss", "eval_loss"):
            value = logs.get(key)
            if value is not None and not math.isfinite(value):
                raise NonFiniteLossError(
                    f"non-finite {key}={value} at step {state.global_step}; "
                    "stopping instead of saving a poisoned adapter. "
                    "Lower train.lr or shorten data.max_len."
                )


class MetricsCallback(TrainerCallback):
    """Mirror Trainer logs to wandb/Goggle with goggle-style metric names.

    Registered *after* :class:`FiniteLossCallback` so a non-finite loss
    aborts before it is ever logged. Post-init tracker failures only warn
    (see :mod:`cyberft.tracking`) and never abort training.
    """

    def __init__(self, tracker: NullTracker | WandbTracker, total_steps: int | None = None) -> None:
        self._tracker = tracker
        self._total_steps = total_steps

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._total_steps is not None:
            self._tracker.summary({"_total_steps": self._total_steps})

    def on_log(self, args: Any, state: Any, control: Any,
               logs: dict[str, float] | None = None, **kwargs: Any) -> None:
        if not logs:
            return
        cleaned = clean_metrics({metric_name(key): value for key, value in logs.items()})
        if cleaned:
            self._tracker.log(cleaned, step=state.global_step)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        history = list(getattr(state, "log_history", None) or [])
        self._tracker.summary(
            final_summary(history, total_steps=self._total_steps, global_step=state.global_step)
        )


class MpsCacheCallback(TrainerCallback):
    """Release cached MPS blocks during long local runs.

    The Metal allocator keeps every freed block in its cache. With varying
    sequence lengths the cache fragments and the process grows well past
    physical RAM (observed: 37 GB for a 0.5B fp32 model on a 32 GB M2 Max),
    at which point macOS swaps every step and throughput collapses from 4.5 s
    to over 90 s per optimizer step. Emptying the cache every few steps keeps
    the working set bounded; it changes no training math.
    """

    def __init__(self, every: int = 10, log_every: int = 100) -> None:
        self.every = every
        self.log_every = log_every

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        import torch

        if not torch.backends.mps.is_available():
            return
        if self.every > 0 and state.global_step % self.every == 0:
            torch.mps.empty_cache()
        if self.log_every > 0 and state.global_step % self.log_every == 0:
            current = torch.mps.current_allocated_memory() / 1e9
            driver = torch.mps.driver_allocated_memory() / 1e9
            log.info(
                "mps memory at step %d: %.2f GB live, %.2f GB driver",
                state.global_step, current, driver,
            )


def expected_total_steps(cfg: Config, train_rows: int, smoke: bool) -> int:
    """Planned optimizer steps, for the Goggle progress bar.

    ``train.max_steps`` is ``-1`` for full-epoch runs, so the image contract
    leaves ``total_steps_path`` blank and the UI reads ``_total_steps``
    from the run summary instead.
    """
    if smoke:
        return SMOKE_STEPS
    if cfg.train.max_steps > 0:
        return cfg.train.max_steps
    per_epoch = max(1, math.ceil(train_rows / (cfg.train.micro_batch * cfg.train.grad_accum)))
    return max(1, int(per_epoch * cfg.train.epochs))


def _run_config(cfg: Config, smoke: bool) -> dict[str, Any]:
    """Flat JSON-primitive run config for ``wandb.init(config=...)``."""
    return {
        "model": cfg.model.id,
        "dtype": cfg.model.dtype,
        "quant": cfg.model.quant,
        "attn": cfg.model.attn,
        "data/max_len": cfg.data.max_len,
        "data/eval_max_len": cfg.data.eval_max_len,
        "data/eval_size": cfg.data.eval_size,
        "seed": cfg.data.seed,
        "lora/r": cfg.lora.r,
        "lora/alpha": cfg.lora.alpha,
        "lora/dropout": cfg.lora.dropout,
        "train/lr": cfg.train.lr,
        "train/epochs": cfg.train.epochs,
        "train/max_steps": cfg.train.max_steps,
        "train/micro_batch": cfg.train.micro_batch,
        "train/grad_accum": cfg.train.grad_accum,
        "smoke": smoke,
    }


def train_dir(cfg: Config, smoke: bool = False) -> Path:
    """Where the adapter, tokenizer and checkpoints for this run live."""
    return Path(cfg.train.output_dir) / ("smoke" if smoke else "")


def load_splits(cfg: Config, smoke: bool) -> DatasetDict:
    # prepare_data() is cached, so this builds the corpus once and reuses it afterwards.
    splits = prepare_data(cfg, smoke=smoke)
    return splits.remove_columns(["text"]) if "text" in splits["train"].column_names else splits


def build_args(cfg: Config, train_rows: int, smoke: bool) -> TrainingArguments:
    if smoke:
        max_steps, epochs = SMOKE_STEPS, 1.0
    elif cfg.train.max_steps > 0:
        max_steps, epochs = cfg.train.max_steps, cfg.train.epochs
    else:
        max_steps, epochs = -1, cfg.train.epochs
    return TrainingArguments(
        output_dir=str(train_dir(cfg, smoke)),
        per_device_train_batch_size=cfg.train.micro_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.train.grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=cfg.train.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps(cfg, train_rows),
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        # Single precision end-to-end: the model is loaded in one dtype and no
        # AMP/GradScaler is used (notebook recipe: fp16 load, fp16=False, bf16=False).
        bf16=False,
        fp16=False,
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


def run_generation_gate(cfg: Config, trainer: Trainer, tokenizer: Any) -> list[str]:
    """Greedy generations on the gate prompts; raises on empty/repetitive output.

    Runs on the in-memory adapter so it costs seconds, not a reload. A finite
    loss does not catch mode collapse; this does, before anything is pushed.
    """
    import torch  # local: cyberft.train already needs transformers, but keep the import tight

    model = trainer.model
    if model is None:
        raise NotBuiltError("trainer.model is None; cannot run the generation gate")
    # Trainer.model is typed PreTrainedModel | Module | None; both real branches
    # here are a causal LM, so treat it as Any instead of narrowing twice.
    model = cast(Any, model)
    was_training = model.training
    model.eval()
    answers: list[str] = []
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.train.gate_max_new_tokens,
        "do_sample": False,
    }
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    try:
        with torch.no_grad():
            for prompt in cfg.train.gate_prompts:
                inputs = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, return_tensors="pt", return_dict=True,
                )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                output = model.generate(**inputs, **generate_kwargs)
                answers.append(
                    tokenizer.decode(
                        output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
                    ).strip()
                )
    finally:
        if was_training:
            model.train()
    assert_generations(list(cfg.train.gate_prompts), answers)
    return answers


def train(cfg: Config, smoke: bool = False, *, wandb: bool | None = None,
         push: bool | None = None) -> Trainer:
    if smoke and push:
        raise ConfigError("--push with --smoke is refused: smoke trains on a 64-row slice")
    # Auto-push when the YAML names a Hub repo (never in smoke); explicit
    # --push/--no-push wins. train is still the only writer: `cyberft push`
    # reuses the same merge+upload for already-trained adapters.
    do_push = push if push is not None else (cfg.export.hf_repo is not None and not smoke)
    if do_push and not cfg.export.hf_repo:
        raise ConfigError("--push given but export.hf_repo is not set")
    tokenizer = build_tokenizer(cfg)
    splits = load_splits(cfg, smoke)
    if smoke:
        splits = DatasetDict({
            "train": splits["train"].select(range(min(SMOKE_TRAIN_ROWS, len(splits["train"])))),
            "eval": splits["eval"].select(range(min(SMOKE_EVAL_ROWS, len(splits["eval"])))),
        })

    model = build_model(cfg)
    args = build_args(cfg, len(splits["train"]), smoke)
    tracker = start(
        cfg.wandb,
        smoke=smoke,
        override=wandb,
        process_index=args.process_index,
        run_config=_run_config(cfg, smoke),
    )
    tracker.summary({"data/train_rows": len(splits["train"]), "data/eval_rows": len(splits["eval"])})
    total_steps = expected_total_steps(cfg, len(splits["train"]), smoke)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=splits["train"],
        eval_dataset=splits["eval"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[FiniteLossCallback(), MetricsCallback(tracker, total_steps)]
        + ([MpsCacheCallback()] if resolve_device().type == "mps" else []),
    )

    resume = None
    if cfg.train.resume and not smoke:
        try:
            resume = get_last_checkpoint(args.output_dir)
        except (ValueError, FileNotFoundError):
            resume = None
        if resume:
            log.info("resuming from %s", resume)

    # Only a run that reaches the adapter save counts as success: wandb.finish
    # maps exit_code 0 to `completed`, anything else to failed. Push runs
    # before the success mark, so a failed upload fails the run instead of
    # reporting `completed` for a model that never reached the Hub. The finally
    # block reports, it never masks the training outcome.
    exit_code = 1
    try:
        trainer.train(resume_from_checkpoint=resume)

        out_dir = train_dir(cfg, smoke)
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        log.info("adapter + tokenizer saved to %s", out_dir)
        if cfg.train.generation_gate and not smoke:
            log.info("running post-train generation gate on %d prompts", len(cfg.train.gate_prompts))
            run_generation_gate(cfg, trainer, tokenizer)
        if do_push:
            push_model(cfg)
        exit_code = 0
        return trainer
    finally:
        tracker.finish(exit_code=exit_code)
