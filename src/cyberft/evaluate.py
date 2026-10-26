"""Per-sample evaluation.

A mean ``eval_loss`` hides which sample broke. This walks the split one row at a time and
returns a report, so a NaN comes with an address instead of a shrug.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import DataCollatorForLanguageModeling

from .config import Config
from .errors import NonFiniteLossError, NotBuiltError
from .loaders import prepare_data
from .model import attach_adapter, build_tokenizer
from .train import train_dir


@dataclass
class RowLoss:
    index: int
    tokens: int
    loss: float
    text: str


@dataclass
class EvalReport:
    split: str
    rows: list[RowLoss] = field(default_factory=list)

    @property
    def finite(self) -> list[RowLoss]:
        return [r for r in self.rows if math.isfinite(r.loss)]

    @property
    def broken(self) -> list[RowLoss]:
        return [r for r in self.rows if not math.isfinite(r.loss)]

    @property
    def mean(self) -> float:
        values = [r.loss for r in self.finite]
        return statistics.fmean(values) if values else float("nan")

    @property
    def median(self) -> float:
        values = [r.loss for r in self.finite]
        return statistics.median(values) if values else float("nan")

    def worst(self, n: int = 3) -> list[RowLoss]:
        return sorted(self.finite, key=lambda r: r.loss, reverse=True)[:n]

    def summary(self) -> str:
        if not self.rows:
            return f"{self.split}: no rows"
        lines = [
            f"{self.split}: rows={len(self.rows)} finite={len(self.finite)} "
            f"mean={self.mean:.4f} p50={self.median:.4f}"
        ]
        for row in self.worst():
            lines.append(
                f"  worst: row={row.index} tokens={row.tokens} loss={row.loss:.4f} {row.text[:80]!r}"
            )
        for row in self.broken:
            lines.append(f"  non-finite: row={row.index} tokens={row.tokens} {row.text[:80]!r}")
        return "\n".join(lines)

    def assert_finite(self) -> EvalReport:
        if self.broken:
            raise NonFiniteLossError(
                f"{len(self.broken)} row(s) produced a non-finite loss: "
                f"{[r.index for r in self.broken[:10]]}"
            )
        return self


def _load_for_inference(cfg: Config, adapter: str | Path | None, smoke: bool):
    tokenizer = build_tokenizer(cfg)
    adapter_dir = Path(adapter) if adapter else train_dir(cfg, smoke)
    if not (adapter_dir / "adapter_config.json").exists():
        raise NotBuiltError(f"no adapter at {adapter_dir} - train first")
    model = attach_adapter(cfg, adapter_dir)
    return model, tokenizer, next(model.parameters()).device


def evaluate(cfg: Config, *, adapter: str | Path | None = None, split: str = "eval",
             smoke: bool = False) -> EvalReport:
    # Single precision, no autocast: matches the no-AMP training path.
    model, tokenizer, device = _load_for_inference(cfg, adapter, smoke)
    dataset = prepare_data(cfg, smoke=smoke)[split]
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    report = EvalReport(split=split)
    with torch.no_grad():
        for index in range(len(dataset)):
            row = dataset[index]
            features = {k: v for k, v in row.items() if k != "text"}
            batch = {k: v.to(device) for k, v in collator([features]).items()}
            loss = model(**batch).loss.item()
            report.rows.append(
                RowLoss(index, len(row["input_ids"]), loss, row.get("text", "")[:200])
            )
    return report


def generate(cfg: Config, prompts: list[str], *, adapter: str | Path | None = None,
             smoke: bool = False, max_new_tokens: int = 256) -> list[str]:
    model, tokenizer, device = _load_for_inference(cfg, adapter, smoke)
    answers: list[str] = []
    for prompt in prompts:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        answers.append(
            tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        )
    return answers
