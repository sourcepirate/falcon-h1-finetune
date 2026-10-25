"""Typed configuration loaded from YAML. Fails fast on unknown keys or bad values."""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from .errors import ConfigError

DTYPES = ("auto", "bf16", "fp16")
ATTENTION = ("sdpa", "flash_attention_2")
# TII: these modules bypass the LoRA forward and feed a fused kernel.
FORBIDDEN_LORA_TARGETS = {"out_proj", "conv1d"}


@dataclass(frozen=True)
class SourceConfig:
    """Where one dataset comes from. How a row becomes messages stays in data.py."""

    name: str                        # must match a normaliser in data.BUILDERS
    repo: str
    splits: tuple[str, ...] = ("train",)
    subset: str | None = None        # load_dataset(name=...)
    parquet: str | None = None       # single file instead of a subset
    revision: str | None = None      # branch, tag or commit
    cap: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    id: str
    dtype: str = "auto"
    attn: str = "sdpa"
    bnb_skip_modules: tuple[str, ...] = ("out_proj",)


@dataclass(frozen=True)
class DataConfig:
    sources: tuple[SourceConfig, ...] = ()
    output_dir: Path = Path("data/processed")
    max_len: int = 2048
    eval_max_len: int = 1024
    eval_size: int = 300
    seed: int = 42


@dataclass(frozen=True)
class LoraConfig:
    r: int = 32
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj", "gate_proj", "up_proj", "down_proj",
    )


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path = Path("outputs/adapter")
    micro_batch: int = 1
    grad_accum: int = 8
    epochs: float = 1
    lr: float = 2e-4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    report_to: tuple[str, ...] = ()
    num_workers: int = 2
    seed: int = 42
    resume: bool = True


@dataclass(frozen=True)
class ExportConfig:
    merged_dir: Path = Path("outputs/merged")
    gguf_dir: Path = Path("outputs/gguf")
    quant: str = "Q4_K_M"
    llama_cpp_dir: Path = Path("vendor/llama.cpp")


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def validate(self) -> Config:
        if not self.model.id:
            raise ConfigError("model.id is required")
        if self.model.dtype not in DTYPES:
            raise ConfigError(f"model.dtype must be one of {DTYPES}, got {self.model.dtype!r}")
        if self.model.attn not in ATTENTION:
            raise ConfigError(f"model.attn must be one of {ATTENTION}, got {self.model.attn!r}")
        if "out_proj" not in self.model.bnb_skip_modules:
            raise ConfigError(
                "model.bnb_skip_modules must contain 'out_proj': Mamba's out_proj feeds a fused "
                "kernel that cannot consume a 4-bit packed weight (TII fine-tuning guide)."
            )

        bad = FORBIDDEN_LORA_TARGETS.intersection(self.lora.target_modules)
        if bad:
            raise ConfigError(f"lora.target_modules must not contain {sorted(bad)}")
        if not self.lora.target_modules:
            raise ConfigError("lora.target_modules is empty")
        if self.lora.r <= 0:
            raise ConfigError("lora.r must be positive")

        self._validate_sources()
        if self.data.max_len < 128:
            raise ConfigError("data.max_len must be >= 128")
        if not 0 < self.data.eval_max_len <= self.data.max_len:
            raise ConfigError("data.eval_max_len must be > 0 and <= data.max_len")
        if self.data.eval_size <= 0:
            raise ConfigError("data.eval_size must be positive")

        if self.train.lr <= 0 or self.train.epochs <= 0:
            raise ConfigError("train.lr and train.epochs must be positive")
        if self.train.micro_batch <= 0 or self.train.grad_accum <= 0:
            raise ConfigError("train.micro_batch and train.grad_accum must be positive")
        if not 0 <= self.train.warmup_ratio < 1:
            raise ConfigError("train.warmup_ratio must be in [0, 1)")
        for name in ("save_steps", "eval_steps", "logging_steps"):
            if getattr(self.train, name) <= 0:
                raise ConfigError(f"train.{name} must be positive")
        if self.train.num_workers < 0:
            raise ConfigError("train.num_workers must be >= 0")
        return self

    def _validate_sources(self) -> None:
        sources = self.data.sources
        if not sources:
            raise ConfigError("data.sources must not be empty")

        duplicates = [name for name, count in Counter(s.name for s in sources).items() if count > 1]
        if duplicates:
            raise ConfigError(f"data.sources has duplicate name(s): {sorted(duplicates)}")

        for source in sources:
            where = f"data.sources[{source.name}]"
            if not source.name or not source.repo:
                raise ConfigError(f"{where}: name and repo are required")
            if not source.splits:
                raise ConfigError(f"{where}: splits must not be empty")
            if source.cap is not None and source.cap <= 0:
                raise ConfigError(f"{where}: cap must be positive")
            if source.parquet and source.subset:
                raise ConfigError(f"{where}: parquet and subset are mutually exclusive")


def _coerce(hint: Any, value: Any, path: str) -> Any:
    """Convert a YAML value to the annotated type, recursing into dataclass tuples."""
    if hint is Path:
        return Path(value)
    if get_origin(hint) is tuple:
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list, got {type(value).__name__}")
        args = get_args(hint)
        inner = args[0] if args else str
        if isinstance(inner, type) and dataclasses.is_dataclass(inner):
            if not isinstance(value, list):
                raise ConfigError(f"{path} must be a list")
            return tuple(_build(inner, item, f"{path}[{i}]") for i, item in enumerate(value))
        return tuple(value)
    return value


def _build(cls: type, raw: dict[str, Any] | None, section: str) -> Any:
    raw = raw or {}
    hints = get_type_hints(cls)
    unknown = set(raw) - set(hints)
    if unknown:
        raise ConfigError(f"{section}: unknown key(s) {sorted(unknown)}")
    return cls(**{name: _coerce(hints[name], value, f"{section}.{name}")
                  for name, value in raw.items()})


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "model" not in raw:
        raise ConfigError(f"{path}: expected a mapping with a 'model' section")

    known = {"model", "data", "lora", "train", "export"}
    if set(raw) - known:
        raise ConfigError(f"{path}: unknown section(s) {sorted(set(raw) - known)}")

    cfg = Config(
        model=_build(ModelConfig, raw["model"], "model"),
        data=_build(DataConfig, raw.get("data"), "data"),
        lora=_build(LoraConfig, raw.get("lora"), "lora"),
        train=_build(TrainConfig, raw.get("train"), "train"),
        export=_build(ExportConfig, raw.get("export"), "export"),
    )
    return cfg.validate()


def warmup_steps(cfg: Config, train_rows: int) -> int:
    """transformers 5 removed warmup_ratio; convert it to steps ourselves."""
    per_epoch = max(1, train_rows // (cfg.train.micro_batch * cfg.train.grad_accum))
    total = max(1, int(per_epoch * cfg.train.epochs))
    return max(1, int(total * cfg.train.warmup_ratio))
