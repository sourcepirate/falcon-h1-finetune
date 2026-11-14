"""Report training metrics with the stock ``wandb`` client (Goggle-compatible).

Goggle implements the wandb control plane (``wandb.init`` / ``log`` /
``summary`` / ``finish``), so the official client logs metrics to Goggle
without code changes. This module is the only place that touches ``wandb``:

- torch-free, so it stays importable and testable without the training stack;
- ``wandb`` itself is imported lazily inside :func:`start`, so installs
  without the ``train`` extra never pay for it;
- init failures raise :class:`TrackingError` (a Goggle job must not run
  blind), while failures *after* init degrade to warnings and never abort
  training or mask its outcome.

Metric contract (mirrors ``examples/gpt2-finetune`` in the goggles repo):

- history: ``train/loss``, ``train/lr``, ``train/epoch``,
  ``train/grad_norm``, ``train/samples_per_second``, ``train/runtime``,
  ``eval/loss``, ``eval/*``;
- summary: ``_total_steps``, ``train/steps``, ``train/final_loss``,
  ``eval/final_loss``, ``data/train_rows``, ``data/eval_rows``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

from .config import WandbConfig
from .errors import DependencyError, TrackingError

log = logging.getLogger(__name__)

# Goggle supports ``wandb>=0.31,<0.32`` (the ``ops`` extra); keep the declared
# dependency in ``pyproject.toml`` inside this range.
WANDB_RANGE = "wandb>=0.31,<0.32"

_TRAIN_KEYS = {
    "loss": "train/loss",
    "learning_rate": "train/lr",
    "epoch": "train/epoch",
    "grad_norm": "train/grad_norm",
}


def resolve_enabled(
    conf: WandbConfig | str,
    *,
    smoke: bool,
    override: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Decide whether this process should open a wandb run.

    Precedence: explicit CLI override first, then ``wandb.enabled``
    (``never``/``always``), then ``auto``: smoke runs stay offline, an
    explicit ``WANDB_MODE=disabled`` stays off, otherwise a key or an
    explicit online/offline mode turns tracking on.
    """
    import os

    enabled = conf.enabled if isinstance(conf, WandbConfig) else conf
    if override is not None:
        return override
    if enabled == "never":
        return False
    if enabled == "always":
        return True
    if enabled != "auto":
        from .errors import ConfigError

        raise ConfigError(f"wandb.enabled must be one of auto|always|never, got {enabled!r}")
    source: Mapping[str, str] = env if env is not None else os.environ
    if smoke:
        return False
    if source.get("WANDB_MODE") == "disabled":
        return False
    if source.get("WANDB_API_KEY"):
        return True
    return source.get("WANDB_MODE") in ("online", "offline")


def metric_name(key: str) -> str:
    """Map a Trainer log key to the goggle-style metric name."""
    if key in _TRAIN_KEYS:
        return _TRAIN_KEYS[key]
    if key.startswith("eval_") and len(key) > len("eval_"):
        rest = key[len("eval_") :]
        return "eval/loss" if rest == "loss" else f"eval/{rest}"
    if key.startswith("train_") and len(key) > len("train_"):
        return f"train/{key[len('train_'):]}"
    return key


def clean_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Keep only finite numeric history values.

    Underscore-prefixed keys are wandb cursors (``_step``, ``_runtime``,
    ``_timestamp``), not metrics; bools and strings are summary-typed and
    Goggle ignores them server-side, so drop them client-side. Non-finite
    values never reach the backend: the ``FiniteLossCallback`` aborts first.
    """
    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            continue
        cleaned[key] = float(value)
    return cleaned


def final_summary(
    log_history: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    total_steps: int | None = None,
    global_step: int | None = None,
) -> dict[str, float | int]:
    """Summarise a Trainer run for ``wandb.summary``.

    ``total_steps`` feeds the Goggle progress bar (our ``train.max_steps``
    is ``-1`` for full-epoch runs, so the contract leaves
    ``total_steps_path`` blank and the UI reads ``_total_steps`` instead).
    """
    summary: dict[str, float | int] = {}
    if total_steps is not None:
        summary["_total_steps"] = int(total_steps)
    steps = global_step if global_step is not None else total_steps
    if steps is not None:
        summary["train/steps"] = int(steps)
    last_loss: float | None = None
    last_eval: float | None = None
    for entry in reversed(log_history):
        if last_loss is None:
            value = entry.get("loss")
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                last_loss = float(value)
        if last_eval is None:
            value = entry.get("eval_loss")
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                last_eval = float(value)
        if last_loss is not None and last_eval is not None:
            break
    if last_loss is not None:
        summary["train/final_loss"] = last_loss
    if last_eval is not None:
        summary["eval/final_loss"] = last_eval
    return summary


class NullTracker:
    """Drop-in that records nothing. Used off-Goggle, in smoke, or off-rank."""

    enabled = False

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        return None

    def summary(self, values: Mapping[str, Any]) -> None:
        return None

    def finish(self, exit_code: int = 0) -> None:
        return None


class WandbTracker:
    """Thin wrapper over a live ``wandb`` run. Never raises after init."""

    enabled = True

    def __init__(self, run: Any) -> None:
        self._run = run
        self._finished = False

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        try:
            self._run.log(dict(metrics), step=step)
        except Exception as exc:  # metrics must not abort training
            log.warning("wandb.log failed (continuing without metrics): %s", exc)

    def summary(self, values: Mapping[str, Any]) -> None:
        try:
            self._run.summary.update(dict(values))
        except Exception as exc:  # metrics must not abort training
            log.warning("wandb summary update failed (continuing): %s", exc)

    def finish(self, exit_code: int = 0) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self._run.finish(exit_code=exit_code)
        except Exception as exc:  # finish must not mask training outcome
            log.warning("wandb.finish failed: %s", exc)


def start(
    conf: WandbConfig,
    *,
    smoke: bool = False,
    override: bool | None = None,
    process_index: int = 0,
    run_config: dict[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> NullTracker | WandbTracker:
    """Open a wandb run, or return a :class:`NullTracker` when disabled.

    Env wins over YAML so Goggle's injected ``WANDB_*`` block binds the run
    without editing the config. Raises :class:`DependencyError` when tracking
    is requested but ``wandb`` is not installed, and :class:`TrackingError`
    when ``wandb.init`` itself fails.
    """
    import os

    source: Mapping[str, str] = env if env is not None else os.environ
    if process_index != 0:
        return NullTracker()
    if not resolve_enabled(conf, smoke=smoke, override=override, env=source):
        return NullTracker()
    try:
        import wandb
    except ImportError as exc:
        raise DependencyError(
            f"wandb tracking is enabled but {WANDB_RANGE} is not installed; "
            "install the train extra or pass --no-wandb"
        ) from exc

    project = source.get("WANDB_PROJECT") or conf.project
    entity = source.get("WANDB_ENTITY") or conf.entity
    run_id = source.get("WANDB_RUN_ID") or None
    mode = conf.mode or source.get("WANDB_MODE")
    if mode is None and not source.get("WANDB_API_KEY"):
        # No key (local/CI smoke with --wandb): disabled mode so init never prompts.
        mode = "disabled"

    kwargs: dict[str, Any] = {"job_type": "train"}
    if project is not None:
        kwargs["project"] = project
    if entity is not None:
        kwargs["entity"] = entity
    if run_id is not None:
        kwargs["id"] = run_id
        kwargs["resume"] = "allow"
    if conf.run_name is not None:
        kwargs["name"] = conf.run_name
    kwargs["config"] = dict(run_config or {})
    kwargs["tags"] = list(conf.tags)
    if mode is not None:
        kwargs["mode"] = mode
    try:
        kwargs["settings"] = wandb.Settings(init_timeout=60)
    except Exception as exc:  # older clients may lack Settings
        log.warning("wandb.Settings unavailable (%s); continuing without it", exc)
    try:
        run = wandb.init(**kwargs)
    except Exception as exc:
        raise TrackingError(f"wandb.init failed: {exc}") from exc
    if run is None:
        raise TrackingError("wandb.init returned None; cannot track this run")
    return WandbTracker(run)
