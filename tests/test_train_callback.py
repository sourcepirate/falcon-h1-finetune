"""FiniteLossCallback aborts on NaN/inf; MetricsCallback mirrors logs to wandb."""

import math
from typing import ClassVar

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from cyberft.errors import NonFiniteLossError
from cyberft.train import FiniteLossCallback, MetricsCallback, expected_total_steps


class _State:
    global_step = 7
    log_history: ClassVar[list] = [{"loss": 2.0}, {"loss": 1.5, "eval_loss": 1.7}]


class _Tracker:
    def __init__(self):
        self.logged = []
        self.summaries = {}
        self.finished = []

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def summary(self, values):
        self.summaries.update(values)

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


def test_silent_on_finite_loss():
    cb = FiniteLossCallback()
    cb.on_log(None, _State(), None, logs={"loss": 1.25})  # must not raise


def test_silent_on_empty_logs():
    cb = FiniteLossCallback()
    cb.on_log(None, _State(), None, logs=None)
    cb.on_log(None, _State(), None, logs={})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("key", ["loss", "eval_loss"])
def test_raises_on_non_finite(bad, key):
    cb = FiniteLossCallback()
    assert not math.isfinite(bad)
    with pytest.raises(NonFiniteLossError, match=key):
        cb.on_log(None, _State(), None, logs={key: bad})


def test_metrics_callback_maps_trainer_keys():
    tracker = _Tracker()
    cb = MetricsCallback(tracker, total_steps=100)
    cb.on_log(None, _State(), None, logs={"loss": 1.25, "learning_rate": 2e-4, "epoch": 0.5})
    assert tracker.logged == [
        ({"train/loss": 1.25, "train/lr": 2e-4, "train/epoch": 0.5}, 7)
    ]


def test_metrics_callback_drops_non_finite_before_logging():
    tracker = _Tracker()
    cb = MetricsCallback(tracker, total_steps=100)
    cb.on_log(None, _State(), None, logs={"loss": float("nan")})
    assert tracker.logged == []  # FiniteLossCallback owns the abort; nothing is logged


def test_metrics_callback_reports_total_steps_and_final_summary():
    tracker = _Tracker()
    cb = MetricsCallback(tracker, total_steps=100)
    cb.on_train_begin(None, _State(), None)
    assert tracker.summaries == {"_total_steps": 100}
    cb.on_train_end(None, _State(), None)
    assert tracker.summaries["_total_steps"] == 100
    assert tracker.summaries["train/steps"] == 7
    assert tracker.summaries["train/final_loss"] == 1.5
    assert tracker.summaries["eval/final_loss"] == 1.7


def test_expected_total_steps():
    from pathlib import Path

    from cyberft.config import load_config, with_train

    repo_cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "qlora-tiny.yaml")
    assert expected_total_steps(repo_cfg, 64, smoke=True) == 5
    per_epoch = -(-1000 // (repo_cfg.train.micro_batch * repo_cfg.train.grad_accum))
    assert expected_total_steps(repo_cfg, 1000, smoke=False) == per_epoch
    probed = with_train(repo_cfg, max_steps=50)
    assert expected_total_steps(probed, 1000, smoke=False) == 50
