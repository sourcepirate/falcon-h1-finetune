"""wandb/Goggle tracking: enablement, metric mapping, init and failure modes.

Torch-free: ``cyberft.tracking`` never imports the training stack, so this
file runs in the CPU lint/test job. The real ``wandb`` client is replaced by
a fake module in ``sys.modules``.
"""

import math
import sys
import types

import pytest

from cyberft.config import WandbConfig, load_config, with_train
from cyberft.errors import ConfigError, DependencyError, TrackingError
from cyberft.tracking import (
    NullTracker,
    WandbTracker,
    clean_metrics,
    final_summary,
    metric_name,
    resolve_enabled,
    start,
)


def _conf(**overrides):
    return WandbConfig(**overrides)


# resolve_enabled ------------------------------------------------------------


def test_auto_off_by_default():
    assert resolve_enabled(_conf(), smoke=False, env={}) is False


def test_auto_on_with_api_key():
    assert resolve_enabled(_conf(), smoke=False, env={"WANDB_API_KEY": "k"}) is True


@pytest.mark.parametrize("mode", ["online", "offline"])
def test_auto_on_with_explicit_mode(mode):
    assert resolve_enabled(_conf(), smoke=False, env={"WANDB_MODE": mode}) is True


def test_auto_off_when_disabled_mode():
    env = {"WANDB_API_KEY": "k", "WANDB_MODE": "disabled"}
    assert resolve_enabled(_conf(), smoke=False, env=env) is False


def test_auto_off_in_smoke_even_with_key():
    assert resolve_enabled(_conf(), smoke=True, env={"WANDB_API_KEY": "k"}) is False


def test_never_wins_over_key():
    assert resolve_enabled(_conf(enabled="never"), smoke=False, env={"WANDB_API_KEY": "k"}) is False


def test_always_wins_over_smoke():
    assert resolve_enabled(_conf(enabled="always"), smoke=True, env={}) is True


def test_cli_override_on_forces_smoke_tracking():
    assert resolve_enabled(_conf(), smoke=True, override=True, env={}) is True


def test_cli_override_off_forces_silence():
    env = {"WANDB_API_KEY": "k", "WANDB_MODE": "online"}
    assert resolve_enabled(_conf(enabled="always"), smoke=False, override=False, env=env) is False


def test_unknown_enabled_is_rejected():
    with pytest.raises(ConfigError, match=r"wandb\.enabled"):
        resolve_enabled("sometimes", smoke=False, env={})


# metric mapping -------------------------------------------------------------


def test_metric_name_mapping():
    assert metric_name("loss") == "train/loss"
    assert metric_name("learning_rate") == "train/lr"
    assert metric_name("epoch") == "train/epoch"
    assert metric_name("grad_norm") == "train/grad_norm"
    assert metric_name("eval_loss") == "eval/loss"
    assert metric_name("eval_runtime") == "eval/runtime"
    assert metric_name("train_samples_per_second") == "train/samples_per_second"
    assert metric_name("total_flos") == "total_flos"


def test_clean_metrics_keeps_only_finite_numbers():
    cleaned = clean_metrics({
        "train/loss": 1.25,
        "train/lr": 2e-4,
        "train/steps": 40,
        "_step": 7,
        "_runtime": 3.0,
        "bad": float("nan"),
        "worse": float("inf"),
        "name": "ckpt",
        "flag": True,
        "nothing": None,
    })
    assert cleaned == {"train/loss": 1.25, "train/lr": 2e-4, "train/steps": 40.0}


def test_final_summary_picks_last_finite_losses():
    history = [
        {"loss": 2.0, "epoch": 0.1},
        {"loss": 1.5, "eval_loss": 1.7},
        {"loss": float("nan")},
    ]
    assert final_summary(history, total_steps=100, global_step=12) == {
        "_total_steps": 100,
        "train/steps": 12,
        "train/final_loss": 1.5,
        "eval/final_loss": 1.7,
    }


def test_final_summary_empty_history():
    assert final_summary([], total_steps=5, global_step=5) == {
        "_total_steps": 5,
        "train/steps": 5,
    }


# start() with a fake client -------------------------------------------------


class _FakeSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeRun:
    def __init__(self):
        self.logged = []
        self.summaries = {}
        self.finished = []

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    @property
    def summary(self):
        run = self

        class _Summary(dict):
            def update(self, values):
                run.summaries.update(values)

        return _Summary()

    def finish(self, exit_code=0):
        self.finished.append(exit_code)


class _FakeWandb(types.ModuleType):
    def __init__(self, run=None, fail_init=False):
        super().__init__("wandb")
        self._run = run if run is not None else _FakeRun()
        self.fail_init = fail_init
        self.init_kwargs = None
        self.Settings = _FakeSettings

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        if self.fail_init:
            raise RuntimeError("no network")
        return self._run


@pytest.fixture
def fake_wandb(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_start_returns_null_when_disabled():
    tracker = start(_conf(), smoke=True, env={})
    assert isinstance(tracker, NullTracker)
    assert tracker.enabled is False
    tracker.log({"train/loss": 1.0}, step=1)  # must not raise
    tracker.summary({"_total_steps": 5})
    tracker.finish(exit_code=0)


def test_start_returns_null_off_rank(fake_wandb):
    tracker = start(_conf(enabled="always"), process_index=1, env={})
    assert isinstance(tracker, NullTracker)
    assert fake_wandb.init_kwargs is None  # never touched the client


def test_start_binds_goggle_env_over_yaml(fake_wandb):
    conf = _conf(project="yaml-project", entity="yaml-entity", run_name="yaml-name",
                 tags=("a",), mode="offline")
    env = {
        "WANDB_API_KEY": "k",
        "WANDB_PROJECT": "env-project",
        "WANDB_ENTITY": "env-entity",
        "WANDB_RUN_ID": "run-123",
    }
    tracker = start(conf, env=env, run_config={"smoke": True})
    assert isinstance(tracker, WandbTracker)
    kwargs = fake_wandb.init_kwargs
    assert kwargs["project"] == "env-project"
    assert kwargs["entity"] == "env-entity"
    assert kwargs["id"] == "run-123"
    assert kwargs["resume"] == "allow"
    assert kwargs["name"] == "yaml-name"
    assert kwargs["tags"] == ["a"]
    assert kwargs["config"] == {"smoke": True}
    assert kwargs["job_type"] == "train"


def test_start_offline_without_key_uses_disabled_mode(fake_wandb):
    tracker = start(_conf(), override=True, env={})
    assert isinstance(tracker, WandbTracker)
    assert fake_wandb.init_kwargs["mode"] == "disabled"


def test_start_missing_package_raises_dependency_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    import builtins

    real_import = builtins.__import__

    def _no_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_wandb)
    with pytest.raises(DependencyError, match="wandb"):
        start(_conf(enabled="always"), env={})


def test_start_init_failure_raises_tracking_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandb(fail_init=True))
    with pytest.raises(TrackingError, match=r"wandb\.init failed"):
        start(_conf(enabled="always"), env={})


def test_tracker_log_failure_degrades_to_warning(fake_wandb, caplog):
    run = fake_wandb._run

    def _boom(metrics, step=None):
        raise RuntimeError("filestream down")

    run.log = _boom
    tracker = WandbTracker(run)
    with caplog.at_level("WARNING", logger="cyberft.tracking"):
        tracker.log({"train/loss": 1.0}, step=1)  # must not raise
    assert "wandb.log failed" in caplog.text


def test_tracker_finish_is_idempotent(fake_wandb):
    tracker = WandbTracker(fake_wandb._run)
    tracker.finish(exit_code=0)
    tracker.finish(exit_code=1)
    assert fake_wandb._run.finished == [0]


def test_tracker_finish_failure_does_not_raise(caplog):
    run = _FakeRun()

    def _boom(exit_code=0):
        raise RuntimeError("finish down")

    run.finish = _boom
    with caplog.at_level("WARNING", logger="cyberft.tracking"):
        WandbTracker(run).finish(exit_code=1)  # must not raise
    assert "wandb.finish failed" in caplog.text


# config surface -------------------------------------------------------------


def test_tiny_config_wandb_block():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "qlora-tiny.yaml")
    assert cfg.wandb.enabled == "auto"
    assert cfg.wandb.project == "falcon-h1-cyber"
    assert "qlora" in cfg.wandb.tags


def test_bad_wandb_enabled_rejected(tmp_path):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    raw = (repo / "configs" / "qlora-tiny.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad-enabled.yaml"
    path.write_text(raw.replace("enabled: auto", "enabled: sometimes"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"wandb\.enabled"):
        load_config(path)


def test_bad_wandb_mode_rejected(tmp_path):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    raw = (repo / "configs" / "qlora-tiny.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad-mode.yaml"
    path.write_text(raw.replace("mode: null", "mode: sometimes"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"wandb\.mode"):
        load_config(path)


def test_empty_wandb_project_rejected(tmp_path):
    path = tmp_path / "empty-project.yaml"
    path.write_text(
        "model:\n  id: test/model\ndata:\n  sources:\n    - name: kali\n      repo: org/d\n"
        "wandb:\n  project: ''\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=r"wandb\.project"):
        load_config(path)


def test_unknown_wandb_key_rejected(tmp_path):
    path = tmp_path / "unknown-key.yaml"
    path.write_text(
        "model:\n  id: test/model\ndata:\n  sources:\n    - name: kali\n      repo: org/d\n"
        "wandb:\n  bogus: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_report_to_wandb_rejected():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs" / "qlora-tiny.yaml")
    with pytest.raises(ConfigError, match=r"train\.report_to"):
        with_train(cfg, report_to=("wandb",))


def test_nan_and_bool_never_become_metrics():
    assert clean_metrics({"loss": float("nan")}) == {}
    assert clean_metrics({"loss": True}) == {}
    assert math.isnan(clean_metrics({"loss": 1.0})["loss"]) is False
