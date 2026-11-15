"""`cyberft push` merges the adapter and uploads the full model to the Hub."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import cyberft.export as export_mod
from cyberft.config import load_config
from cyberft.errors import ConfigError

REPO = Path(__file__).resolve().parents[1]
TINY = REPO / "configs" / "qlora-tiny.yaml"


class _FakeApi:
    def __init__(self):
        self.created = []
        self.uploaded = []

    def create_repo(self, repo_id, exist_ok=False):
        self.created.append((repo_id, exist_ok))

    def upload_folder(self, repo_id=None, folder_path=None, **kwargs):
        self.uploaded.append((repo_id, folder_path))


def _cfg_with_repo(tmp_path, repo="org/falcon-h1-tiny-cyber"):
    from cyberft.config import with_export

    return with_export(load_config(TINY), hf_repo=repo)


def test_push_without_repo_is_rejected():
    from cyberft.config import with_export

    cfg = with_export(load_config(TINY), hf_repo=None)
    assert cfg.export.hf_repo is None
    with pytest.raises(ConfigError, match=r"export\.hf_repo"):
        export_mod.push(cfg)


def test_empty_hf_repo_rejected(tmp_path):
    import re

    raw = TINY.read_text(encoding="utf-8")
    path = tmp_path / "empty-repo.yaml"
    path.write_text(re.sub(r"(?m)^  hf_repo:.*$", "  hf_repo: ''", raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"export\.hf_repo"):
        load_config(path)


def test_push_merges_then_uploads(tmp_path, monkeypatch):
    cfg = _cfg_with_repo(tmp_path)
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"fake")
    monkeypatch.setattr(export_mod, "merge", lambda cfg_, adapter_dir: merged)
    fake = _FakeApi()
    monkeypatch.setattr(export_mod, "HfApi", lambda: fake)

    assert export_mod.push(cfg) == "org/falcon-h1-tiny-cyber"
    assert fake.created == [("org/falcon-h1-tiny-cyber", True)]
    assert fake.uploaded == [("org/falcon-h1-tiny-cyber", str(merged))]


def test_push_uses_train_output_dir_by_default(tmp_path, monkeypatch):
    cfg = _cfg_with_repo(tmp_path)
    seen = {}
    merged = tmp_path / "merged"
    merged.mkdir()

    def _merge(cfg_, adapter_dir):
        seen["adapter_dir"] = adapter_dir
        return merged

    monkeypatch.setattr(export_mod, "merge", _merge)
    fake = _FakeApi()
    monkeypatch.setattr(export_mod, "HfApi", lambda: fake)

    export_mod.push(cfg)
    assert seen["adapter_dir"] == Path(cfg.train.output_dir)


# train-triggered push -------------------------------------------------------


class _FakeTrainer:
    def __init__(self, **kwargs):
        self.callbacks = kwargs.get("callbacks", [])

    def train(self, resume_from_checkpoint=None):
        return None


def _setup_train(monkeypatch, tmp_path, cfg, push_calls):
    """Stub the heavy train path; real config/validation/callback logic runs."""
    import cyberft.train as train_mod

    monkeypatch.setattr(train_mod, "build_tokenizer", lambda cfg_: calls_tracker(tmp_path, "tok"))
    monkeypatch.setattr(train_mod, "build_model", lambda cfg_: calls_tracker(tmp_path, "model"))
    monkeypatch.setattr(
        train_mod, "load_splits", lambda cfg_, smoke: {"train": [0] * 8, "eval": [0] * 2}
    )
    monkeypatch.setattr(train_mod, "Trainer", _FakeTrainer)
    monkeypatch.setattr(train_mod, "push_model", lambda cfg_: push_calls.append(cfg_))
    return train_mod


class _Saver:
    def __init__(self, tmp_path, name):
        self.tmp_path = tmp_path
        self.name = name

    def save_pretrained(self, *args, **kwargs):
        (self.tmp_path / f"{self.name}-saved").write_text("ok", encoding="utf-8")


def calls_tracker(tmp_path, name):
    return _Saver(tmp_path, name)


def test_train_auto_pushes_when_repo_set(tmp_path, monkeypatch):
    cfg = _cfg_with_repo(tmp_path)
    push_calls: list = []
    train_mod = _setup_train(monkeypatch, tmp_path, cfg, push_calls)
    train_mod.train(cfg, smoke=False)
    assert push_calls == [cfg]


def test_train_skips_push_without_repo(tmp_path, monkeypatch):
    from cyberft.config import with_export

    cfg = with_export(load_config(TINY), hf_repo=None)
    push_calls: list = []
    train_mod = _setup_train(monkeypatch, tmp_path, cfg, push_calls)
    train_mod.train(cfg, smoke=False)
    assert push_calls == []


def test_train_no_push_overrides_repo(tmp_path, monkeypatch):
    cfg = _cfg_with_repo(tmp_path)
    push_calls: list = []
    train_mod = _setup_train(monkeypatch, tmp_path, cfg, push_calls)
    train_mod.train(cfg, smoke=False, push=False)
    assert push_calls == []


def test_train_push_flag_without_repo_raises(tmp_path, monkeypatch):
    from cyberft.config import with_export

    cfg = with_export(load_config(TINY), hf_repo=None)
    push_calls: list = []
    train_mod = _setup_train(monkeypatch, tmp_path, cfg, push_calls)
    with pytest.raises(ConfigError, match="hf_repo"):
        train_mod.train(cfg, smoke=False, push=True)
    assert push_calls == []


def test_train_push_refused_in_smoke(tmp_path, monkeypatch):
    cfg = _cfg_with_repo(tmp_path)
    push_calls: list = []
    train_mod = _setup_train(monkeypatch, tmp_path, cfg, push_calls)
    with pytest.raises(ConfigError, match="smoke"):
        train_mod.train(cfg, smoke=True, push=True)
    assert push_calls == []


def test_train_push_failure_propagates(tmp_path, monkeypatch):
    cfg = _cfg_with_repo(tmp_path)
    push_calls: list = []
    train_mod = _setup_train(monkeypatch, tmp_path, cfg, push_calls)

    def _boom(cfg_):
        raise RuntimeError("hub down")

    monkeypatch.setattr(train_mod, "push_model", _boom)
    with pytest.raises(RuntimeError, match="hub down"):
        train_mod.train(cfg, smoke=False)
