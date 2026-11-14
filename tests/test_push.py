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
    raw = TINY.read_text(encoding="utf-8")
    assert "hf_repo: null" in raw, "fixture drifted"
    path = tmp_path / "push.yaml"
    path.write_text(raw.replace("hf_repo: null", f"hf_repo: {repo}"), encoding="utf-8")
    return load_config(path)


def test_push_without_repo_is_rejected():
    cfg = load_config(TINY)
    assert cfg.export.hf_repo is None
    with pytest.raises(ConfigError, match=r"export\.hf_repo"):
        export_mod.push(cfg)


def test_empty_hf_repo_rejected(tmp_path):
    raw = TINY.read_text(encoding="utf-8")
    path = tmp_path / "empty-repo.yaml"
    path.write_text(raw.replace("hf_repo: null", "hf_repo: ''"), encoding="utf-8")
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
