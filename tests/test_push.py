"""`push` publishes a complete repo: merged weights, model card, and a GGUF.

Also covers the repair path (`--from-hub`) that rebuilds card + GGUF from the
weights already on the Hub, without the adapter or a retrain.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import cyberft.export as export_mod
import cyberft.hub as hub
from cyberft.config import load_config, with_export
from cyberft.errors import ConfigError, DependencyError

REPO = Path(__file__).resolve().parents[1]
TINY = REPO / "configs" / "qlora-tiny.yaml"


class _FakeApi:
    def __init__(self, remote: set[str] | None = None):
        self.created: list[dict] = []
        self.folders: list[dict] = []
        self.files: list[dict] = []
        self._remote = remote

    def create_repo(self, repo_id, repo_type="model", exist_ok=False):
        self.created.append(
            {"repo_id": repo_id, "repo_type": repo_type, "exist_ok": exist_ok}
        )

    def upload_folder(self, repo_id=None, folder_path=None, **kwargs):
        self.folders.append({"repo_id": repo_id, "folder_path": folder_path})

    def upload_file(self, repo_id=None, path_or_fileobj=None, path_in_repo=None, **kwargs):
        self.files.append(
            {"repo_id": repo_id, "local": path_or_fileobj, "remote": path_in_repo}
        )

    def list_repo_files(self, repo_id=None, repo_type="model", **kwargs):
        if self._remote is not None:
            return sorted(self._remote)
        remote = {entry["remote"] for entry in self.files}
        remote.add("config.json")
        if self.folders:
            remote |= {"model.safetensors", "README.md", "training_config.json"}
        return sorted(remote)


def _cfg(tmp_path, repo="org/falcon-h1-tiny-cyber"):
    cfg = with_export(load_config(TINY), hf_repo=repo, merged_dir=tmp_path / "merged",
                      gguf_dir=tmp_path / "gguf")
    return cfg


def _merged_dir(tmp_path) -> Path:
    merged = tmp_path / "merged"
    merged.mkdir(exist_ok=True)
    (merged / "config.json").write_text("{}", encoding="utf-8")
    (merged / "model.safetensors").write_bytes(b"weights")
    return merged


def _fake_gguf(cfg):
    path = Path(cfg.export.gguf_dir) / f"model-{cfg.export.quant.lower()}.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")

    def _build(cfg_, merged):
        assert Path(merged) == Path(cfg.export.merged_dir)
        return path

    return path, _build


def test_push_without_repo_is_rejected(tmp_path):
    cfg = with_export(load_config(TINY), hf_repo=None)
    with pytest.raises(ConfigError, match=r"export\.hf_repo"):
        hub.push(cfg)


def test_empty_hf_repo_rejected(tmp_path):
    import re

    raw = TINY.read_text(encoding="utf-8")
    path = tmp_path / "empty-repo.yaml"
    path.write_text(re.sub(r"(?m)^  hf_repo:.*$", "  hf_repo: ''", raw), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"export\.hf_repo"):
        load_config(path)


def test_push_uploads_weights_card_and_gguf(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    merged = _merged_dir(tmp_path)
    gguf, build = _fake_gguf(cfg)
    fake = _FakeApi()

    monkeypatch.setattr(export_mod, "merge", lambda cfg_, adapter_dir: merged)
    monkeypatch.setattr(export_mod, "build_gguf", build)
    monkeypatch.setattr(hub, "HfApi", lambda: fake)
    seen: dict[str, bool] = {}

    original_upload_folder = fake.upload_folder

    def _folder(**kwargs):
        seen["card"] = (merged / "README.md").is_file()
        seen["config"] = (merged / "training_config.json").is_file()
        original_upload_folder(**kwargs)

    fake.upload_folder = _folder  # type: ignore[method-assign]

    assert hub.push(cfg) == "org/falcon-h1-tiny-cyber"
    assert seen == {"card": True, "config": True}
    assert fake.created == [
        {"repo_id": "org/falcon-h1-tiny-cyber", "repo_type": "model", "exist_ok": True}
    ]
    assert fake.folders == [
        {"repo_id": "org/falcon-h1-tiny-cyber", "folder_path": str(merged)}
    ]
    assert fake.files == [
        {
            "repo_id": "org/falcon-h1-tiny-cyber",
            "local": str(gguf),
            "remote": gguf.name,
        }
    ]


def test_push_fails_when_gguf_generation_fails(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    merged = _merged_dir(tmp_path)
    fake = _FakeApi()
    monkeypatch.setattr(export_mod, "merge", lambda cfg_, adapter_dir: merged)
    monkeypatch.setattr(
        export_mod, "build_gguf", lambda cfg_, merged_: (_ for _ in ()).throw(DependencyError("no converter"))
    )
    monkeypatch.setattr(hub, "HfApi", lambda: fake)

    with pytest.raises(DependencyError, match="no converter"):
        hub.push(cfg)
    assert fake.created == [] and fake.folders == [] and fake.files == []


def test_push_verifies_remote_and_refuses_partial_repo(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    merged = _merged_dir(tmp_path)
    _, build = _fake_gguf(cfg)
    # Remote is missing the GGUF: push must not report success.
    fake = _FakeApi(remote={"README.md", "training_config.json", "config.json", "model.safetensors"})
    monkeypatch.setattr(export_mod, "merge", lambda cfg_, adapter_dir: merged)
    monkeypatch.setattr(export_mod, "build_gguf", build)
    monkeypatch.setattr(hub, "HfApi", lambda: fake)

    with pytest.raises(RuntimeError, match="incomplete"):
        hub.push(cfg)


def test_push_from_hub_repairs_without_weights_upload(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    gguf, _ = _fake_gguf(cfg)
    fake = _FakeApi()  # remote already holds the weights; verification checks the new files
    downloaded: dict[str, object] = {}

    def _snapshot(repo_id=None, local_dir=None, ignore_patterns=None):
        downloaded.update(
            {"repo_id": repo_id, "local_dir": local_dir, "ignore": ignore_patterns}
        )
        Path(local_dir, "config.json").write_text("{}", encoding="utf-8")
        Path(local_dir, "model.safetensors").write_bytes(b"w")
        return str(local_dir)

    monkeypatch.setattr(hub, "snapshot_download", _snapshot)
    monkeypatch.setattr(
        export_mod, "build_gguf", lambda cfg_, merged: (Path(merged, "config.json").is_file(), gguf)[1]
    )
    monkeypatch.setattr(hub, "HfApi", lambda: fake)
    monkeypatch.setattr(export_mod, "merge", lambda *a, **k: (_ for _ in ()).throw(AssertionError("merged")))

    assert hub.push(cfg, from_hub=True) == "org/falcon-h1-tiny-cyber"
    assert downloaded["repo_id"] == "org/falcon-h1-tiny-cyber"
    assert downloaded["ignore"] == ["*.gguf"]
    assert fake.folders == []  # weights stay untouched
    assert {entry["remote"] for entry in fake.files} == {
        "README.md",
        "training_config.json",
        gguf.name,
    }


def test_push_skip_merge_reuses_directory(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    merged = _merged_dir(tmp_path)
    _, build = _fake_gguf(cfg)
    fake = _FakeApi()
    monkeypatch.setattr(export_mod, "build_gguf", build)
    monkeypatch.setattr(hub, "HfApi", lambda: fake)
    monkeypatch.setattr(export_mod, "merge", lambda *a, **k: (_ for _ in ()).throw(AssertionError("merged")))

    assert hub.push(cfg, skip_merge=True) == "org/falcon-h1-tiny-cyber"
    assert fake.folders[0]["folder_path"] == str(merged)


def test_push_skip_merge_requires_existing_dir(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(Exception, match="skip_merge"):
        hub.push(cfg, skip_merge=True)


def test_from_hub_and_skip_merge_are_exclusive(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ConfigError, match="mutually exclusive"):
        hub.push(cfg, from_hub=True, skip_merge=True)


# --- train wiring: gate must run before any push -----------------------------


class _FakeTrainer:
    def __init__(self, **kwargs):
        self.callbacks = kwargs.get("callbacks", [])

    def train(self, resume_from_checkpoint=None):
        return None


class _Saver:
    def save_pretrained(self, *args, **kwargs):
        return None


def _setup_train(monkeypatch, cfg, *, order, gate_error=None):
    import cyberft.train as train_mod

    monkeypatch.setattr(train_mod, "build_tokenizer", lambda cfg_: _Saver())
    monkeypatch.setattr(train_mod, "build_model", lambda cfg_: _Saver())
    from datasets import Dataset, DatasetDict

    monkeypatch.setattr(
        train_mod,
        "load_splits",
        lambda cfg_, smoke: DatasetDict({"train": Dataset(range(8)), "eval": Dataset(range(2))}),
    )
    monkeypatch.setattr(train_mod, "Trainer", _FakeTrainer)
    monkeypatch.setattr(train_mod, "push_model", lambda cfg_: order.append("push"))

    def _gate(cfg_, trainer, tokenizer):
        order.append("gate")
        if gate_error is not None:
            raise gate_error
        return ["a plausible answer"]

    monkeypatch.setattr(train_mod, "run_generation_gate", _gate)
    return train_mod


def test_train_gates_before_pushing(tmp_path, monkeypatch):
    from cyberft.config import with_train

    cfg = with_train(_cfg(tmp_path), generation_gate=True)
    order: list[str] = []
    train_mod = _setup_train(monkeypatch, cfg, order=order)
    train_mod.train(cfg, smoke=False)
    assert order == ["gate", "push"]


def test_train_skips_gate_when_disabled(tmp_path, monkeypatch):
    from cyberft.config import with_train

    cfg = with_train(_cfg(tmp_path), generation_gate=False)
    order: list[str] = []
    train_mod = _setup_train(monkeypatch, cfg, order=order)
    train_mod.train(cfg, smoke=False)
    assert order == ["push"]


def test_train_gate_failure_blocks_push(tmp_path, monkeypatch):
    from cyberft.errors import DegenerateOutputError

    cfg = _cfg(tmp_path)
    order: list[str] = []
    train_mod = _setup_train(
        monkeypatch, cfg, order=order, gate_error=DegenerateOutputError("collapsed")
    )
    with pytest.raises(DegenerateOutputError, match="collapsed"):
        train_mod.train(cfg, smoke=False)
    assert order == ["gate"]


def test_train_smoke_never_gates_or_pushes(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    order: list[str] = []
    train_mod = _setup_train(monkeypatch, cfg, order=order)
    train_mod.train(cfg, smoke=True)
    assert order == []
