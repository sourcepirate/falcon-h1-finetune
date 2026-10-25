"""Corpus caching must never hand back data built with a different configuration."""

import dataclasses
import json
from pathlib import Path

import pytest

pytest.importorskip("datasets")

from cyberft.config import load_config
from cyberft.data import fingerprint, prepare_data
from cyberft.errors import NotBuiltError

REPO = Path(__file__).resolve().parents[1]


def make_cfg(tmp_path: Path):
    cfg = load_config(REPO / "configs" / "qlora-3b.yaml")
    return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, output_dir=tmp_path))


def with_source(cfg, name: str = "redteam", **changes):
    sources = tuple(
        dataclasses.replace(source, **changes) if source.name == name else source
        for source in cfg.data.sources
    )
    return dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, sources=sources))


def test_fingerprint_changes_with_max_len(tmp_path):
    cfg = make_cfg(tmp_path)
    shorter = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, max_len=1024))
    assert fingerprint(cfg, smoke=False) != fingerprint(shorter, smoke=False)


def test_fingerprint_changes_with_cap(tmp_path):
    cfg = make_cfg(tmp_path)
    assert fingerprint(cfg, smoke=False) != fingerprint(with_source(cfg, cap=100), smoke=False)


def test_fingerprint_changes_with_repo(tmp_path):
    cfg = make_cfg(tmp_path)
    moved = with_source(cfg, repo="someone/else")
    assert fingerprint(cfg, smoke=False) != fingerprint(moved, smoke=False)


def test_fingerprint_changes_with_revision(tmp_path):
    cfg = make_cfg(tmp_path)
    pinned = with_source(cfg, revision="0" * 40)
    assert fingerprint(cfg, smoke=False) != fingerprint(pinned, smoke=False)


def test_fingerprint_separates_smoke_from_full(tmp_path):
    cfg = make_cfg(tmp_path)
    assert fingerprint(cfg, smoke=True) != fingerprint(cfg, smoke=False)


def test_manifest_wiring(tmp_path, monkeypatch):
    """Regression: fingerprint() is keyword-only; manifest() must call it that way."""
    from datasets import Dataset, DatasetDict

    import cyberft.data as data

    monkeypatch.setattr(data, "_resolved_revisions", lambda cfg: {})
    splits = DatasetDict({
        "train": Dataset.from_dict({"input_ids": [[1, 2, 3], [4, 5]]}),
        "eval": Dataset.from_dict({"input_ids": [[1, 2]]}),
    })

    cfg = make_cfg(tmp_path)
    report = data.manifest(cfg, splits, smoke=False)
    assert report["fingerprint"] == fingerprint(cfg, smoke=False)
    assert report["rows"] == {"train": 2, "eval": 1}
    assert report["tokens"]["max"] == 3
    assert set(report["sources"]) == {source.name for source in cfg.data.sources}


def test_content_hash_distinguishes_equal_length_corpora():
    from datasets import Dataset, DatasetDict

    from cyberft.data import content_hash

    def splits(ids):
        return DatasetDict({"train": Dataset.from_dict({"input_ids": ids}),
                            "eval": Dataset.from_dict({"input_ids": []})})

    assert content_hash(splits([[1, 2, 3]])) != content_hash(splits([[1, 2, 4]]))


def test_stale_cache_is_rejected(tmp_path):
    cfg = make_cfg(tmp_path)
    full = tmp_path / "full"
    full.mkdir()
    (full / "dataset_dict.json").write_text("{}", encoding="utf-8")
    (full / "manifest.json").write_text(json.dumps({"fingerprint": "deadbeef"}), encoding="utf-8")

    with pytest.raises(NotBuiltError, match="different configuration"):
        prepare_data(cfg)
