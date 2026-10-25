from pathlib import Path

import pytest

from cyberft.config import (
    Config,
    ConfigError,
    DataConfig,
    ModelConfig,
    SourceConfig,
    load_config,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "qlora-3b.yaml"

MINIMAL = """
model:
  id: test/model
data:
  sources:
    - name: kali
      repo: org/dataset
      cap: 10
"""


def _variant(tmp_path: Path, old: str, new: str) -> Path:
    raw = CONFIG.read_text(encoding="utf-8")
    assert old in raw, f"fixture drifted: {old!r} not in config"
    path = tmp_path / "config.yaml"
    path.write_text(raw.replace(old, new), encoding="utf-8")
    return path


def _cfg_with(*sources: SourceConfig) -> Config:
    return Config(model=ModelConfig(id="test/model"), data=DataConfig(sources=sources))


def test_repository_config_is_valid():
    cfg = load_config(CONFIG)
    names = [source.name for source in cfg.data.sources]

    assert len(names) == len(set(names)) == 7
    assert all(source.repo for source in cfg.data.sources)
    assert all(source.cap for source in cfg.data.sources)
    assert "out_proj" in cfg.model.bnb_skip_modules
    assert "out_proj" not in cfg.lora.target_modules
    assert "conv1d" not in cfg.lora.target_modules
    assert cfg.data.eval_max_len <= cfg.data.max_len
    # the auto-converted Parquet repo must stay pinned to the conversion ref
    redteam = next(source for source in cfg.data.sources if source.name == "redteam")
    assert redteam.parquet and redteam.revision


def test_minimal_config_is_valid(tmp_path):
    path = tmp_path / "minimal.yaml"
    path.write_text(MINIMAL, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.data.sources[0].name == "kali"


def test_unknown_section_is_rejected(tmp_path):
    path = _variant(tmp_path, "  max_len: 2048", "  max_len: 2048\n  bogus: 1")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_unknown_key_inside_source_is_rejected(tmp_path):
    path = tmp_path / "minimal.yaml"
    path.write_text(MINIMAL.replace("      cap: 10", "      cap: 10\n      bogus: 1"), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"data\.sources\[0\]: unknown key"):
        load_config(path)


def test_scalar_where_list_expected_is_rejected(tmp_path):
    path = _variant(tmp_path, "splits: [train, validation]", "splits: train")
    with pytest.raises(ConfigError, match="must be a list"):
        load_config(path)


def test_out_proj_as_lora_target_is_rejected(tmp_path):
    path = _variant(tmp_path, "target_modules: [q_proj", "target_modules: [out_proj, q_proj")
    with pytest.raises(ConfigError, match="out_proj"):
        load_config(path)


def test_missing_bnb_skip_modules_is_rejected(tmp_path):
    path = _variant(tmp_path, 'bnb_skip_modules: ["out_proj"]', "bnb_skip_modules: []")
    with pytest.raises(ConfigError, match="out_proj"):
        load_config(path)


def test_bad_dtype_is_rejected(tmp_path):
    path = _variant(tmp_path, "dtype: auto", "dtype: float128")
    with pytest.raises(ConfigError, match="dtype"):
        load_config(path)


def test_empty_sources_rejected():
    with pytest.raises(ConfigError, match="must not be empty"):
        _cfg_with().validate()


def test_duplicate_source_names_rejected():
    sources = (
        SourceConfig(name="kali", repo="org/a"),
        SourceConfig(name="kali", repo="org/b"),
    )
    with pytest.raises(ConfigError, match="duplicate"):
        _cfg_with(*sources).validate()


def test_cap_must_be_positive():
    with pytest.raises(ConfigError, match="cap must be positive"):
        _cfg_with(SourceConfig(name="kali", repo="org/a", cap=0)).validate()


def test_empty_splits_rejected():
    with pytest.raises(ConfigError, match="splits must not be empty"):
        _cfg_with(SourceConfig(name="kali", repo="org/a", splits=())).validate()


def test_parquet_and_subset_are_exclusive():
    source = SourceConfig(name="kali", repo="org/a", parquet="p.parquet", subset="default")
    with pytest.raises(ConfigError, match="mutually exclusive"):
        _cfg_with(source).validate()
