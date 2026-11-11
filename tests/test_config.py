from pathlib import Path

import pytest

from cyberft.config import (
    Config,
    ConfigError,
    DataConfig,
    ModelConfig,
    SourceConfig,
    load_config,
    with_train,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "qlora-3b.yaml"
T4_CONFIG = REPO / "configs" / "qlora-3b-t4.yaml"

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
    assert "lm_head" in cfg.model.bnb_skip_modules
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


def test_missing_bnb_skip_modules_warns_but_loads(tmp_path):
    path = _variant(tmp_path, 'bnb_skip_modules: ["out_proj", "lm_head"]', "bnb_skip_modules: []")
    with pytest.warns(UserWarning, match="out_proj"):
        cfg = load_config(path)
    assert cfg.model.bnb_skip_modules == ()


def test_missing_lm_head_skip_warns_but_loads(tmp_path):
    path = _variant(
        tmp_path,
        'bnb_skip_modules: ["out_proj", "lm_head"]',
        'bnb_skip_modules: ["out_proj"]',
    )
    with pytest.warns(UserWarning, match="lm_head"):
        cfg = load_config(path)
    assert cfg.model.bnb_skip_modules == ("out_proj",)


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


TINY_CONFIG = REPO / "configs" / "qlora-tiny.yaml"


def test_tiny_config_is_valid_and_matches_3b_sources():
    full = load_config(CONFIG)
    tiny = load_config(TINY_CONFIG)
    assert tiny.model.id == "tiiuae/Falcon-H1-Tiny-R-90M"
    assert tiny.model.quant in ("4bit", "none")
    assert [s.name for s in tiny.data.sources] == [s.name for s in full.data.sources]
    assert len(tiny.data.sources) == 7
    assert tiny.data.eval_max_len <= tiny.data.max_len
    assert tiny.export.llama_cpp_rev is None


def test_bad_quant_is_rejected(tmp_path):
    path = _variant(tmp_path, "dtype: auto", "dtype: auto\n  quant: int8")
    with pytest.raises(ConfigError, match="quant"):
        load_config(path)


def test_quant_none_is_valid(tmp_path):
    raw = TINY_CONFIG.read_text(encoding="utf-8")
    assert "quant: 4bit" in raw, "fixture drifted"
    path = tmp_path / "tiny-none.yaml"
    path.write_text(raw.replace("quant: 4bit", "quant: none"), encoding="utf-8")
    assert load_config(path).model.quant == "none"


def test_t4_config_is_valid_and_smaller():
    full = load_config(CONFIG)
    t4 = load_config(T4_CONFIG)
    names = [source.name for source in t4.data.sources]
    assert names == [source.name for source in full.data.sources]
    assert len(names) == 7
    assert t4.data.max_len <= 1024
    assert t4.data.eval_max_len <= t4.data.max_len
    assert t4.train.save_steps <= 100


def test_max_steps_validation():
    cfg = load_config(CONFIG)
    with pytest.raises(ConfigError, match="max_steps"):
        with_train(cfg, max_steps=0)
    assert with_train(cfg, max_steps=100).train.max_steps == 100
    assert with_train(cfg, max_steps=-1).train.max_steps == -1


def test_with_train_overrides_output_dir(tmp_path):
    cfg = load_config(CONFIG)
    updated = with_train(cfg, output_dir=tmp_path / "probe", max_steps=100, resume=False)
    assert updated.train.output_dir == tmp_path / "probe"
    assert updated.train.max_steps == 100
    assert updated.train.resume is False
