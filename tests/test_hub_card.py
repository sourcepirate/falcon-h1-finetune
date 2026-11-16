"""The generated model card must carry real Hub metadata, not just prose."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("huggingface_hub")

from cyberft.config import load_config
from cyberft.hub import TRAINING_CONFIG_FILE, render_model_card, write_repo_metadata

REPO = Path(__file__).resolve().parents[1]
TINY = REPO / "configs" / "qlora-tiny.yaml"


def _card() -> str:
    cfg = load_config(TINY)
    return render_model_card(cfg, ["model-q8_0.gguf"])


def _front_matter(card: str) -> dict:
    assert card.startswith("---\n")
    return yaml.safe_load(card.split("---", 2)[1])


def test_front_matter_has_hub_metadata():
    front = _front_matter(_card())
    assert front["base_model"] == "tiiuae/Falcon-H1-Tiny-R-90M"
    assert front["library_name"] == "transformers"
    assert front["pipeline_tag"] == "text-generation"
    assert front["license"] == "other"
    assert front["license_name"] == "falcon-llm-license"
    assert "gguf" in front["tags"]
    assert "qlora" in front["tags"]
    assert len(front["datasets"]) == 7
    assert "WNT3D/Ultimate-Offensive-Red-Team" in front["datasets"]


def test_card_names_the_gguf_and_how_to_run_it():
    card = _card()
    assert "model-q8_0.gguf" in card
    assert "llama-cli -m model-q8_0.gguf" in card
    assert "llama.cpp" in card
    assert "AutoModelForCausalLM" in card


def test_card_includes_training_and_data_details():
    card = _card()
    assert "LoRA rank / alpha / dropout" in card
    assert "`q_proj`" in card
    assert "CC-BY-NC-SA-4.0" in card
    assert "authorised" in card.lower()


def test_write_repo_metadata_writes_card_and_config(tmp_path):
    cfg = load_config(TINY)
    written = write_repo_metadata(tmp_path, cfg, ["model-q8_0.gguf"])
    names = {path.name for path in written}
    assert names == {"README.md", TRAINING_CONFIG_FILE}
    assert (tmp_path / "README.md").read_text(encoding="utf-8").startswith("---")
    config = (tmp_path / TRAINING_CONFIG_FILE).read_text(encoding="utf-8")
    assert '"hf_repo"' in config and '"model"' in config
