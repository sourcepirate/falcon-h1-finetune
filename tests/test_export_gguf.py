"""GGUF build picks the smallest correct path: direct outtype vs llama-quantize."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import cyberft.export as export_mod
from cyberft.config import load_config, with_export

REPO = Path(__file__).resolve().parents[1]
TINY = REPO / "configs" / "qlora-tiny.yaml"


def _cfg(tmp_path, quant: str):
    return with_export(
        load_config(TINY),
        quant=quant,
        gguf_dir=tmp_path / "gguf",
        llama_cpp_dir=tmp_path / "llama.cpp",
    )


def test_q8_0_uses_converter_directly(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, "Q8_0")
    merged = tmp_path / "merged"
    merged.mkdir()
    calls: dict[str, object] = {}

    def _convert(cfg_, merged_, outtype="f16"):
        calls["convert"] = outtype
        out = Path(cfg.export.gguf_dir) / f"model-{outtype}.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"GGUF")
        return out

    monkeypatch.setattr(export_mod, "convert", _convert)
    monkeypatch.setattr(
        export_mod, "quantize", lambda *a, **k: (_ for _ in ()).throw(AssertionError("quantize"))
    )

    out = export_mod.build_gguf(cfg, merged)
    assert calls["convert"] == "q8_0"
    assert out.name == "model-q8_0.gguf"


def test_k_quant_goes_through_llama_quantize(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, "Q4_K_M")
    merged = tmp_path / "merged"
    merged.mkdir()
    calls: dict[str, object] = {}

    def _convert(cfg_, merged_, outtype="f16"):
        calls["convert"] = outtype
        out = Path(cfg.export.gguf_dir) / f"model-{outtype}.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"GGUF")
        return out

    def _quantize(cfg_, f16):
        calls["quantize_input"] = Path(f16).name
        out = Path(cfg.export.gguf_dir) / "model-q4_k_m.gguf"
        out.write_bytes(b"GGUF")
        return out

    monkeypatch.setattr(export_mod, "convert", _convert)
    monkeypatch.setattr(export_mod, "quantize", _quantize)

    out = export_mod.build_gguf(cfg, merged)
    assert calls == {"convert": "f16", "quantize_input": "model-f16.gguf"}
    assert out.name == "model-q4_k_m.gguf"


def test_convert_rejects_unknown_outtype(tmp_path):
    cfg = _cfg(tmp_path, "Q8_0")
    with pytest.raises(Exception, match="cannot write"):
        export_mod.convert(cfg, tmp_path, outtype="q4_k_m")


def test_find_binary_falls_back_to_path(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, "Q4_K_M")
    monkeypatch.setattr(export_mod.shutil, "which", lambda name: "/usr/local/bin/llama-quantize")
    assert export_mod._find_binary(cfg, "llama-quantize") == Path("/usr/local/bin/llama-quantize")
