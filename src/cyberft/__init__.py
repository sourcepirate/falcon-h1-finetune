"""Falcon-H1 cyber fine-tuning (QLoRA).

Library usage:

    from cyberft import load_config, prepare_data, train, evaluate

    cfg = load_config("configs/qlora-3b.yaml")
    splits = prepare_data(cfg, smoke=True)
    train(cfg, smoke=True)

Command line: ``python -m cyberft.cli --help``.

torch, transformers and datasets are imported lazily, so ``import cyberft.config`` and
``import cyberft.errors`` stay cheap.
"""

from __future__ import annotations

from .errors import (
    ConfigError,
    CyberftError,
    DependencyError,
    NonFiniteLossError,
    NotBuiltError,
)

__version__ = "0.4.0"

# public name -> module defining it
_LAZY = {
    "Config": "config",
    "load_config": "config",
    "with_train": "config",
    "with_data": "config",
    "with_export": "config",
    "amp_dtype": "model",
    "build_tokenizer": "model",
    "prepare_data": "loaders",
    "data_manifest": "loaders",
    "train": "train",
    "train_dir": "train",
    "evaluate": "evaluate",
    "generate": "evaluate",
    "EvalReport": "evaluate",
    "export_gguf": "export",
}

__all__ = [
    "__version__",
    *_LAZY,
    "ConfigError",
    "CyberftError",
    "DependencyError",
    "NonFiniteLossError",
    "NotBuiltError",
]


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
