"""CLI argument parsing. Deliberately free of heavy imports so the parser stays
testable without the training stack (torch, transformers, datasets)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyberft", description="Falcon-H1 cyber QLoRA toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="configs/qlora-3b.yaml")

    p_data = sub.add_parser("data", help="build and cache the corpus")
    common(p_data)
    p_data.add_argument("--smoke", action="store_true")
    p_data.add_argument("--force", action="store_true", help="rebuild even if cached")

    p_train = sub.add_parser("train", help="fine-tune the adapter")
    common(p_train)
    p_train.add_argument("--smoke", action="store_true")

    p_eval = sub.add_parser("eval", help="per-row evaluation of the trained adapter")
    common(p_eval)
    p_eval.add_argument("--smoke", action="store_true")

    p_export = sub.add_parser("export", help="merge and export GGUF")
    common(p_export)

    p_info = sub.add_parser("info", help="config, hardware and corpus summary")
    common(p_info)

    return parser
