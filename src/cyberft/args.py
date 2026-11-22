"""CLI argument parsing. Deliberately free of heavy imports so the parser stays
testable without the training stack (torch, transformers, datasets)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyberft", description="Falcon-H1 cyber QLoRA toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", default="configs/qlora-3b.yaml")

    p_data = sub.add_parser("data", help="build and cache the corpus")
    common(p_data)
    p_data.add_argument("--smoke", action="store_true")
    p_data.add_argument("--force", action="store_true", help="rebuild even if cached")

    p_train = sub.add_parser("train", help="fine-tune the adapter")
    common(p_train)
    p_train.add_argument("--smoke", action="store_true")
    p_train.add_argument(
        "--wandb",
        "--no-wandb",
        dest="wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force wandb/Goggle tracking on or off (default: auto from config + env)",
    )
    p_train.add_argument(
        "--push",
        "--no-push",
        dest="push",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force Hub push on or off (default: push when export.hf_repo is set, never in smoke)",
    )

    p_eval = sub.add_parser("eval", help="per-row evaluation of the trained adapter")
    common(p_eval)
    p_eval.add_argument("--smoke", action="store_true")

    p_export = sub.add_parser("export", help="merge and export GGUF")
    common(p_export)

    p_push = sub.add_parser("push", help="merge, build a model card + GGUF, and upload to the Hub")
    common(p_push)
    source = p_push.add_mutually_exclusive_group()
    source.add_argument(
        "--from-hub",
        action="store_true",
        help="repair an existing Hub repo: rebuild card + GGUF from its published weights",
    )
    source.add_argument(
        "--skip-merge",
        action="store_true",
        help="reuse export.merged_dir instead of merging the adapter again",
    )

    p_info = sub.add_parser("info", help="config, hardware and corpus summary")
    common(p_info)

    return parser
