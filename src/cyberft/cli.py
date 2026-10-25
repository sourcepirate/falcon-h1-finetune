"""Command line entry point. Thin wrapper over the library; heavy imports are lazy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

from .errors import CyberftError

log = logging.getLogger("cyberft")


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

    p_eval = sub.add_parser("eval", help="per-row evaluation, or a generation prompt")
    common(p_eval)
    p_eval.add_argument("--adapter", default=None)
    p_eval.add_argument("--split", default="eval")
    p_eval.add_argument("--smoke", action="store_true")
    p_eval.add_argument("--prompt", action="append", default=[], help="generate instead of scoring")

    p_export = sub.add_parser("export", help="merge and export GGUF")
    common(p_export)
    p_export.add_argument("--adapter", default=None)
    p_export.add_argument("--skip-merge", action="store_true")

    p_info = sub.add_parser("info", help="config, hardware and corpus summary")
    common(p_info)
    p_info.add_argument("--smoke", action="store_true")

    return parser


def run(args: argparse.Namespace) -> int:
    from .config import load_config

    cfg = load_config(args.config)

    if args.command == "data":
        from .data import data_manifest, prepare_data

        splits = prepare_data(cfg, smoke=args.smoke, force=args.force)
        print({name: len(ds) for name, ds in splits.items()})
        print(json.dumps(data_manifest(cfg, smoke=args.smoke), indent=2))
        return 0

    if args.command == "train":
        from .train import train

        train(cfg, smoke=args.smoke)
        return 0

    if args.command == "eval":
        from .evaluate import evaluate, generate

        if args.prompt:
            answers = generate(cfg, args.prompt, adapter=args.adapter, smoke=args.smoke)
            for prompt, answer in zip(args.prompt, answers, strict=True):
                print(f"\nQ: {prompt}\nA: {answer}\n{'-' * 78}")
            return 0
        report = evaluate(cfg, adapter=args.adapter, split=args.split, smoke=args.smoke)
        print(report.summary())
        return 0 if not report.broken else 1

    if args.command == "export":
        from .export import export_gguf

        path = export_gguf(cfg, adapter=args.adapter, skip_merge=args.skip_merge)
        print(f"{path} ({path.stat().st_size / 1e9:.2f} GB)")
        return 0

    if args.command == "info":
        import torch

        from .data import data_manifest
        from .model import amp_dtype

        print(f"gpu      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
        print(f"dtype    : {amp_dtype(cfg)}")
        print("config   :")
        print(json.dumps(asdict(cfg), indent=2, default=str))
        try:
            print("corpus   :")
            print(json.dumps(data_manifest(cfg, smoke=args.smoke), indent=2))
        except CyberftError as exc:
            print(f"corpus   : not built ({exc})")
        return 0

    raise NotImplementedError(args.command)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except CyberftError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
