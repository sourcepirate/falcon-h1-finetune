"""Command line entry point. Thin wrapper over the library; all imports are top-level."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

import torch

from .args import build_parser
from .config import load_config
from .errors import CyberftError
from .evaluate import evaluate
from .export import export_gguf
from .loaders import data_manifest, prepare_data
from .model import amp_dtype
from .train import train

log = logging.getLogger("cyberft")

__all__ = ["build_parser", "main", "run"]


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)

    if args.command == "data":
        splits = prepare_data(cfg, smoke=args.smoke, force=args.force)
        print({name: len(ds) for name, ds in splits.items()})
        print(json.dumps(data_manifest(cfg, smoke=args.smoke), indent=2))
        return 0

    if args.command == "train":
        train(cfg, smoke=args.smoke)
        return 0

    if args.command == "eval":
        report = evaluate(cfg, smoke=args.smoke)
        print(report.summary())
        return 0 if not report.broken else 1

    if args.command == "export":
        path = export_gguf(cfg)
        print(f"{path} ({path.stat().st_size / 1e9:.2f} GB)")
        return 0

    if args.command == "info":
        print(f"gpu      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
        print(f"dtype    : {amp_dtype(cfg)}")
        print("config   :")
        print(json.dumps(asdict(cfg), indent=2, default=str))
        try:
            print("corpus   :")
            print(json.dumps(data_manifest(cfg), indent=2))
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
