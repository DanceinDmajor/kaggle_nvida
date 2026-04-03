"""Command line entrypoint for kaggle_nvida."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle-nvida",
        description="Toolkit for synthetic data, curation, and LoRA experiment tracking.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("info", help="Print project status information.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "info"):
        print("kaggle_nvida: project scaffold is ready.")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

