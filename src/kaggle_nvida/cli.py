"""Command line entrypoint for kaggle_nvida."""

from __future__ import annotations

import argparse
from pathlib import Path

from kaggle_nvida.pipeline import create_bootstrap_math_pack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle-nvida",
        description="Toolkit for synthetic data, curation, and LoRA experiment tracking.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("info", help="Print project status information.")

    synth_parser = subparsers.add_parser(
        "bootstrap-math",
        help="Create a starter synthetic math dataset with multiple reasoning variants.",
    )
    synth_parser.add_argument("--output-dir", type=Path, required=True)
    synth_parser.add_argument("--count", type=int, default=100)
    synth_parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "info"):
        print("kaggle_nvida: project scaffold is ready.")
        return 0

    if args.command == "bootstrap-math":
        summary = create_bootstrap_math_pack(
            output_dir=args.output_dir,
            count=args.count,
            seed=args.seed,
        )
        print(
            "Created bootstrap math pack "
            f"with {summary['dataset_rows']} dataset rows and {summary['manifest_rows']} manifest rows."
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
