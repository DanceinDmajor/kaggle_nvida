"""Command line entrypoint for kaggle_nvida."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_nvida.curation import apply_profile_results, build_stage_selection, curate_manifest
from kaggle_nvida.exporters import export_training_dataset
from kaggle_nvida.importers import import_jsonl_dataset
from kaggle_nvida.pipeline import create_bootstrap_math_pack, create_bootstrap_mixed_pack
from kaggle_nvida.tracking import init_experiment_run, record_experiment_result


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

    mixed_parser = subparsers.add_parser(
        "bootstrap-mixed",
        help="Create a broader synthetic starter pack across multiple task families.",
    )
    mixed_parser.add_argument("--output-dir", type=Path, required=True)
    mixed_parser.add_argument("--count-per-family", type=int, default=100)
    mixed_parser.add_argument("--seed", type=int, default=7)

    curate_parser = subparsers.add_parser(
        "curate-jsonl",
        help="Apply heuristic curation to a manifest JSONL.",
    )
    curate_parser.add_argument("--dataset", type=Path, required=False)
    curate_parser.add_argument("--manifest", type=Path, required=True)
    curate_parser.add_argument("--output", type=Path, required=True)

    mixture_parser = subparsers.add_parser(
        "build-mixture",
        help="Build a stage-specific selection from a curated manifest.",
    )
    mixture_parser.add_argument("--manifest", type=Path, required=True)
    mixture_parser.add_argument("--config", type=Path, required=False)
    mixture_parser.add_argument("--stage", choices=["stage1", "stage2", "stage3"], required=True)
    mixture_parser.add_argument("--output", type=Path, required=True)
    mixture_parser.add_argument("--max-examples", type=int, default=1000)

    init_run_parser = subparsers.add_parser(
        "init-run",
        help="Create a run directory with starter config, notes, and metrics files.",
    )
    init_run_parser.add_argument("--runs-dir", type=Path, required=True)
    init_run_parser.add_argument("--experiment-id", required=True)

    profile_parser = subparsers.add_parser(
        "apply-profile",
        help="Merge profile buckets back into a manifest.",
    )
    profile_parser.add_argument("--manifest", type=Path, required=True)
    profile_parser.add_argument("--profile", type=Path, required=True)
    profile_parser.add_argument("--output", type=Path, required=True)

    tracker_parser = subparsers.add_parser(
        "record-run",
        help="Append a summary row to the experiment tracker CSV from a JSON file.",
    )
    tracker_parser.add_argument("--tracker", type=Path, required=True)
    tracker_parser.add_argument("--summary-json", type=Path, required=True)

    import_parser = subparsers.add_parser(
        "import-jsonl",
        help="Import an external JSONL dataset using a mapping config.",
    )
    import_parser.add_argument("--input", type=Path, required=True)
    import_parser.add_argument("--mapping", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    import_parser.add_argument("--manifest-output", type=Path, required=False)

    export_parser = subparsers.add_parser(
        "export-training",
        help="Export selected manifest rows into a training-ready file.",
    )
    export_parser.add_argument("--manifest", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument(
        "--format",
        choices=["chat_jsonl", "prompt_completion", "tagged_text"],
        default="chat_jsonl",
    )
    export_parser.add_argument("--summary", type=Path, required=False)
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

    if args.command == "bootstrap-mixed":
        summary = create_bootstrap_mixed_pack(
            output_dir=args.output_dir,
            count_per_family=args.count_per_family,
            seed=args.seed,
        )
        print(
            "Created bootstrap mixed pack "
            f"with {summary['dataset_rows']} dataset rows and {summary['manifest_rows']} manifest rows."
        )
        return 0

    if args.command == "curate-jsonl":
        summary = curate_manifest(manifest_path=args.manifest, output_path=args.output)
        print(f"Curated manifest with summary: {summary}")
        return 0

    if args.command == "build-mixture":
        config_path = args.config or Path("configs/mixtures/stage1_default.json")
        summary = build_stage_selection(
            manifest_path=args.manifest,
            config_path=config_path,
            output_path=args.output,
            stage=args.stage,
            max_examples=args.max_examples,
        )
        print(f"Built mixture with summary: {summary}")
        return 0

    if args.command == "init-run":
        run_dir = init_experiment_run(args.runs_dir, args.experiment_id)
        print(f"Initialized experiment run at {run_dir}")
        return 0

    if args.command == "apply-profile":
        rows = apply_profile_results(
            manifest_path=args.manifest,
            profiling_path=args.profile,
            output_path=args.output,
        )
        print(f"Updated {rows} manifest rows with profile buckets.")
        return 0

    if args.command == "record-run":
        summary = json.loads(args.summary_json.read_text(encoding='utf-8'))
        record_experiment_result(args.tracker, summary)
        print(f"Recorded experiment summary to {args.tracker}")
        return 0

    if args.command == "import-jsonl":
        summary = import_jsonl_dataset(
            input_path=args.input,
            mapping_path=args.mapping,
            output_path=args.output,
            manifest_output_path=args.manifest_output,
        )
        print(f"Imported dataset with summary: {summary}")
        return 0

    if args.command == "export-training":
        summary = export_training_dataset(
            manifest_path=args.manifest,
            output_path=args.output,
            format_name=args.format,
            summary_path=args.summary,
        )
        print(f"Exported training data with summary: {summary}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
