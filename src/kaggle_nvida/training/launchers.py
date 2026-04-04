"""Stage-oriented training bundle generation and launch helpers."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from kaggle_nvida.evaluation import build_eval_slices
from kaggle_nvida.exporters import export_training_dataset
from kaggle_nvida.integrations import materialize_tool_stack_bundle
from kaggle_nvida.io_utils import ensure_parent

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRAINING_DIR = REPO_ROOT / "configs" / "training"


def _default_stage_config_path(stage: str) -> Path:
    return DEFAULT_TRAINING_DIR / f"{stage}.json"


def _load_stage_config(stage: str, config_path: Path | None) -> dict[str, Any]:
    final_path = config_path or _default_stage_config_path(stage)
    return json.loads(final_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _export_filename(stage: str, export_format: str) -> str:
    suffix = {
        "chat_jsonl": "chat.jsonl",
        "prompt_completion": "prompt_completion.jsonl",
        "tagged_text": "tagged_text.jsonl",
    }[export_format]
    return f"{stage}_train.{suffix}"


def prepare_stage_run_bundle(
    stage: str,
    manifest_path: Path,
    run_dir: Path,
    config_path: Path | None = None,
    export_format: str = "chat_jsonl",
    eval_slice_dir: Path | None = None,
) -> dict[str, Any]:
    """Prepare a stage run directory with exports, configs, and a launch plan."""
    stage_config = _load_stage_config(stage, config_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    exports_dir = run_dir / "exports"
    tool_stack_dir = run_dir / "tool_stack"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    exports_dir.mkdir(parents=True, exist_ok=True)
    tool_stack_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if eval_slice_dir is None:
        eval_slice_dir = run_dir / "eval_slices"
    eval_slice_dir.mkdir(parents=True, exist_ok=True)
    eval_slice_index = build_eval_slices(manifest_path=manifest_path, output_dir=eval_slice_dir)

    train_export_path = exports_dir / _export_filename(stage, export_format)
    export_summary_path = exports_dir / f"{stage}_export_summary.json"
    export_summary = export_training_dataset(
        manifest_path=manifest_path,
        output_path=train_export_path,
        format_name=export_format,
        summary_path=export_summary_path,
    )

    tool_stack_summary = materialize_tool_stack_bundle(
        stage=stage,
        train_manifest_path=manifest_path,
        export_path=train_export_path,
        eval_slice_dir=eval_slice_dir,
        output_dir=tool_stack_dir,
    )

    stage_config_path = run_dir / "stage_config.json"
    _write_json(stage_config_path, stage_config)

    nemo_recipe_path = Path(tool_stack_summary["configs"]["nemo_rl"])
    launch_command = [
        "python3",
        "-m",
        stage_config["nemo_rl"]["launcher_module"],
        "--config",
        str(nemo_recipe_path),
    ]
    for key, value in stage_config["nemo_rl"].get("extra_cli_args", {}).items():
        launch_command.extend([f"--{key.replace('_', '-')}", str(value)])

    launch_plan = {
        "stage": stage,
        "manifest_path": str(manifest_path),
        "run_dir": str(run_dir),
        "export_format": export_format,
        "stage_config_path": str(stage_config_path),
        "train_export_path": str(train_export_path),
        "export_summary_path": str(export_summary_path),
        "eval_slice_index": eval_slice_index,
        "tool_stack_bundle": tool_stack_summary,
        "launch_command": launch_command,
        "checkpoints_dir": str(checkpoints_dir),
        "logs_dir": str(logs_dir),
        "notes": [
            "The exporter and tool stack bundle are already materialized.",
            "The launch command is intended for a NeMo RL capable environment.",
            "Eval slices can be prebuilt in the run directory before execute mode.",
        ],
    }
    launch_plan_path = run_dir / "launch_plan.json"
    _write_json(launch_plan_path, launch_plan)
    return launch_plan


def execute_stage_launch(run_dir: Path) -> int:
    """Execute the external launch command from a prepared run bundle."""
    launch_plan_path = run_dir / "launch_plan.json"
    launch_plan = json.loads(launch_plan_path.read_text(encoding="utf-8"))
    return subprocess.run(launch_plan["launch_command"], check=False).returncode


def _build_stage_arg_parser(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"launch_{stage}",
        description=f"Prepare or launch a {stage} training run bundle.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--export-format", choices=["chat_jsonl", "prompt_completion", "tagged_text"], default="chat_jsonl")
    parser.add_argument("--eval-slice-dir", type=Path, required=False)
    parser.add_argument("--execute", action="store_true")
    return parser


def run_stage_cli(stage: str) -> int:
    """Entry point for stage wrapper scripts."""
    parser = _build_stage_arg_parser(stage)
    args = parser.parse_args()
    launch_plan = prepare_stage_run_bundle(
        stage=stage,
        manifest_path=args.manifest,
        run_dir=args.run_dir,
        config_path=args.config,
        export_format=args.export_format,
        eval_slice_dir=args.eval_slice_dir,
    )
    print(json.dumps(launch_plan, indent=2))
    if args.execute:
        return execute_stage_launch(args.run_dir)
    return 0
