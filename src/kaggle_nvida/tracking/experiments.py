"""Experiment folder and tracker management."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from kaggle_nvida.io_utils import ensure_parent

TRACKER_HEADER = [
    "experiment_id",
    "objective",
    "data_mix",
    "rank",
    "alpha",
    "dropout",
    "learning_rate",
    "target_modules",
    "best_step",
    "math_norm",
    "inst_exact",
    "struct_format",
    "short_reasoning_acc",
    "invalid_output_rate",
    "notes",
]


def init_experiment_run(runs_dir: Path, experiment_id: str) -> Path:
    """Create a standard run directory with starter files."""
    run_dir = runs_dir / experiment_id
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    notes_path = run_dir / "notes.md"
    metrics_path = run_dir / "metrics.json"

    if not config_path.exists():
        config_path.write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "objective": "stage1_sft",
                    "data_mix": "stage1_default",
                    "lora": {
                        "rank": 32,
                        "alpha": 64,
                        "dropout": 0.05,
                        "target_modules": "attn_only",
                    },
                    "training": {"learning_rate": 1e-4},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not notes_path.exists():
        notes_path.write_text(
            "# Experiment Notes\n\n"
            "## Goal\n\n"
            "Describe the hypothesis for this run.\n",
            encoding="utf-8",
        )
    if not metrics_path.exists():
        metrics_path.write_text(json.dumps({"experiment_id": experiment_id, "status": "initialized"}, indent=2) + "\n")
    return run_dir


def record_experiment_result(tracker_path: Path, row: dict[str, Any]) -> None:
    """Append an experiment summary row to the tracker CSV."""
    ensure_parent(tracker_path)
    tracker_exists = tracker_path.exists()
    with tracker_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKER_HEADER)
        if not tracker_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in TRACKER_HEADER})

