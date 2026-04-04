"""Materialize DataDesigner, Curator, and NeMo RL style config bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaggle_nvida.io_utils import ensure_parent

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOOL_STACK_DIR = REPO_ROOT / "configs" / "tool_stack"

STAGE_OVERRIDES: dict[str, dict[str, Any]] = {
    "stage1": {
        "datadesigner": {
            "task_mix": {
                "math": 0.30,
                "retrieval": 0.20,
                "structured": 0.18,
                "instruction": 0.14,
                "self_correction": 0.10,
                "code": 0.08,
            },
            "generation_mode": "broad_coverage",
        },
        "curator": {
            "quality_threshold": 0.78,
            "profile_focus": ["coverage", "semantic_dedup", "verifier_pass"],
        },
        "nemo_rl": {
            "objective": "sft",
            "max_steps": 3200,
            "rollout_enabled": False,
        },
    },
    "stage2": {
        "datadesigner": {
            "task_mix": {
                "math": 0.35,
                "retrieval": 0.15,
                "structured": 0.12,
                "instruction": 0.10,
                "self_correction": 0.18,
                "code": 0.10,
            },
            "generation_mode": "hardening",
        },
        "curator": {
            "quality_threshold": 0.84,
            "profile_focus": ["hard_bucket", "borderline_bucket", "failure_replay"],
        },
        "nemo_rl": {
            "objective": "sft_hardening",
            "max_steps": 1800,
            "rollout_enabled": False,
        },
    },
    "stage3": {
        "datadesigner": {
            "task_mix": {
                "math": 0.18,
                "retrieval": 0.08,
                "structured": 0.30,
                "instruction": 0.20,
                "self_correction": 0.10,
                "code": 0.14,
            },
            "generation_mode": "calibration",
        },
        "curator": {
            "quality_threshold": 0.82,
            "profile_focus": ["format_stability", "short_reasoning", "answer_only"],
        },
        "nemo_rl": {
            "objective": "sft_calibration",
            "max_steps": 1200,
            "rollout_enabled": False,
        },
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_config_path(name: str) -> Path:
    return DEFAULT_TOOL_STACK_DIR / f"{name}_default.json"


def _load_stack_config(name: str, override_path: Path | None, stage: str) -> dict[str, Any]:
    config = _load_json(_default_config_path(name))
    if override_path is not None:
        config = _merge_dict(config, _load_json(override_path))
    return _merge_dict(config, STAGE_OVERRIDES[stage][name])


def materialize_tool_stack_bundle(
    stage: str,
    train_manifest_path: Path,
    export_path: Path,
    eval_slice_dir: Path,
    output_dir: Path,
    datadesigner_config_path: Path | None = None,
    curator_config_path: Path | None = None,
    nemo_rl_config_path: Path | None = None,
) -> dict[str, Any]:
    """Write a stage-specific tool bundle with external-style configs."""
    if stage not in STAGE_OVERRIDES:
        raise ValueError(f"Unsupported stage: {stage}")

    output_dir.mkdir(parents=True, exist_ok=True)
    datadesigner = _load_stack_config("datadesigner", datadesigner_config_path, stage)
    curator = _load_stack_config("curator", curator_config_path, stage)
    nemo_rl = _load_stack_config("nemo_rl", nemo_rl_config_path, stage)

    datadesigner["stage"] = stage
    datadesigner["inputs"]["seed_manifest"] = str(train_manifest_path)
    datadesigner["outputs"]["generated_dir"] = str(output_dir / "generated")
    datadesigner["outputs"]["curated_candidate_manifest"] = str(output_dir / "generated_candidates.jsonl")

    curator["stage"] = stage
    curator["inputs"]["manifest"] = str(train_manifest_path)
    curator["outputs"]["curated_manifest"] = str(output_dir / "curated_manifest.jsonl")
    curator["outputs"]["report_path"] = str(output_dir / "curation_report.json")

    nemo_rl["stage"] = stage
    nemo_rl["inputs"]["train_manifest"] = str(train_manifest_path)
    nemo_rl["inputs"]["train_export"] = str(export_path)
    nemo_rl["inputs"]["eval_slice_dir"] = str(eval_slice_dir)
    nemo_rl["outputs"]["checkpoint_dir"] = str(output_dir / "checkpoints")
    nemo_rl["outputs"]["metrics_path"] = str(output_dir / "nemo_rl_metrics.json")

    datadesigner_path = output_dir / "datadesigner_recipe.json"
    curator_path = output_dir / "curator_pipeline.json"
    nemo_rl_path = output_dir / "nemo_rl_recipe.json"
    summary_path = output_dir / "tool_stack_bundle.json"

    _write_json(datadesigner_path, datadesigner)
    _write_json(curator_path, curator)
    _write_json(nemo_rl_path, nemo_rl)

    summary = {
        "stage": stage,
        "train_manifest_path": str(train_manifest_path),
        "export_path": str(export_path),
        "eval_slice_dir": str(eval_slice_dir),
        "bundle_dir": str(output_dir),
        "configs": {
            "datadesigner": str(datadesigner_path),
            "curator": str(curator_path),
            "nemo_rl": str(nemo_rl_path),
        },
    }
    _write_json(summary_path, summary)
    return summary

