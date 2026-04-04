"""Build stage-specific evaluation slices from a selection manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaggle_nvida.datasets import resolve_manifest_examples
from kaggle_nvida.io_utils import ensure_parent, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_CONFIG = REPO_ROOT / "configs" / "eval" / "slices_default.json"


def _load_config(config_path: Path | None) -> dict[str, Any]:
    final_path = config_path or DEFAULT_EVAL_CONFIG
    return json.loads(final_path.read_text(encoding="utf-8"))


def _matches_filter(record: dict[str, Any], example: dict[str, Any], spec: dict[str, Any]) -> bool:
    if spec.get("task_families") and record["task_family"] not in spec["task_families"]:
        return False
    if spec.get("reasoning_styles") and record["reasoning_style"] not in spec["reasoning_styles"]:
        return False
    if spec.get("variant_ids") and record["variant_id"] not in spec["variant_ids"]:
        return False
    if spec.get("profile_buckets") and record.get("profile_bucket") not in spec["profile_buckets"]:
        return False
    if spec.get("difficulty_buckets") and record.get("difficulty_bucket") not in spec["difficulty_buckets"]:
        return False
    if spec.get("min_judge_score") is not None and record.get("judge_score", 0.0) < spec["min_judge_score"]:
        return False
    if spec.get("require_verifier_pass") and not record.get("verifier_pass", False):
        return False
    if spec.get("answer_formats") and example["target"]["answer_format"] not in spec["answer_formats"]:
        return False
    return True


def build_eval_slices(
    manifest_path: Path,
    output_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Write multiple eval slice manifests and an index summary."""
    config = _load_config(config_path)
    examples = resolve_manifest_examples(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_slices: list[dict[str, Any]] = []
    for spec in config["slices"]:
        matched_records = []
        for example in examples:
            record = dict(example["_manifest"])
            if _matches_filter(record, example, spec):
                record["eval_slice_name"] = spec["name"]
                matched_records.append(record)

        matched_records.sort(
            key=lambda item: (-float(item.get("judge_score", 0.0)), int(item.get("token_count_total", 0)))
        )
        selected = matched_records[: spec["max_examples"]]
        slice_path = output_dir / f"{spec['name']}.jsonl"
        write_jsonl(slice_path, selected)
        summary_slices.append(
            {
                "name": spec["name"],
                "path": str(slice_path),
                "num_examples": len(selected),
            }
        )

    index = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "slices": summary_slices,
    }
    index_path = output_dir / "index.json"
    ensure_parent(index_path)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index

