"""Stage mixture selection utilities."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from kaggle_nvida.io_utils import read_jsonl, write_jsonl


def _stage_flag_name(stage: str) -> str:
    return {
        "stage1": "selected_for_stage1",
        "stage2": "selected_for_stage2",
        "stage3": "selected_for_stage3",
    }[stage]


def build_stage_selection(
    manifest_path: Path,
    config_path: Path,
    output_path: Path,
    stage: str,
    max_examples: int,
) -> dict[str, int]:
    """Select a stage-specific subset by task family weights."""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = [row for row in read_jsonl(manifest_path) if row.get(_stage_flag_name(stage), False)]
    records.sort(key=lambda item: item.get("judge_score", 0.0), reverse=True)

    by_family: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_family[record["task_family"]].append(record)

    selected: list[dict] = []
    selected_ids: set[str] = set()
    summary: dict[str, int] = {}
    for family, weight in config["task_family_weights"].items():
        budget = max(1, math.floor(weight * max_examples))
        chosen = by_family.get(family, [])[:budget]
        selected.extend(chosen)
        selected_ids.update(row["example_id"] for row in chosen)
        summary[family] = len(chosen)

    if len(selected) < max_examples:
        remaining_budget = max_examples - len(selected)
        backfill = [row for row in records if row["example_id"] not in selected_ids][:remaining_budget]
        selected.extend(backfill)
        summary["backfill"] = len(backfill)

    selected.sort(key=lambda item: (item["task_family"], -item.get("judge_score", 0.0)))
    write_jsonl(output_path, selected)
    summary["total"] = len(selected)
    return summary
