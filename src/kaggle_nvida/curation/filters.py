"""Heuristic manifest curation for the first repository iteration."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from kaggle_nvida.io_utils import read_jsonl, write_jsonl


def _quality_band(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.7:
        return "medium"
    return "low"


def _heuristic_score(record: dict) -> float:
    score = 0.5
    if record.get("verifier_pass"):
        score += 0.2
    if record.get("token_count_completion", 0) < 160:
        score += 0.1
    if record.get("reasoning_style") in {"brief_reasoning", "self_correction"}:
        score += 0.1
    if record.get("task_family") in {"math", "instruction", "structured"}:
        score += 0.1
    return min(score, 1.0)


def curate_manifest(manifest_path: Path, output_path: Path) -> dict[str, int]:
    """Apply exact dedup preference and heuristic quality scoring."""
    records = list(read_jsonl(manifest_path))
    kept: list[dict] = []
    seen_exact = set()
    counts = Counter()

    for record in sorted(records, key=lambda item: (item["problem_id"], item["variant_id"])):
        counts["input"] += 1
        exact_key = (record["exact_dup_cluster"], record["reasoning_style"])
        if exact_key in seen_exact:
            counts["dropped_exact_dup"] += 1
            continue
        seen_exact.add(exact_key)
        score = max(float(record.get("judge_score", 0.0)), _heuristic_score(record))
        record["judge_score"] = round(score, 4)
        record["quality_band"] = _quality_band(score)
        record["mixture_weight"] = round(0.5 + score, 4)
        kept.append(record)
        counts["kept"] += 1

    write_jsonl(output_path, kept)
    return dict(counts)

