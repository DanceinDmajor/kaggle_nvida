"""Score prediction files against eval slice manifests."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from kaggle_nvida.datasets import extract_final_answer, resolve_manifest_examples


def _normalize_prediction(prediction: str, answer_format: str) -> tuple[str, bool]:
    final_answer = extract_final_answer(prediction)
    if answer_format == "json":
        try:
            payload = json.loads(final_answer)
        except json.JSONDecodeError:
            return final_answer, False
        return json.dumps(payload, ensure_ascii=True, sort_keys=True), True
    return final_answer.strip(), True


def _normalize_target(target: dict[str, Any]) -> str:
    if target["answer_format"] == "json":
        payload = json.loads(target["normalized_final_answer"])
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)
    return target["normalized_final_answer"].strip()


def score_prediction_file(
    slice_manifest_path: Path,
    predictions_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Score predictions against the referenced dataset examples."""
    examples = resolve_manifest_examples(slice_manifest_path)
    predictions = {
        row["example_id"]: row["prediction"]
        for row in (json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip())
    }

    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "exact_match": 0, "format_adherent": 0})
    total = 0
    exact_match = 0
    format_adherent = 0
    missing_predictions = []

    for example in examples:
        record = example["_manifest"]
        prediction = predictions.get(example["example_id"])
        if prediction is None:
            missing_predictions.append(example["example_id"])
            continue
        normalized_prediction, format_ok = _normalize_prediction(prediction, example["target"]["answer_format"])
        normalized_target = _normalize_target(example["target"])
        family = record["task_family"]
        by_family[family]["total"] += 1
        total += 1
        if format_ok:
            by_family[family]["format_adherent"] += 1
            format_adherent += 1
        if normalized_prediction == normalized_target:
            by_family[family]["exact_match"] += 1
            exact_match += 1

    summary = {
        "slice_manifest_path": str(slice_manifest_path),
        "predictions_path": str(predictions_path),
        "num_scored": total,
        "exact_match": exact_match / total if total else 0.0,
        "format_adherence": format_adherent / total if total else 0.0,
        "missing_predictions": missing_predictions,
        "by_family": {
            family: {
                "total": stats["total"],
                "exact_match": stats["exact_match"] / stats["total"] if stats["total"] else 0.0,
                "format_adherence": stats["format_adherent"] / stats["total"] if stats["total"] else 0.0,
            }
            for family, stats in sorted(by_family.items())
        },
    }

    if output_path is not None:
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

