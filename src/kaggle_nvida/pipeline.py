"""High-level helpers for synthetic dataset creation."""

from __future__ import annotations

from pathlib import Path

from kaggle_nvida.io_utils import write_jsonl
from kaggle_nvida.manifest import build_manifest_for_jsonl
from kaggle_nvida.synthesis import (
    build_candidate_selection_variant,
    build_self_correction_variant,
    generate_math_dataset,
)


def create_bootstrap_math_pack(
    output_dir: Path,
    count: int = 100,
    seed: int = 7,
) -> dict[str, int]:
    """Generate a small starter pack with math and repair-style variants."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_examples = generate_math_dataset(count=count, seed=seed)
    expanded = []

    grouped: dict[str, list[dict]] = {}
    for example in base_examples:
        expanded.append(example.to_dict())
        grouped.setdefault(example.problem_id, []).append(example)

    for variants in grouped.values():
        full_variant = next(example for example in variants if example.variant_id == "full")
        wrong_answer = str(int(full_variant.target.final_answer) + 1)
        expanded.append(build_candidate_selection_variant(full_variant, wrong_answer).to_dict())
        expanded.append(build_self_correction_variant(full_variant, wrong_answer).to_dict())

    dataset_path = output_dir / "train.jsonl"
    manifest_path = output_dir / "manifest.jsonl"
    dataset_rows = write_jsonl(dataset_path, expanded)
    manifest_rows = build_manifest_for_jsonl(dataset_path=dataset_path, manifest_path=manifest_path)
    return {"dataset_rows": dataset_rows, "manifest_rows": manifest_rows}

