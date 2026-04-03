"""Manifest construction for training JSONL files."""

from __future__ import annotations

from pathlib import Path

from kaggle_nvida.io_utils import read_jsonl, write_jsonl
from kaggle_nvida.schemas import ManifestRecord


def _token_count(text: str) -> int:
    return len(text.split())


def _exact_dup_cluster(example: dict) -> str:
    return f"exact_{abs(hash((example['problem_id'], example['target']['normalized_final_answer']))) % 10_000_000}"


def _semantic_dup_cluster(example: dict) -> str:
    prompt = example["messages"][1]["content"].lower().strip()
    return f"sem_{abs(hash(prompt)) % 10_000_000}"


def build_manifest_for_jsonl(dataset_path: Path, manifest_path: Path) -> int:
    """Create a manifest JSONL from a training dataset JSONL."""
    rows: list[dict] = []
    for line_number, example in enumerate(read_jsonl(dataset_path), start=1):
        prompt_text = "\n".join(message["content"] for message in example["messages"][:-1])
        completion_text = example["messages"][-1]["content"]
        record = ManifestRecord(
            example_id=example["example_id"],
            problem_id=example["problem_id"],
            variant_id=example["variant_id"],
            path=str(dataset_path),
            line_number=line_number,
            task_family=example["task_family"],
            task_subtype=example["task_subtype"],
            reasoning_style=example["reasoning_style"],
            token_count_prompt=_token_count(prompt_text),
            token_count_completion=_token_count(completion_text),
            token_count_total=_token_count(prompt_text) + _token_count(completion_text),
            source_name=example["provenance"]["source_name"],
            source_sample_id=example["provenance"]["source_sample_id"],
            generator_model=example["provenance"]["generator_model"],
            verifier_type="metadata",
            verifier_pass=bool(example["quality"]["verified"]),
            judge_score=float(example["quality"]["judge_score"]),
            exact_dup_cluster=_exact_dup_cluster(example),
            semantic_dup_cluster=_semantic_dup_cluster(example),
            difficulty_bucket=example["difficulty"]["source_level"],
            profile_bucket=example["difficulty"]["profile_bucket"],
            mixture_weight=1.0,
            selected_for_stage1="stage1" in example["stage_target"],
            selected_for_stage2="stage2" in example["stage_target"],
            selected_for_stage3="stage3" in example["stage_target"],
        )
        rows.append(record.to_dict())
    return write_jsonl(manifest_path, rows)

