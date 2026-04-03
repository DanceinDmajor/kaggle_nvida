"""Generic JSONL importer based on a small field-mapping configuration."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from kaggle_nvida.io_utils import read_jsonl, write_jsonl
from kaggle_nvida.manifest import build_manifest_for_jsonl
from kaggle_nvida.schemas import (
    Difficulty,
    Message,
    Provenance,
    Quality,
    Safety,
    Supervision,
    Target,
    TrainingExample,
)


def _get_value(row: dict[str, Any], field_name: str | None, default: Any = "") -> Any:
    if not field_name:
        return default
    return row.get(field_name, default)


def _brief_reasoning(reasoning: str) -> str:
    if not reasoning:
        return ""
    compact = " ".join(reasoning.strip().split())
    for delimiter in (". ", "; ", "\n"):
        if delimiter in compact:
            head = compact.split(delimiter, 1)[0].strip()
            if head:
                return head.rstrip(".") + "."
    return compact[:160].rstrip() + ("..." if len(compact) > 160 else "")


def _assistant_content(reasoning_style: str, reasoning: str, answer: str) -> tuple[str, Supervision]:
    final_answer = f"Final answer: {answer}"
    if reasoning_style == "full_reasoning" and reasoning:
        content = f"Reasoning:\n{reasoning.strip()}\n{final_answer}"
        return content, Supervision(
            loss_mask_style="assistant_only",
            has_reasoning=True,
            has_final_answer_tag=True,
            final_answer_span=final_answer,
        )
    if reasoning_style == "brief_reasoning" and reasoning:
        content = f"Reasoning: {_brief_reasoning(reasoning)}\n{final_answer}"
        return content, Supervision(
            loss_mask_style="assistant_only",
            has_reasoning=True,
            has_final_answer_tag=True,
            final_answer_span=final_answer,
        )
    return final_answer, Supervision(
        loss_mask_style="assistant_only",
        has_reasoning=False,
        has_final_answer_tag=True,
        final_answer_span=final_answer,
    )


def _build_variants(row: dict[str, Any], mapping: dict[str, Any], row_index: int) -> list[TrainingExample]:
    source_name = mapping["source_name"]
    system_prompt = _get_value(row, mapping.get("system_field"), mapping.get("default_system_prompt", ""))
    user_text = _get_value(row, mapping.get("user_field"))
    answer_text = _get_value(row, mapping.get("answer_field"))
    reasoning_text = _get_value(row, mapping.get("reasoning_field"))

    raw_problem_id = _get_value(row, mapping.get("problem_id_field"))
    raw_source_id = _get_value(row, mapping.get("source_sample_id_field"), raw_problem_id)
    problem_id = str(raw_problem_id or f"{source_name}_{row_index:06d}_{uuid.uuid4().hex[:8]}")
    source_sample_id = str(raw_source_id or problem_id)

    variants = mapping.get("variants", ["answer_only"])
    task_family = mapping["task_family"]
    task_subtype = mapping.get("task_subtype", "imported")
    split = _get_value(row, mapping.get("split_field"), mapping.get("default_split", "train"))
    stages = mapping.get("stage_target", ["stage1"])
    source_level = mapping.get("source_level", "unknown")
    generator_model = mapping.get("generator_model", "imported_dataset")
    generator_prompt_id = mapping.get("generator_prompt_id", "import_jsonl_v1")

    examples: list[TrainingExample] = []
    for variant_id, reasoning_style in (
        ("full", "full_reasoning"),
        ("brief", "brief_reasoning"),
        ("ansonly", "answer_only"),
    ):
        if reasoning_style not in variants:
            continue
        assistant_content, supervision = _assistant_content(reasoning_style, str(reasoning_text or ""), str(answer_text))
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=str(system_prompt)))
        messages.extend(
            [
                Message(role="user", content=str(user_text)),
                Message(role="assistant", content=assistant_content),
            ]
        )
        examples.append(
            TrainingExample(
                example_id=f"{problem_id}_{variant_id}",
                problem_id=problem_id,
                variant_id=variant_id,
                task_family=task_family,
                task_subtype=task_subtype,
                split=str(split),
                stage_target=list(stages),
                format_type="chat",
                reasoning_style=reasoning_style,
                messages=messages,
                target=Target(
                    final_answer=str(answer_text),
                    normalized_final_answer=str(answer_text).strip(),
                    answer_format=mapping.get("answer_format", "short_text"),
                ),
                supervision=supervision,
                quality=Quality(
                    verified=bool(mapping.get("default_verified", False)),
                    judge_score=float(mapping.get("default_judge_score", 0.0)),
                    quality_band=mapping.get("default_quality_band", "unknown"),
                ),
                difficulty=Difficulty(
                    source_level=source_level,
                    profile_bucket="unprofiled",
                    estimated_steps=1 if reasoning_style == "answer_only" else 2,
                ),
                provenance=Provenance(
                    source_name=source_name,
                    source_sample_id=source_sample_id,
                    generator_model=generator_model,
                    generator_prompt_id=generator_prompt_id,
                    transform_chain=["import_jsonl_v1"],
                ),
                safety=Safety(),
            )
        )
    return examples


def import_jsonl_dataset(
    input_path: Path,
    mapping_path: Path,
    output_path: Path,
    manifest_output_path: Path | None = None,
) -> dict[str, int]:
    """Normalize an external JSONL dataset into repository training examples."""
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows = list(read_jsonl(input_path))
    examples: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for example in _build_variants(row, mapping, row_index):
            examples.append(example.to_dict())

    dataset_rows = write_jsonl(output_path, examples)
    manifest_rows = 0
    if manifest_output_path is not None:
        manifest_rows = build_manifest_for_jsonl(output_path, manifest_output_path)
    return {"input_rows": len(rows), "dataset_rows": dataset_rows, "manifest_rows": manifest_rows}

