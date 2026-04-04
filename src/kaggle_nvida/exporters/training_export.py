"""Export selected manifest rows into training-ready files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from kaggle_nvida.io_utils import ensure_parent, read_jsonl, write_jsonl


def _load_line_cache(dataset_paths: set[str]) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    for dataset_path in dataset_paths:
        cache[dataset_path] = Path(dataset_path).read_text(encoding="utf-8").splitlines()
    return cache


def _resolve_examples(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_rows = list(read_jsonl(manifest_path))
    required_fields = {"path", "line_number", "example_id"}
    if manifest_rows and not required_fields.issubset(manifest_rows[0]):
        missing = sorted(required_fields.difference(manifest_rows[0]))
        raise ValueError(
            "Manifest rows must include dataset pointers for export. "
            f"Missing fields: {missing}"
        )
    line_cache = _load_line_cache({row["path"] for row in manifest_rows})
    examples = []
    for record in manifest_rows:
        raw_line = line_cache[record["path"]][record["line_number"] - 1]
        example = json.loads(raw_line)
        example["_manifest"] = record
        examples.append(example)
    return examples


def _to_prompt_completion(example: dict[str, Any]) -> dict[str, Any]:
    prompt = ""
    for message in example["messages"][:-1]:
        prompt += f"[{message['role'].upper()}]\n{message['content']}\n\n"
    completion = example["messages"][-1]["content"]
    return {
        "example_id": example["example_id"],
        "task_family": example["task_family"],
        "reasoning_style": example["reasoning_style"],
        "prompt": prompt.rstrip(),
        "completion": completion,
    }


def _to_tagged_text(example: dict[str, Any]) -> dict[str, Any]:
    chunks = []
    for message in example["messages"]:
        chunks.append(f"<|{message['role']}|>\n{message['content']}")
    return {
        "example_id": example["example_id"],
        "task_family": example["task_family"],
        "reasoning_style": example["reasoning_style"],
        "text": "\n\n".join(chunks),
    }


def _to_chat_jsonl(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": example["example_id"],
        "problem_id": example["problem_id"],
        "task_family": example["task_family"],
        "task_subtype": example["task_subtype"],
        "reasoning_style": example["reasoning_style"],
        "messages": example["messages"],
        "target": example["target"],
    }


def export_training_dataset(
    manifest_path: Path,
    output_path: Path,
    format_name: str = "chat_jsonl",
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Export manifest-selected rows into a training file."""
    examples = _resolve_examples(manifest_path)
    exporters = {
        "chat_jsonl": _to_chat_jsonl,
        "prompt_completion": _to_prompt_completion,
        "tagged_text": _to_tagged_text,
    }
    if format_name not in exporters:
        raise ValueError(f"Unsupported export format: {format_name}")

    output_rows = [exporters[format_name](example) for example in examples]
    write_jsonl(output_path, output_rows)

    family_counts = Counter(example["task_family"] for example in examples)
    style_counts = Counter(example["reasoning_style"] for example in examples)
    summary = {
        "manifest_path": str(manifest_path),
        "output_path": str(output_path),
        "format": format_name,
        "num_examples": len(output_rows),
        "task_family_counts": dict(family_counts),
        "reasoning_style_counts": dict(style_counts),
    }

    if summary_path is not None:
        ensure_parent(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
