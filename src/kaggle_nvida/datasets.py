"""Helpers for resolving examples from manifest files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaggle_nvida.io_utils import read_jsonl


def _load_line_cache(dataset_paths: set[str]) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    for dataset_path in dataset_paths:
        cache[dataset_path] = Path(dataset_path).read_text(encoding="utf-8").splitlines()
    return cache


def resolve_manifest_examples(manifest_path: Path) -> list[dict[str, Any]]:
    """Resolve dataset rows referenced by manifest path and line number."""
    manifest_rows = list(read_jsonl(manifest_path))
    required_fields = {"path", "line_number", "example_id"}
    if manifest_rows and not required_fields.issubset(manifest_rows[0]):
        missing = sorted(required_fields.difference(manifest_rows[0]))
        raise ValueError(
            "Manifest rows must include dataset pointers for export or evaluation. "
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


def extract_final_answer(text: str) -> str:
    """Extract the answer after the last final-answer tag, or fall back to the raw text."""
    marker = "Final answer:"
    if marker in text:
        return text.rsplit(marker, maxsplit=1)[-1].strip()
    return text.strip()

