"""Utilities to merge model failure profiling back into the manifest."""

from __future__ import annotations

from pathlib import Path

from kaggle_nvida.io_utils import read_jsonl, write_jsonl


def apply_profile_results(manifest_path: Path, profiling_path: Path, output_path: Path) -> int:
    """Attach profile buckets from a JSONL file keyed by example_id."""
    profile_lookup = {
        row["example_id"]: row["profile_bucket"]
        for row in read_jsonl(profiling_path)
        if "example_id" in row and "profile_bucket" in row
    }
    rows = []
    for row in read_jsonl(manifest_path):
        row["profile_bucket"] = profile_lookup.get(row["example_id"], row.get("profile_bucket", "unprofiled"))
        rows.append(row)
    return write_jsonl(output_path, rows)

