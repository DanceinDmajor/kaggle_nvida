"""Lightweight I/O helpers for JSONL-based datasets and manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Write dictionaries to a JSONL file and return the row count."""
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Append dictionaries to a JSONL file and return the appended row count."""
    ensure_parent(path)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield dictionaries from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

