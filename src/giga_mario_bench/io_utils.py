"""JSONL and pipe-delimited tables shared by the bench commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write objects as JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_pipe(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    """Write a Giga_Mario pipe-delimited table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("|".join(fields) + "\n")
        for row in rows:
            handle.write("|".join(str(row.get(field, "")) for field in fields) + "\n")


def read_pipe(path: Path) -> list[dict[str, str]]:
    """Read a pipe-delimited table with a header row."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    fields = lines[0].split("|")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        values = line.split("|")
        rows.append(dict(zip(fields, values)))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
