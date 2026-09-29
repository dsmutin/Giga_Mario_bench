"""Shared ACGT integer encoding used by every toy model adapter."""

from __future__ import annotations

from typing import Any

INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
ALPHABET = "ACGT"


def encode_sequence(sequence: str) -> list[int]:
    """Map an ACGT string to integers."""
    try:
        return [INDEX[base] for base in sequence]
    except KeyError as exc:
        raise ValueError(f"non-ACGT base in sequence: {exc}") from exc


def decode_sequence(indices: list[int] | Any) -> str:
    """Map integers back to ACGT."""
    return "".join(ALPHABET[int(i)] for i in indices)


def prepare_records(
    records: list[dict[str, Any]], answers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join inputs and answers into model-ready ``x`` / ``y`` rows."""
    by_id = {row["id"]: row["sequence"] for row in answers}
    ready = []
    for row in records:
        if row["id"] not in by_id:
            raise ValueError(f"missing answer for {row['id']}")
        ready.append(
            {
                "id": row["id"],
                "pair_id": row["pair_id"],
                "role": row["role"],
                "x": encode_sequence(row["sequence"]),
                "y": encode_sequence(by_id[row["id"]]),
            }
        )
    return ready
