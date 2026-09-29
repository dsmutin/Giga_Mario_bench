"""Turn generated DNA into reverse-complement answers."""

from __future__ import annotations

from typing import Any

_COMP = str.maketrans("ACGT", "TGCA")


def reverse_complement(sequence: str) -> str:
    """Return the DNA reverse complement. Only ACGT is accepted."""
    unknown = sorted(set(sequence) - set("ACGT"))
    if unknown:
        raise ValueError(f"sequence has non-ACGT bases: {unknown}")
    return sequence.translate(_COMP)[::-1]


def answers_for(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Map each record id to its reverse-complement answer."""
    return [
        {"id": row["id"], "sequence": reverse_complement(row["sequence"])}
        for row in records
    ]
