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
    """Map each record to the reverse complement of its own sequence.

    A mutant is labeled with the complement of the mutant, not of the original.
    """
    return [
        {"id": row["id"], "sequence": reverse_complement(row["sequence"])}
        for row in records
    ]


_INDEX = {base: index for index, base in enumerate("ACGT")}


def prediction_rows(
    records: list[dict[str, Any]],
    predict,
) -> list[dict[str, Any]]:
    """Score rows whose truth is the reverse complement.

    ``predict`` maps an input DNA string to a same-length ACGT string.
    Probabilities are one-hot on that prediction, so a perfect reverse
    complement scores F1, R2, and ROC AUC at 1.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        truth = reverse_complement(record["sequence"])
        pred = predict(record["sequence"])
        if len(pred) != len(truth):
            raise ValueError(f"prediction length mismatch for id {record.get('id')}")
        proba = [[0.0, 0.0, 0.0, 0.0] for _ in pred]
        for index, base in enumerate(pred):
            proba[index][_INDEX[base]] = 1.0
        rows.append(
            {
                "id": record["id"],
                "y_true": truth,
                "y_pred": pred,
                "proba": proba,
            }
        )
    return rows
