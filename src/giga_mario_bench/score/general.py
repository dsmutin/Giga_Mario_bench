"""F1, R2, and ROC AUC from per-position base predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

ALPHABET = "ACGT"
INDEX = {base: i for i, base in enumerate(ALPHABET)}


def _as_index(sequence: str) -> np.ndarray:
    unknown = sorted(set(sequence) - set(ALPHABET))
    if unknown:
        raise ValueError(f"non-ACGT symbols in prediction row: {unknown}")
    return np.asarray([INDEX[base] for base in sequence], dtype=int)


def score_general(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score prediction rows.

    Each row needs ``y_true``, ``y_pred``, and ``proba`` with shape
    ``(length, 4)``. The similarity matrix counts true base versus predicted
    base. Macro F1 is computed from that matrix. R2 compares the true one-hot
    encoding with ``proba``. ROC AUC is the macro one-versus-rest area under
    the probability scores.
    """
    if not rows:
        raise ValueError("score_general received no rows")
    truth_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    proba_parts: list[np.ndarray] = []
    for row in rows:
        truth = _as_index(row["y_true"])
        pred = _as_index(row["y_pred"])
        if len(truth) != len(pred):
            raise ValueError(f"length mismatch for id {row.get('id')}")
        proba = np.asarray(row["proba"], dtype=float)
        if proba.shape != (len(truth), 4):
            raise ValueError(f"proba shape {proba.shape} does not match length {len(truth)}")
        truth_parts.append(truth)
        pred_parts.append(pred)
        proba_parts.append(proba)
    y_true = np.concatenate(truth_parts)
    y_pred = np.concatenate(pred_parts)
    proba = np.concatenate(proba_parts, axis=0)
    matrix = np.zeros((4, 4), dtype=int)
    for truth_i, pred_i in zip(y_true, y_pred):
        matrix[int(truth_i), int(pred_i)] += 1
    return {
        "f1": macro_f1(matrix),
        "r2": r2_one_hot(y_true, proba),
        "rocauc": macro_ovr_auc(y_true, proba),
        "similarity_matrix": matrix.tolist(),
        "n_tokens": int(y_true.size),
        "n_rows": len(rows),
    }


def character_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over bases, one count per output character."""
    truth = np.asarray(y_true, dtype=int).ravel()
    pred = np.asarray(y_pred, dtype=int).ravel()
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same number of characters")
    if truth.size == 0:
        raise ValueError("character_macro_f1 received no characters")
    matrix = np.zeros((4, 4), dtype=int)
    np.add.at(matrix, (truth, pred), 1)
    return macro_f1(matrix)


def macro_f1(matrix: np.ndarray) -> float:
    """Macro F1 over bases that appear at least once as the true label."""
    scores: list[float] = []
    for label in range(4):
        tp = float(matrix[label, label])
        support = float(matrix[label, :].sum())
        predicted = float(matrix[:, label].sum())
        if support == 0:
            continue
        fp = predicted - tp
        fn = support - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    if not scores:
        raise ValueError("similarity matrix has no labeled tokens")
    return float(np.mean(scores))


def r2_one_hot(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Coefficient of determination between true one-hot bases and probabilities."""
    truth = np.eye(4)[y_true]
    residual = float(np.sum((truth - proba) ** 2))
    center = truth - truth.mean(axis=0, keepdims=True)
    total = float(np.sum(center**2))
    if total == 0.0:
        return 0.0
    return float(1.0 - residual / total)


def macro_ovr_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Macro one-versus-rest ROC AUC. Classes absent from ``y_true`` are skipped."""
    areas: list[float] = []
    for label in range(4):
        binary = (y_true == label).astype(int)
        if binary.min() == binary.max():
            continue
        areas.append(_binary_auc(binary, proba[:, label]))
    if not areas:
        return float("nan")
    return float(np.mean(areas))


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via average ranks, with tie midranks."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are 1-based. Ties share the average rank.
        average = 0.5 * ((start + 1) + end)
        for pos in range(start, end):
            ranks[order[pos]] = average
        start = end
    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    rank_sum = float(ranks[y_true == 1].sum())
    return (rank_sum - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
