"""Shared train and test loop for reverse-complement sequence models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from giga_mario_bench.io_utils import read_jsonl, read_pipe, write_json, write_jsonl
from giga_mario_bench.models.data_prepare.encode import decode_sequence
from giga_mario_bench.models.general import MODELS
from giga_mario_bench.models.seqmodels import EncoderDecoder, ManyToManyRNN, fit_plateau

_CLASSES = {
    "many_to_many_rnn": ManyToManyRNN,
    "encoder_decoder": EncoderDecoder,
}


def train_model(
    model_name: str,
    prepared: Path,
    split_csv: Path,
    outdir: Path,
    *,
    hidden: int | None = None,
    max_epochs: int = 20,
    patience: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train until validation loss plateaus. Reuse an existing checkpoint."""
    outdir = Path(outdir)
    weights = outdir / "weights.npz"
    history_path = outdir / "train_history.json"
    if weights.is_file() and history_path.is_file():
        print(f"checkpoint already present: {weights}")
        return {"status": "exists", "checkpoint": str(weights)}
    if model_name not in _CLASSES:
        raise ValueError(f"unknown model {model_name!r}")
    width = MODELS[model_name]["hidden"] if hidden is None else hidden
    rows = read_jsonl(prepared)
    by_id = {row["id"]: row for row in rows}
    split_rows = read_pipe(split_csv)
    buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "train": [],
        "val": [],
        "test": [],
        "zsv": [],
    }
    for row in split_rows:
        if row["ID"] not in by_id:
            raise ValueError(f"split id {row['ID']} is missing from prepared data")
        item = by_id[row["ID"]]
        pair = (np.asarray(item["x"], dtype=int), np.asarray(item["y"], dtype=int))
        buckets[row["train_test"]].append(pair)
    if len(buckets["train"]) < 1:
        raise ValueError("train bucket is empty")
    model = _CLASSES[model_name](hidden=width, seed=seed, lr=MODELS[model_name]["lr"])
    result = fit_plateau(
        model,
        buckets["train"],
        buckets["val"],
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(weights, **{key: value for key, value in model.p.items()})
    payload = {
        "status": "trained",
        "model": model_name,
        "hidden": width,
        "stopped_reason": result.stopped_reason,
        "epochs_ran": result.epochs_ran,
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
        "train_f1": result.train_f1,
        "val_f1": result.val_f1,
        "best_epoch": result.best_epoch,
        "checkpoint": str(weights),
    }
    write_json(history_path, payload)
    print(
        f"trained {model_name} until {result.stopped_reason} "
        f"in {result.epochs_ran} epochs"
    )
    return payload


def test_model(
    model_name: str,
    prepared: Path,
    split_csv: Path,
    checkpoint: Path,
    out_path: Path,
    *,
    buckets: tuple[str, ...] = ("test", "zsv", "val"),
) -> dict[str, Any]:
    """Run a checkpoint on the requested split buckets."""
    out_path = Path(out_path)
    if out_path.is_file():
        print(f"predictions already present: {out_path}")
        return {"status": "exists", "predictions": str(out_path)}
    weights = np.load(checkpoint)
    if model_name == "many_to_many_rnn":
        width = int(weights["b_h_f"].shape[0])
    else:
        width = int(weights["b_h"].shape[0])
    model = _CLASSES[model_name](hidden=width, seed=0)
    model.p = {key: np.array(weights[key]) for key in model.p}
    rows = read_jsonl(prepared)
    by_id = {row["id"]: row for row in rows}
    predictions = []
    for row in read_pipe(split_csv):
        if row["train_test"] not in buckets:
            continue
        item = by_id[row["ID"]]
        x_idx = np.asarray(item["x"], dtype=int)
        proba = model.predict_proba(x_idx)
        pred_idx = np.argmax(proba, axis=1)
        predictions.append(
            {
                "id": row["ID"],
                "train_test": row["train_test"],
                "y_true": decode_sequence(item["y"]),
                "y_pred": decode_sequence(pred_idx),
                "proba": proba.tolist(),
            }
        )
    write_jsonl(out_path, predictions)
    return {"status": "tested", "predictions": str(out_path), "n": len(predictions)}
