#!/usr/bin/env python3
"""Larger reverse-complement panel: learning curves and split gaps.

Forty originals and forty mutants (length 32, rate 0.1). Four pairs are
zero-shot. The other sequences get a Giga_Mario random split. Both baseline
models train with the registered hidden size. Weights are left at the last
epoch so the final table is the overfit end state; the best validation-loss
epoch is scored as well.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giga_mario_bench.bench.build import build_one  # noqa: E402
from giga_mario_bench.bin.traintestsplit import split_from_bench  # noqa: E402
from giga_mario_bench.io_utils import read_jsonl, read_pipe, write_json  # noqa: E402
from giga_mario_bench.models.data_prepare.encode import decode_sequence, prepare_records  # noqa: E402
from giga_mario_bench.models.general import MODELS  # noqa: E402
from giga_mario_bench.models.seqmodels import EncoderDecoder, ManyToManyRNN, fit_plateau  # noqa: E402
from giga_mario_bench.score.general import score_general  # noqa: E402

SPEC = {
    "n_pairs": 40,
    "length": 32,
    "rate": 0.1,
    "seed": 42,
    "zsv_pairs": 4,
}
HIDDEN = 32
MAX_EPOCHS = 40
# Val is only a handful of sequences, so loss noise trips a short patience
# before either model has fit the training panel. This run keeps going so the
# curves can show whether train F1 pulls away from val and zero-shot.
PATIENCE = 40
MODELS_TO_RUN = ("many_to_many_rnn", "encoder_decoder")
CLASSES = {
    "many_to_many_rnn": ManyToManyRNN,
    "encoder_decoder": EncoderDecoder,
}


def _buckets(prepared: list[dict], split_rows: list[dict]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    by_id = {row["id"]: row for row in prepared}
    buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in ("train", "val", "test", "zsv")
    }
    for row in split_rows:
        item = by_id[row["ID"]]
        buckets[row["train_test"]].append(
            (np.asarray(item["x"], dtype=int), np.asarray(item["y"], dtype=int))
        )
    return buckets


def _score_bucket(model, pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    rows = []
    for index, (x_idx, y_idx) in enumerate(pairs):
        proba = model.predict_proba(x_idx)
        pred = np.argmax(proba, axis=1)
        rows.append(
            {
                "id": str(index),
                "y_true": decode_sequence(y_idx),
                "y_pred": decode_sequence(pred),
                "proba": proba.tolist(),
            }
        )
    scored = score_general(rows)
    return {key: scored[key] for key in ("f1", "r2", "rocauc", "n_tokens")}


def _apply(model, params: dict[str, np.ndarray]) -> None:
    model.p = {key: value.copy() for key, value in params.items()}


def _plot(history: dict, path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].axvline(history["best_epoch"], color="0.5", linestyle="--", linewidth=1, label="best val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("mean cross-entropy")
    axes[0].set_title(f"{history['model']} loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_f1"], label="train")
    axes[1].plot(epochs, history["val_f1"], label="val")
    axes[1].axvline(history["best_epoch"], color="0.5", linestyle="--", linewidth=1, label="best val loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("per-character macro F1")
    axes[1].set_title(f"{history['model']} F1")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run() -> int:
    """Build the larger panel, train both models, and write curves plus split scores."""
    out = Path(__file__).resolve().parent / "out"
    bench_dir = out / "bench"
    built = build_one("reverse_complement", bench_dir, spec=SPEC)
    records = read_jsonl(bench_dir / "input" / "records.jsonl")
    answers = read_jsonl(bench_dir / "output" / "answers.jsonl")
    prepared = prepare_records(records, answers)
    written = split_from_bench(
        bench_dir,
        bench_dir / "splits" / "random",
        ["random"],
        seed=int(SPEC["seed"]),
        zsv_pairs=int(SPEC["zsv_pairs"]),
    )
    split_rows = read_pipe(written["random"])
    buckets = _buckets(prepared, split_rows)
    counts = {name: len(rows) for name, rows in buckets.items()}
    print("split counts", counts, "build", built["status"], flush=True)

    summaries = []
    for name in MODELS_TO_RUN:
        model = CLASSES[name](hidden=HIDDEN, seed=SPEC["seed"], lr=MODELS[name]["lr"])
        best_params: dict[str, np.ndarray] = {}

        def on_epoch(epoch: int, is_best: bool, _model=model, _best=best_params, _name=name) -> None:
            if is_best:
                _best.clear()
                _best.update({key: value.copy() for key, value in _model.p.items()})
            print(f"{_name} epoch {epoch} best={is_best}", flush=True)

        result = fit_plateau(
            model,
            buckets["train"],
            buckets["val"],
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed=int(SPEC["seed"]),
            restore_best=False,
            on_epoch=on_epoch,
        )
        last_params = {key: value.copy() for key, value in model.p.items()}
        end_scores = {split: _score_bucket(model, buckets[split]) for split in buckets}
        _apply(model, best_params)
        best_scores = {split: _score_bucket(model, buckets[split]) for split in buckets}
        _apply(model, last_params)
        history = {
            "model": name,
            "hidden": HIDDEN,
            "stopped_reason": result.stopped_reason,
            "epochs_ran": result.epochs_ran,
            "best_epoch": result.best_epoch,
            "train_loss": result.train_loss,
            "val_loss": result.val_loss,
            "train_f1": result.train_f1,
            "val_f1": result.val_f1,
            "end": end_scores,
            "best_val_loss": best_scores,
            "split_counts": counts,
            "spec": SPEC,
        }
        model_dir = out / name
        model_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_dir / "history.json", history)
        _plot(history, model_dir / "curves.png")
        end_gap = end_scores["val"]["f1"] - end_scores["zsv"]["f1"]
        best_gap = best_scores["val"]["f1"] - best_scores["zsv"]["f1"]
        history["end_val_minus_zsv_f1"] = end_gap
        history["best_val_minus_zsv_f1"] = best_gap
        write_json(model_dir / "history.json", history)
        summaries.append(history)
        print(
            f"{name} stopped={result.stopped_reason} epochs={result.epochs_ran} "
            f"best_epoch={result.best_epoch} end val-zsv F1={end_gap:.3f}",
            flush=True,
        )
    write_json(out / "summary.json", {"models": summaries, "split_counts": counts, "spec": SPEC})
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
