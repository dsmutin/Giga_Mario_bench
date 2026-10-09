"""Hydra entry for score prepare, with an optional exec afterwards."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from giga_mario_bench.score.execute import execute
from giga_mario_bench.score.prepare import prepare_grid


@hydra.main(version_base=None, config_path="../../../configs", config_name="score")
def main(cfg: DictConfig) -> None:
    """Prepare the score grid from a Hydra config. Run it when ``run=true``."""
    if not cfg.data_out or not cfg.model_out or not cfg.score_out:
        raise ValueError("data_out, model_out, and score_out are required")
    spec: dict = {"rate": cfg.rate, "seed": int(cfg.seed)}
    for key in ("n_sequences", "n_pairs", "min_length", "max_length", "length", "zsv_pairs"):
        value = cfg.get(key)
        if value is None or str(value) == "null":
            continue
        spec[key] = int(value)
    ratios = cfg.get("split_ratios")
    if ratios is not None and str(ratios) != "null":
        spec["split_ratios"] = [float(item) for item in ratios]
    prepare_grid(
        data_out=Path(str(cfg.data_out)),
        model_out=Path(str(cfg.model_out)),
        score_out=Path(str(cfg.score_out)),
        all_cells=bool(cfg.all),
        bench=None if cfg.bench in (None, "null") else str(cfg.bench),
        split=None if cfg.split in (None, "null") else str(cfg.split),
        model=None if cfg.model in (None, "null") else str(cfg.model),
        spec=spec,
        exec_root=None if cfg.exec_root in (None, "null") else Path(str(cfg.exec_root)),
        max_epochs=int(cfg.max_epochs),
        patience=int(cfg.patience),
    )
    if bool(cfg.run):
        root = None if cfg.exec_root in (None, "null") else Path(str(cfg.exec_root))
        code = execute(root)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
