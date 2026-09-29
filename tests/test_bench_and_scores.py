"""Mandatory checks that do not train a model."""

from __future__ import annotations

from pathlib import Path

import pytest

from giga_mario_bench.bench.build import build_one
from giga_mario_bench.bench.data.random_cds import generate_pairs, parse_rate, sample_rate
from giga_mario_bench.models.seqmodels import EncoderDecoder, ManyToManyRNN, finite_difference_ok
from giga_mario_bench.score.general import score_general
from giga_mario_bench.vizualisation.metrics import plot_scores

import numpy as np

pytestmark = pytest.mark.mandatory


def test_build_is_idempotent_and_uses_pregenerated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A second build does not redraw, and datadir records are reused."""
    first = build_one("reverse_complement", tmp_path / "out", spec={"n_pairs": 10, "length": 8, "seed": 1})
    assert first["status"] == "built"
    second = build_one("reverse_complement", tmp_path / "out", spec={"n_pairs": 10, "length": 8, "seed": 1})
    assert second["status"] == "exists"
    assert "benchmark already generated" in capsys.readouterr().out

    pre = tmp_path / "datadir"
    pre.mkdir()
    rows = generate_pairs(n_pairs=10, length=8, rate=0.1, seed=3)
    from giga_mario_bench.io_utils import write_jsonl

    write_jsonl(pre / "records.jsonl", rows)
    used = build_one(
        "reverse_complement",
        tmp_path / "from_pre",
        datadir=pre,
        spec={"n_pairs": 10, "length": 8, "seed": 9},
    )
    assert used["status"] == "from_pregenerated"
    assert "using pre-generated data" in capsys.readouterr().out


def test_rate_choices_and_range() -> None:
    """A list of rates and a numeric range both draw a legal rate."""
    rng = np.random.default_rng(0)
    choice = parse_rate("0.05,0.2")
    drawn = sample_rate(choice, rng)
    assert drawn in {0.05, 0.2}
    span = parse_rate((0.0, 0.3))
    value = sample_rate(span, rng)
    assert 0.0 <= value <= 0.3


def test_score_general_perfect_prediction() -> None:
    """Identical strings score perfectly on F1 and ROC AUC."""
    proba = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    # Repeat so every class appears and the one-hot R2 is defined.
    row = {"id": "a", "y_true": "ACGT", "y_pred": "ACGT", "proba": proba}
    scored = score_general([row, row])
    assert scored["f1"] == pytest.approx(1.0)
    assert scored["rocauc"] == pytest.approx(1.0)
    assert scored["r2"] == pytest.approx(1.0)
    assert sum(sum(line) for line in scored["similarity_matrix"]) == 8


def test_analytical_gradients_match_finite_differences() -> None:
    """BPTT gradients stay close to a central finite difference."""
    rnn = ManyToManyRNN(hidden=3, seed=0)
    ed = EncoderDecoder(hidden=3, seed=0)
    assert finite_difference_ok(rnn, "W_y") < 1e-6
    assert finite_difference_ok(rnn, "W_xh_f") < 1e-6
    assert finite_difference_ok(ed, "W_xh") < 1e-5
    assert finite_difference_ok(ed, "W_xh_d") < 1e-5


def test_score_prepare_writes_a_cell_script(tmp_path: Path) -> None:
    """score prepare emits a standalone script under the exec tree."""
    from giga_mario_bench.score.prepare import prepare_grid

    scripts = prepare_grid(
        data_out=tmp_path / "data",
        model_out=tmp_path / "models",
        score_out=tmp_path / "scores",
        all_cells=False,
        bench="reverse_complement",
        split="random",
        model="many_to_many_rnn",
        exec_root=tmp_path / "exec",
        max_epochs=1,
        patience=1,
    )
    assert len(scripts) == 1
    text = scripts[0].read_text(encoding="utf-8")
    assert "run_cell" in text
    assert scripts[0].name.startswith("many_to_many_rnn_")
    assert scripts[0].parent.name == "random"


def test_score_prepare_writes_a_cell_script(tmp_path: Path) -> None:
    """score prepare emits a standalone script under the exec tree."""
    from giga_mario_bench.score.prepare import prepare_grid

    scripts = prepare_grid(
        data_out=tmp_path / "data",
        model_out=tmp_path / "models",
        score_out=tmp_path / "scores",
        all_cells=False,
        bench="reverse_complement",
        split="random",
        model="many_to_many_rnn",
        exec_root=tmp_path / "exec",
        max_epochs=1,
        patience=1,
    )
    assert len(scripts) == 1
    text = scripts[0].read_text(encoding="utf-8")
    assert "run_cell" in text
    assert scripts[0].name.startswith("many_to_many_rnn_")
    assert scripts[0].parent.name == "random"


def test_metric_plot_writes_html(tmp_path: Path) -> None:
    """Altair writes an HTML chart for the three scores."""
    paths = plot_scores({"f1": 0.5, "r2": 0.1, "rocauc": 0.7}, tmp_path / "metrics")
    assert Path(paths["html"]).is_file()
    text = Path(paths["html"]).read_text(encoding="utf-8")
    assert "f1" in text
