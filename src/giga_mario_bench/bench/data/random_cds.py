"""randomCDS: random DNA strings and point-mutated partners.

The locked pipeline panel is 10_000 sequences of length 100
(``bench.build.DEFAULT_SPEC``): 4_000 originals and 4_000 mutants for
train/test/val, and 1_000 originals plus 1_000 mutants held out as zero-shot
pairs. The toy example still asks for 10 originals plus 10 mutants of fixed
length 16. Mutations are substitutions only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ALPHABET = "ACGT"


def parse_rate(rate: Any) -> tuple[Any, ...]:
    """Normalize a mutation rate.

    A float is fixed. A ``(low, high)`` pair is a uniform range. A longer
    sequence, or a comma-separated CLI string, is a set of choices. One value
    is drawn per mutant sequence.
    """
    if isinstance(rate, str):
        text = rate.strip()
        if ":" in text and "," not in text:
            low, high = text.split(":")
            return ("range", float(low), float(high))
        if "," in text:
            return ("choice", tuple(float(part) for part in text.split(",")))
        return ("fixed", float(text))
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        return ("fixed", float(rate))
    if isinstance(rate, tuple) and len(rate) == 2 and all(
        isinstance(item, (int, float)) for item in rate
    ):
        return ("range", float(rate[0]), float(rate[1]))
    values = tuple(float(item) for item in rate)
    if len(values) == 1:
        return ("fixed", values[0])
    return ("choice", values)


def sample_rate(spec: tuple[Any, ...], rng: np.random.Generator) -> float:
    """Draw one mutation rate from a parsed spec."""
    kind = spec[0]
    if kind == "fixed":
        return float(spec[1])
    if kind == "range":
        return float(rng.uniform(spec[1], spec[2]))
    return float(rng.choice(np.asarray(spec[1], dtype=float)))


def mutate(sequence: str, rate: float, rng: np.random.Generator) -> str:
    """Apply SNP substitutions. Insertions and deletions are not used.

    Each site changes independently with probability ``rate`` to one of the
    other three bases. The mutant has the same length as the original.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"mutation rate must be in [0, 1], got {rate}")
    chars = []
    for base in sequence:
        if rng.random() < rate:
            options = [item for item in ALPHABET if item != base]
            chars.append(options[int(rng.integers(0, len(options)))])
        else:
            chars.append(base)
    return "".join(chars)


def generate_pairs(
    *,
    n_pairs: int | None = None,
    length: int | None = None,
    rate: Any = 0.1,
    seed: int = 42,
    n_sequences: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> list[dict[str, Any]]:
    """Build originals and point-mutated partners.

    Omitted size arguments use ``resolve_panel`` (10_000 sequences, length
    100). ``length`` fixes every read. ``n_pairs`` counts pairs, so the row
    count is ``2 * n_pairs``. Identifiers are ``pXX_orig`` and ``pXX_mut``.
    A read length is drawn uniformly from ``min_length`` to ``max_length``
    inclusive; the locked default sets both ends to 100.
    """
    from giga_mario_bench.bench.build import resolve_panel

    panel = resolve_panel(
        {
            "n_pairs": n_pairs,
            "n_sequences": n_sequences,
            "length": length,
            "min_length": min_length,
            "max_length": max_length,
            "rate": rate,
            "seed": seed,
        }
    )
    n_pairs = int(panel["n_pairs"])
    min_length = int(panel["min_length"])
    max_length = int(panel["max_length"])
    spec = parse_rate(panel["rate"])
    rng = np.random.default_rng(int(panel["seed"]))
    width = max(2, len(str(n_pairs - 1)))
    rows: list[dict[str, Any]] = []
    for index in range(n_pairs):
        pair_id = f"p{index:0{width}d}"
        read_length = int(rng.integers(min_length, max_length + 1))
        original = "".join(ALPHABET[int(i)] for i in rng.integers(0, 4, size=read_length))
        drawn = sample_rate(spec, rng)
        mutant = mutate(original, drawn, rng)
        rows.append(
            {
                "id": f"{pair_id}_orig",
                "pair_id": pair_id,
                "role": "original",
                "sequence": original,
                "rate": None,
                "length": read_length,
            }
        )
        rows.append(
            {
                "id": f"{pair_id}_mut",
                "pair_id": pair_id,
                "role": "mutant",
                "sequence": mutant,
                "rate": drawn,
                "length": read_length,
            }
        )
    return rows
