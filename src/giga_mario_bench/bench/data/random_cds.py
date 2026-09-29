"""randomCDS: random DNA strings and point-mutated partners.

The toy panel is 10 originals plus 10 mutants. Split assignment later holds
out 2 pairs (4 sequences) as zero-shot and assigns the other 16 sequences
with Giga_Mario.
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
    """Substitute each base independently with probability ``rate``."""
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
    n_pairs: int = 10,
    length: int = 16,
    rate: Any = 0.1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build ``n_pairs`` originals and ``n_pairs`` mutants.

    Identifiers are ``pXX_orig`` and ``pXX_mut`` sharing ``pair_id`` ``pXX``.
    """
    if n_pairs < 3:
        raise ValueError("n_pairs must be >= 3 so a train/val/test split can drop two zero-shot pairs")
    if length < 2:
        raise ValueError("length must be >= 2")
    spec = parse_rate(rate)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for index in range(n_pairs):
        pair_id = f"p{index:02d}"
        original = "".join(ALPHABET[int(i)] for i in rng.integers(0, 4, size=length))
        drawn = sample_rate(spec, rng)
        mutant = mutate(original, drawn, rng)
        rows.append(
            {
                "id": f"{pair_id}_orig",
                "pair_id": pair_id,
                "role": "original",
                "sequence": original,
                "rate": None,
                "length": length,
            }
        )
        rows.append(
            {
                "id": f"{pair_id}_mut",
                "pair_id": pair_id,
                "role": "mutant",
                "sequence": mutant,
                "rate": drawn,
                "length": length,
            }
        )
    return rows
