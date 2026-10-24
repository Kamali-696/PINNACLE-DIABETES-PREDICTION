"""Create a fully fabricated mixed-type table for the DP-CTGAN toy demo.

This module intentionally has no paths, imports, or references to project
datasets. Every value comes from fixed distributions and a fixed random seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20261023
N_RECORDS = 500
CATEGORIES = ["never", "former", "current", "unknown"]


def _truncated_normal(rng: np.random.Generator, mean: float, sd: float,
                      low: float, high: float, n: int) -> np.ndarray:
    """Sample a normal distribution truncated to the public interval."""
    values: list[float] = []
    while len(values) < n:
        draw = rng.normal(mean, sd, size=max(2 * (n - len(values)), 32))
        values.extend(draw[(draw >= low) & (draw <= high)].tolist())
    return np.asarray(values[:n], dtype=np.float64)


def make_fabricated_data(n_records: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    """Return only fabricated rows using predefined distributions."""
    if n_records <= 0:
        raise ValueError("n_records must be positive")
    rng = np.random.default_rng(seed)
    age = rng.uniform(18.0, 90.0, size=n_records)
    bmi = _truncated_normal(rng, 28.0, 7.0, 12.0, 60.0, n_records)
    glucose = _truncated_normal(rng, 125.0, 35.0, 50.0, 300.0, n_records)
    smoking = rng.choice(CATEGORIES, size=n_records, p=[0.48, 0.24, 0.20, 0.08])

    # Fixed synthetic label-generating rule; it does not estimate any statistic.
    log_odds = -2.4 + 0.035 * (age - 45.0) + 0.055 * (bmi - 28.0) + 0.018 * (glucose - 125.0)
    probability = 1.0 / (1.0 + np.exp(-log_odds))
    diabetes = rng.binomial(1, probability, size=n_records).astype(np.int8)
    return pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "glucose": glucose,
        "smoking_history": pd.Categorical(smoking, categories=CATEGORIES),
        "diabetes": diabetes,
    })


def main() -> None:
    frame = make_fabricated_data()
    path = HERE / "fabricated_data.csv"
    frame.to_csv(path, index=False)
    metadata = {
        "origin": "entirely fabricated; no EHR/Pima data read or used",
        "rows": len(frame),
        "seed": SEED,
        "distributions": {
            "age": "Uniform(18, 90)",
            "bmi": "Normal(28, 7), truncated to [12, 60]",
            "glucose": "Normal(125, 35), truncated to [50, 300]",
            "smoking_history": {k: v for k, v in zip(CATEGORIES, [0.48, 0.24, 0.20, 0.08])},
            "diabetes": "Bernoulli from fixed logistic rule documented in this script",
        },
        "bounds": {"age": [18, 90], "bmi": [12, 60], "glucose": [50, 300]},
        "categories": CATEGORIES,
    }
    (HERE / "fabricated_data_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(frame)} fabricated rows to {path}")


if __name__ == "__main__":
    main()
