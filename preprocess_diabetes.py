"""Prepare the EHR and Pima diabetes datasets independently.

The exported split CSVs retain original feature names and categorical strings.
The serialized sklearn pipelines are fitted on training features only and can
be used later to encode/scale inputs consistently. Targets are always separate
from the transformer input and are appended unchanged to each exported split.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inspect_dataset(df: pd.DataFrame, name: str, target: str) -> str:
    """Print and return a comprehensive initial inspection."""
    _require(target in df.columns, f"{name}: expected target column {target!r} is absent. Found: {df.columns.tolist()}")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    lines = [f"DATASET INSPECTION: {name}", f"Shape: {df.shape}", f"Columns: {df.columns.tolist()}",
             "Data types:\n" + df.dtypes.to_string(), f"Numerical columns: {numeric}",
             f"Categorical columns: {categorical}", "First five rows:\n" + df.head().to_string(index=False),
             "Missing values by column:\n" + df.isna().sum().to_string(),
             f"Exact duplicate rows: {int(df.duplicated().sum())}",
             f"Target {target} distribution:\n{df[target].value_counts(dropna=False).sort_index().to_string()}"]
    result = "\n".join(lines)
    print(result)
    return result


def _split(df: pd.DataFrame, target: str):
    train, remainder = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df[target])
    validation, test = train_test_split(remainder, test_size=0.50, random_state=SEED, stratify=remainder[target])
    return train.copy(), validation.copy(), test.copy()


def _make_transformer(train: pd.DataFrame, features: list[str], categorical: list[str]):
    numerical = [c for c in features if c not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    return ColumnTransformer([("numeric", numeric_pipe, numerical), ("categorical", categorical_pipe, categorical)],
                             remainder="drop", verbose_feature_names_out=True)


def _counts(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    return {c: int(df[c].isna().sum()) for c in cols}


def _distribution(df: pd.DataFrame, target: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in df[target].value_counts().sort_index().items()}


def _prepare(name: str, path: Path, target: str, invalid_rules: dict[str, tuple[float, float]],
             categorical: list[str], zero_missing: list[str] | None = None,
             normalize_categories: dict[str, dict[str, str]] | None = None) -> None:
    _require(path.exists(), f"{name}: input CSV not found: {path}")
    original = pd.read_csv(path)
    inspection = inspect_dataset(original, name, target)
    _require(original.columns.is_unique, f"{name}: duplicate column names are not supported")
    _require(original[target].notna().all(), f"{name}: target contains missing labels")
    _require(set(pd.to_numeric(original[target], errors="coerce").dropna().unique()).issubset({0, 1}),
             f"{name}: target {target} must contain only binary 0/1 values")
    original_shape = original.shape
    original_dist = _distribution(original, target)

    df = original.copy()
    # Normalize whitespace and known sentinel spellings before counting missingness.
    for col in categorical:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    if normalize_categories:
        for col, mapping in normalize_categories.items():
            df[col] = df[col].replace(mapping)
    if name.lower() == "ehr" and "smoking_history" in df:
        df["smoking_history"] = df["smoking_history"].replace({"No Info": np.nan, "no info": np.nan, "NO INFO": np.nan})

    # Retain rows, but turn out-of-range clinical observations into missing
    # values for training-only median imputation. This avoids silently clipping.
    invalid_counts: dict[str, int] = {}
    for col, (low, high) in invalid_rules.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        invalid = numeric.notna() & ((numeric < low) | (numeric > high))
        invalid_counts[col] = int(invalid.sum())
        df[col] = numeric.mask(invalid)
    for col in zero_missing or []:
        invalid_counts[f"{col} (zero treated as missing)"] = int((df[col] == 0).sum())
        df[col] = df[col].mask(df[col] == 0)

    # Labels must remain valid and binary; duplicates are exact repeated records.
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    df[target] = pd.to_numeric(df[target], errors="raise").astype("int8")
    features = [c for c in df.columns if c != target]
    # Unknown category values remain identifiable and are handled through the
    # train-fitted imputer/encoder; do not discard rows for them.
    categorical = [c for c in categorical if c in features]
    before_impute = _counts(df, features)
    splits = _split(df, target)
    train, validation, test = splits
    transformer = _make_transformer(train, features, categorical)
    transformer.fit(train[features])  # Only training records determine statistics/mappings.
    # Export imputed source-scale splits with original columns/categories. These
    # fill values come only from training; scaling/encoding stays in the pickle.
    numeric_cols = [c for c in features if c not in categorical]
    numeric_stats = dict(zip(numeric_cols, transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_.tolist()))
    categorical_stats = (dict(zip(categorical, transformer.named_transformers_["categorical"].named_steps["imputer"].statistics_.tolist()))
                         if categorical else {})
    imputation_values: dict[str, Any] = {**numeric_stats, **categorical_stats}
    splits = tuple(part.fillna(value=imputation_values) for part in splits)
    train, validation, test = splits

    out = ROOT / "processed" / name.lower()
    out.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in zip(("train", "validation", "test"), splits):
        split_df.to_csv(out / f"{split_name}.csv", index=False)
    with (out / "preprocessor.pkl").open("wb") as f:
        pickle.dump(transformer, f)
    with (out / "preprocessing_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"dataset": name, "target": target, "feature_columns": features,
                   "categorical_columns": categorical, "numerical_columns": [c for c in features if c not in categorical],
                   "seed": SEED, "split_proportions": {"train": .70, "validation": .15, "test": .15},
                   "csv_values_are_unscaled": True,
                   "imputation_values_from_train": {k: (v.item() if hasattr(v, "item") else v) for k, v in imputation_values.items()},
                   "note": "CSV splits preserve source units/categories and are imputed with train-only statistics. preprocessor.pkl is fitted on train only for reusable impute/encode/scale."}, f, indent=2)

    after_counts = {k: _counts(v, features) for k, v in zip(("train", "validation", "test"), splits)}
    report = [inspection, "\nPREPROCESSING REPORT", f"Original shape: {original_shape}", f"Cleaned shape after exact duplicate removal: {df.shape}",
              f"Exact duplicate rows removed: {duplicate_count}", f"Invalid/out-of-range values converted to missing by column: {invalid_counts}",
              f"Missing values after invalid/sentinel handling and before imputation: {before_impute}",
              "Missing values remaining in exported CSVs after train-only imputation: " + json.dumps(after_counts),
              f"Features retained: {features}", "Features removed: [] (target excluded from transformer inputs)",
              f"Target retained separately as binary 0/1: {target}", f"Target distribution before cleaning: {original_dist}",
              f"Target distribution after duplicate removal/cleaning: {_distribution(df, target)}",
              "Split sizes and target distributions:"]
    for split_name, split_df in zip(("train", "validation", "test"), splits):
        report.append(f"  {split_name}: n={len(split_df)}, target={_distribution(split_df, target)}")
    report += ["\nImputation/encoding/scaling decisions:",
               "- Missing numeric feature values are imputed with the training split median in preprocessor.pkl.",
               "- Missing categorical feature values are imputed with the training split most frequent category.",
               "- Categorical features are one-hot encoded with unknown categories ignored; category vocabulary is learned from training only.",
               "- Numeric features are standardized using training-only mean and standard deviation in preprocessor.pkl.",
               f"- Exported split CSVs retain source feature names/categories and units, with train-only imputation values applied: {imputation_values}.",
               "- The saved transformer is fitted on the unscaled training features and retains reusable imputation/encoding/scaling operations; no target is passed to it.",
               "- No records are removed for invalid measurements; invalid observations become missing and are imputed at transform time.",
               "- Exact duplicate rows are removed before splitting to reduce leakage from repeated records."]
    if name.lower() == "ehr":
        report += ["- smoking_history 'No Info' is treated as missing, then imputed from training-set mode.",
                   "- EHR accepted ranges used for quality screening (outside -> missing): age 0–120 years; BMI 10–100 kg/m²; HbA1c 3–20%; blood glucose 20–1000 mg/dL; hypertension/heart_disease 0 or 1.",
                   "- Gender and smoking_history remain categorical in CSV and are one-hot encoded only by the reusable transformer."]
    else:
        report += ["- Pima zero values are treated as missing for Glucose, BloodPressure, SkinThickness, Insulin and BMI because zero is not a physiologically plausible measured value for these fields in this dataset.",
                   "- Pregnancies=0 is retained as a valid value.",
                   "- Pima screening ranges (outside -> missing): Pregnancies 0–30; Glucose 20–400; BloodPressure 20–200; SkinThickness 1–100; Insulin 1–1000; BMI 10–80; DiabetesPedigreeFunction >0 and <=10; Age 18–120."]
    (out / "preprocessing_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote {name} outputs to {out}")


def main() -> None:
    _prepare("ehr", ROOT / "dataset" / "diabetes_prediction_dataset.csv", "diabetes",
             {"age": (0, 120), "bmi": (10, 100), "HbA1c_level": (3, 20),
              "blood_glucose_level": (20, 1000), "hypertension": (0, 1), "heart_disease": (0, 1)},
             ["gender", "smoking_history"],
             normalize_categories={"gender": {"Other ": "Other"}})
    _prepare("pima", ROOT / "dataset" / "pima_diabetes.csv", "Outcome",
             {"Pregnancies": (0, 30), "Glucose": (20, 400), "BloodPressure": (20, 200),
              "SkinThickness": (1, 100), "Insulin": (1, 1000), "BMI": (10, 80),
              "DiabetesPedigreeFunction": (0, 10), "Age": (18, 120)},
             [], zero_missing=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"])


if __name__ == "__main__":
    main()
