"""Train the XGBoost diabetes classifier and write artifacts for the API.

Run from the backend directory:
    python -m training.train_xgboost

Leakage controls:
- Exact duplicate rows are removed BEFORE splitting, so identical records cannot sit in
  both the training and the test set.
- The test set is split off first and is used exactly once, for final evaluation.
- Encoders are fitted on the training-fit split only; early stopping uses a validation
  split carved out of the training data, never the test set.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_DATA = PROJECT_DIR / "dataset" / "diabetes_prediction_dataset.csv"
DEFAULT_OUT = BACKEND_DIR / "model_artifacts" / "xgboost"

RANDOM_SEED = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10  # fraction of the training split used for early stopping
DECISION_THRESHOLD = 0.5

TARGET = "diabetes"
CATEGORICAL_FEATURES = ["gender", "smoking_history"]
NUMERIC_FEATURES = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
FEATURES = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]

XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 5,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stopping_rounds": 50,
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "categorical_levels": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
    }


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def class_counts(y) -> dict:
    s = pd.Series(y).value_counts().sort_index()
    return {str(k): int(v) for k, v in s.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    missing_cols = set(FEATURES + [TARGET]) - set(raw.columns)
    if missing_cols:
        raise SystemExit(f"Dataset is missing expected columns: {sorted(missing_cols)}")

    raw_profile = profile(raw)
    df = raw.drop_duplicates().reset_index(drop=True)
    if df[FEATURES + [TARGET]].isna().any().any():
        raise SystemExit("Unexpected missing values; add an imputation step before training.")

    X, y = df[FEATURES], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE, stratify=y_train, random_state=RANDOM_SEED
    )

    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor.fit(X_fit)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(preprocessor.transform(X_fit), y_fit, eval_set=[(preprocessor.transform(X_val), y_val)], verbose=False)

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    metrics = {
        "decision_threshold": DECISION_THRESHOLD,
        "training": evaluate(y_fit, pipeline.predict_proba(X_fit)[:, 1], DECISION_THRESHOLD),
        "validation": evaluate(y_val, pipeline.predict_proba(X_val)[:, 1], DECISION_THRESHOLD),
        "test": evaluate(y_test, pipeline.predict_proba(X_test)[:, 1], DECISION_THRESHOLD),
    }

    numeric_ranges = {
        c: {"min": float(X_fit[c].min()), "max": float(X_fit[c].max())} for c in NUMERIC_FEATURES
    }
    transformed_names = list(preprocessor.get_feature_names_out())
    importances = dict(zip(transformed_names, (float(v) for v in model.feature_importances_)))

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "model_name": "XGBoost diabetes classifier",
        "trained_at": trained_at,
        "target": TARGET,
        "positive_label": 1,
        "feature_order": FEATURES,
        "categorical_features": {
            c: [str(v) for v in cats] for c, cats in zip(CATEGORICAL_FEATURES, preprocessor.named_transformers_["categorical"].categories_)
        },
        "numeric_features": NUMERIC_FEATURES,
        "training_numeric_ranges": numeric_ranges,
        "transformed_feature_order": transformed_names,
        "decision_threshold": DECISION_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "best_iteration": int(model.best_iteration),
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgboost.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    report = {
        "model_name": metadata["model_name"],
        "trained_at": trained_at,
        "dataset": {
            "file": args.data.name,
            "raw": raw_profile,
            "duplicates_removed": int(len(raw) - len(df)),
            "rows_after_deduplication": int(len(df)),
            "class_counts_after_deduplication": class_counts(y),
        },
        "split": {
            "method": "stratified",
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
            "validation_size_of_train": VALIDATION_SIZE,
            "class_counts": {
                "training": class_counts(y_fit),
                "validation": class_counts(y_val),
                "test": class_counts(y_test),
            },
        },
        "metrics": metrics,
        "feature_importance_gain": importances,
        "best_iteration": metadata["best_iteration"],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.out / "diabetes_xgb_pipeline.joblib")
    (args.out / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    (args.out / "evaluation_report.json").write_text(json.dumps(report, indent=2))

    print(f"Rows: {len(raw)} raw, {len(df)} after removing {len(raw) - len(df)} duplicates")
    print(f"Split: train-fit {len(X_fit)}, validation {len(X_val)}, test {len(X_test)} (best_iteration={model.best_iteration})")
    for split in ("training", "validation", "test"):
        m = metrics[split]
        print(
            f"{split:>10}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
            f"f1={m['f1']:.4f} auc={m['roc_auc']:.4f} cm={m['confusion_matrix']}"
        )
    print(f"Artifacts written to {args.out}")


if __name__ == "__main__":
    main()
