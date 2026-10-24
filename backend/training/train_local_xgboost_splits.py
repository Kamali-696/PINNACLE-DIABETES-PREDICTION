"""Train independent XGBoost classifiers from existing processed train/validation/test splits.

Run from the project root with:
    python backend\training\train_local_xgboost_splits.py

The fitted preprocessing artifacts created by preprocess_diabetes.py are reused without refitting.
Each dataset is trained and evaluated independently. The test split is not used until fitting,
early stopping, and validation-threshold selection are complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "processed"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "local_model_results"
RANDOM_SEED = 42

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 700,
    "learning_rate": 0.04,
    "max_depth": 4,
    "min_child_weight": 2,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    expected_target: str


DATASETS = (
    DatasetSpec("ehr", "diabetes"),
    DatasetSpec("pima", "Outcome"),
)


def class_counts(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def read_metadata(spec: DatasetSpec) -> tuple[dict[str, Any], list[str]]:
    metadata_path = PROCESSED_DIR / spec.name / "preprocessing_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target = metadata.get("target")
    if target != spec.expected_target:
        raise ValueError(
            f"{spec.name}: metadata target is {target!r}; expected {spec.expected_target!r}."
        )
    features = metadata.get("feature_columns")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{spec.name}: metadata has no usable feature_columns list.")
    if target in features:
        raise ValueError(f"{spec.name}: target {target!r} appears in input feature_columns.")
    return metadata, features


def read_split(spec: DatasetSpec, split: str, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    path = PROCESSED_DIR / spec.name / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split} split for {spec.name}: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in [*features, target] if column not in frame.columns]
    if missing:
        raise ValueError(f"{spec.name} {split}: missing required columns {missing} in {path}.")
    X = frame.loc[:, features]
    y_raw = frame[target]
    if y_raw.isna().any():
        raise ValueError(f"{spec.name} {split}: target contains missing values.")
    observed = set(pd.unique(y_raw))
    if not observed.issubset({0, 1, 0.0, 1.0}) or observed != {0, 1}:
        raise ValueError(f"{spec.name} {split}: expected both binary target classes 0 and 1; got {observed}.")
    y = y_raw.astype(np.int8)
    return X, y


def select_f1_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Select an F1-maximizing cutoff from validation scores only; ties prefer 0.5."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    candidates = list(float(value) for value in thresholds)
    candidates.append(0.5)
    scored = []
    for threshold in sorted(set(candidates)):
        predictions = (probability >= threshold).astype(np.int8)
        scored.append((float(f1_score(y_true, predictions, zero_division=0)), threshold))
    best_f1 = max(item[0] for item in scored)
    tied = [threshold for score, threshold in scored if np.isclose(score, best_f1)]
    chosen = min(tied, key=lambda value: (abs(value - 0.5), value))
    return chosen, {
        "method": "maximize validation F1; ties choose threshold closest to 0.5",
        "validation_best_f1": best_f1,
        "candidate_threshold_count": len(set(candidates)),
    }


def evaluate(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "sample_count": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "classification_report": classification_report(
            y_true,
            prediction,
            labels=[0, 1],
            target_names=["negative (0)", "positive (1)"],
            output_dict=True,
            zero_division=0,
        ),
        "classification_report_text": classification_report(
            y_true,
            prediction,
            labels=[0, 1],
            target_names=["negative (0)", "positive (1)"],
            zero_division=0,
        ),
    }


def fit_with_early_stopping(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_validation: np.ndarray,
    y_validation: pd.Series,
    params: dict[str, Any],
    early_stopping_rounds: int,
    scale_pos_weight: float,
) -> XGBClassifier:
    model_params = dict(params)
    model_params["scale_pos_weight"] = scale_pos_weight
    model_params["early_stopping_rounds"] = early_stopping_rounds
    model = XGBClassifier(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    return model


def train_one(
    spec: DatasetSpec,
    dataset_out: Path,
    params: dict[str, Any],
    early_stopping_rounds: int,
    tune_threshold: bool,
) -> dict[str, Any]:
    metadata, features = read_metadata(spec)
    target = metadata["target"]
    preprocessor_path = PROCESSED_DIR / spec.name / "preprocessor.pkl"
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Missing fitted preprocessing artifact: {preprocessor_path}")

    # These preprocessor artifacts were fitted on the preprocessing training split.
    # Reuse them without fitting or updating their imputation, scaling, or encoding state.
    preprocessor = joblib.load(preprocessor_path)
    X_train, y_train = read_split(spec, "train", features, target)
    X_validation, y_validation = read_split(spec, "validation", features, target)

    counts = class_counts(y_train)
    negatives = counts.get("0", 0)
    positives = counts.get("1", 0)
    if negatives == 0 or positives == 0:
        raise ValueError(f"{spec.name}: training split needs both classes for class weighting.")
    scale_pos_weight = negatives / positives

    X_train_transformed = preprocessor.transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)
    model = fit_with_early_stopping(
        X_train_transformed,
        y_train,
        X_validation_transformed,
        y_validation,
        params,
        early_stopping_rounds,
        scale_pos_weight,
    )

    validation_probability = model.predict_proba(X_validation_transformed)[:, 1]
    if tune_threshold:
        threshold, threshold_selection = select_f1_threshold(y_validation, validation_probability)
    else:
        threshold = 0.5
        threshold_selection = {"method": "fixed default", "validation_best_f1": None}
    validation_metrics = evaluate(y_validation, validation_probability, threshold)

    # Test data is first loaded and evaluated after training, early stopping, and threshold
    # selection have completed. The already-fitted preprocessing artifact is transform-only.
    X_test, y_test = read_split(spec, "test", features, target)
    X_test_transformed = preprocessor.transform(X_test)
    test_probability = model.predict_proba(X_test_transformed)[:, 1]
    test_metrics = evaluate(y_test, test_probability, threshold)

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    joblib.dump(pipeline, dataset_out / "model.joblib")
    (dataset_out / "classification_report.txt").write_text(
        test_metrics["classification_report_text"], encoding="utf-8"
    )
    report = {
        "dataset": spec.name,
        "target": target,
        "features": features,
        "training_samples": int(len(y_train)),
        "training_class_counts": counts,
        "scale_pos_weight": float(scale_pos_weight),
        "validation_samples": int(len(y_validation)),
        "validation_class_counts": class_counts(y_validation),
        "test_samples": int(len(y_test)),
        "test_class_counts": class_counts(y_test),
        "threshold": float(threshold),
        "threshold_selection": threshold_selection,
        "early_stopping_rounds": int(early_stopping_rounds),
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators - 1)),
        "best_score": float(model.best_score) if getattr(model, "best_score", None) is not None else None,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "preprocessing_artifact": str(preprocessor_path.relative_to(PROJECT_DIR)),
        "original_processed_splits_unchanged": True,
    }
    (dataset_out / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (dataset_out / "training_metadata.json").write_text(
        json.dumps(
            {
                "status": "success",
                "xgboost_parameters": {**params, "scale_pos_weight": scale_pos_weight,
                                        "early_stopping_rounds": early_stopping_rounds},
                "versions": {
                    "python": platform.python_version(),
                    "xgboost": xgboost.__version__,
                    "scikit_learn": sklearn.__version__,
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "joblib": joblib.__version__,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def unique_run_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"run_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"run_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_PARAMS["n_estimators"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    parser.add_argument("--max-depth", type=int, default=DEFAULT_PARAMS["max_depth"])
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--fixed-threshold", action="store_true", help="Keep threshold at 0.5 instead of validation F1 selection.")
    args = parser.parse_args()
    if args.n_estimators < 1 or args.early_stopping_rounds < 1 or args.max_depth < 1:
        parser.error("n_estimators, early_stopping_rounds, and max_depth must be positive.")

    params = dict(DEFAULT_PARAMS)
    params.update({
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
    })
    run_dir = unique_run_dir(args.output_root)
    log_path = run_dir / "training.log"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers, force=True)
    log = logging.getLogger("local_xgboost")

    configuration = {
        "project": "PINNACLE_DIABETES_DETECTION",
        "run_started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": RANDOM_SEED,
        "output_directory": str(run_dir),
        "processed_directory": str(PROCESSED_DIR),
        "datasets": [spec.name for spec in DATASETS],
        "xgboost_parameters": {**params, "early_stopping_rounds": args.early_stopping_rounds},
        "threshold_policy": "validation F1 tuning" if not args.fixed_threshold else "fixed 0.5",
        "test_split_policy": "test is only loaded after fit and validation threshold selection",
        "class_weight_policy": "negative training count / positive training count",
        "versions": {
            "python": platform.python_version(), "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "joblib": joblib.__version__,
        },
    }
    (run_dir / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    log.info("Local XGBoost training run: %s", run_dir)
    log.info("Runtime versions: %s", json.dumps(configuration["versions"], sort_keys=True))

    results: dict[str, dict[str, Any]] = {}
    for spec in DATASETS:
        dataset_out = run_dir / spec.name
        dataset_out.mkdir(parents=True, exist_ok=False)
        log.info("\n=== %s ===", spec.name.upper())
        try:
            result = train_one(
                spec,
                dataset_out,
                params,
                args.early_stopping_rounds,
                tune_threshold=not args.fixed_threshold,
            )
            results[spec.name] = {"status": "success", **result}
            metrics = result["test_metrics"]
            log.info("Training status: SUCCESS")
            log.info("Training samples: %d", result["training_samples"])
            log.info("Training classes: negative=%d positive=%d", result["training_class_counts"]["0"], result["training_class_counts"]["1"])
            log.info("scale_pos_weight: %.8f", result["scale_pos_weight"])
            log.info("Selected threshold: %.6f (%s)", result["threshold"], result["threshold_selection"]["method"])
            log.info("Test metrics: accuracy=%.4f precision=%.4f recall=%.4f F1=%.4f ROC-AUC=%.4f",
                     metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"])
            log.info("Confusion matrix [ [TN, FP], [FN, TP] ]: [[%d, %d], [%d, %d]]",
                     metrics["confusion_matrix"]["tn"], metrics["confusion_matrix"]["fp"],
                     metrics["confusion_matrix"]["fn"], metrics["confusion_matrix"]["tp"])
            log.info("Test classification report:\n%s", metrics["classification_report_text"])
        except Exception as exc:
            logging.exception("Training status: FAILED for %s: %s", spec.name, exc)
            results[spec.name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            (dataset_out / "error.txt").write_text(results[spec.name]["error"], encoding="utf-8")

    comparison = {
        name: ({
            "status": result["status"],
            "test_metrics": {
                key: result["test_metrics"][key]
                for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix")
            } if result["status"] == "success" else None,
            "error": result.get("error"),
        })
        for name, result in results.items()
    }
    (run_dir / "comparison_report.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    log.info("\n=== EHR vs PIMA TEST COMPARISON ===")
    for name, entry in comparison.items():
        if entry["status"] == "success":
            metrics = entry["test_metrics"]
            log.info("%s: accuracy=%.4f precision=%.4f recall=%.4f F1=%.4f ROC-AUC=%.4f confusion=%s",
                     name.upper(), metrics["accuracy"], metrics["precision"], metrics["recall"],
                     metrics["f1"], metrics["roc_auc"], metrics["confusion_matrix"])
        else:
            log.info("%s: FAILED — %s", name.upper(), entry["error"])
    log.info("Results saved to: %s", run_dir)
    return 0 if all(result["status"] == "success" for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
