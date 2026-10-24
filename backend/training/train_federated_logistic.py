"""Run a simulated two-node FedAvg experiment on the processed EHR and Pima splits.

Run from the project root:
    python backend\training\train_federated_logistic.py

Only Age and BMI are shared because these are the only features with matching meanings and
units documented for both sources. The processed CSVs already contain train-only imputation
and remain in source units. Shared deterministic scaling (years / 100, BMI kg/m^2 / 100)
keeps both nodes on the same parameter coordinate system without fitting a scaler on either
evaluation split or sharing training records.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "processed"
LOCAL_RUN_DIR = PROJECT_DIR / "local_model_results" / "run_20261024T003644Z"
OUTPUT_ROOT = PROJECT_DIR / "federated_results"
SEED = 42
NODE_SPECS = {
    "ehr": {"target": "diabetes", "age_column": "age", "bmi_column": "bmi"},
    "pima": {"target": "Outcome", "age_column": "Age", "bmi_column": "BMI"},
}
SHARED_FEATURES = ["age_years_scaled", "bmi_kg_m2_scaled"]
SCALE_DIVISORS = {"age_years_scaled": 100.0, "bmi_kg_m2_scaled": 100.0}
DEFAULT_CONFIG = {
    "communication_rounds": 50,
    "local_epochs": 5,
    "learning_rate": 0.05,
    "local_logistic_C": 1.0,
    "threshold_selection": "validation_f1",
}


def counts(y: np.ndarray) -> dict[str, int]:
    values, frequencies = np.unique(y, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(values, frequencies)}


def load_node(node: str) -> dict[str, Any]:
    spec = NODE_SPECS[node]
    metadata_path = PROCESSED_DIR / node / "preprocessing_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("target") != spec["target"]:
        raise ValueError(f"{node}: unexpected target in metadata; expected {spec['target']!r}.")
    declared_features = metadata.get("feature_columns", [])
    for col in (spec["age_column"], spec["bmi_column"]):
        if col not in declared_features:
            raise ValueError(f"{node}: required shared feature {col!r} is absent from metadata.")
    if spec["target"] in (spec["age_column"], spec["bmi_column"]):
        raise ValueError(f"{node}: target cannot be an input feature.")

    node_data: dict[str, Any] = {"spec": spec, "splits": {}}
    for split in ("train", "validation", "test"):
        path = PROCESSED_DIR / node / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {node} {split} split: {path}")
        frame = pd.read_csv(path)
        needed = [spec["age_column"], spec["bmi_column"], spec["target"]]
        missing = [column for column in needed if column not in frame]
        if missing:
            raise ValueError(f"{node} {split}: missing required columns {missing}.")
        y = frame[spec["target"]].to_numpy()
        if pd.isna(y).any() or set(np.unique(y)) != {0, 1}:
            raise ValueError(f"{node} {split}: target must contain nonmissing binary labels 0 and 1.")
        # Exported processed CSVs are in source units and carry imputation values learned from
        # the training split. Do not fit or refit any transformer in this federated experiment.
        raw_X = frame[[spec["age_column"], spec["bmi_column"]]].apply(pd.to_numeric, errors="raise")
        if raw_X.isna().any().any() or not np.isfinite(raw_X.to_numpy(dtype=float)).all():
            raise ValueError(f"{node} {split}: shared inputs contain missing/nonfinite values.")
        x = np.column_stack([
            raw_X[spec["age_column"]].to_numpy(dtype=np.float64) / SCALE_DIVISORS["age_years_scaled"],
            raw_X[spec["bmi_column"]].to_numpy(dtype=np.float64) / SCALE_DIVISORS["bmi_kg_m2_scaled"],
        ])
        node_data["splits"][split] = {"X": x, "y": y.astype(np.int64), "rows": int(len(frame))}
    node_data["feature_mapping"] = {
        "age_years_scaled": {"source_column": spec["age_column"], "unit": "years", "transformation": "value / 100.0"},
        "bmi_kg_m2_scaled": {"source_column": spec["bmi_column"], "unit": "kg/m^2", "transformation": "value / 100.0"},
    }
    return node_data


def choose_threshold(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, dict[str, Any]]:
    _, _, candidates = precision_recall_curve(y_true, probability)
    values = sorted(set([0.5, *(float(item) for item in candidates)]))
    scores = [(float(f1_score(y_true, probability >= value, zero_division=0)), value) for value in values]
    best = max(score for score, _ in scores)
    threshold = min((value for score, value in scores if np.isclose(score, best)), key=lambda value: (abs(value - 0.5), value))
    return threshold, {
        "method": "maximize validation F1; ties select threshold closest to 0.5",
        "validation_best_f1": best,
        "candidate_threshold_count": len(values),
    }


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y_true, prediction, labels=[0, 1]).ravel())
    return {
        "sample_count": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def weighted_log_loss(X: np.ndarray, y: np.ndarray, parameters: np.ndarray, positive_weight: float) -> float:
    logits = np.clip(X @ parameters[:-1] + parameters[-1], -40.0, 40.0)
    row_loss = np.logaddexp(0.0, logits) - y * logits
    row_weights = np.where(y == 1, positive_weight, 1.0)
    return float(np.mean(row_loss * row_weights))


def local_fedavg_train(
    node_name: str,
    X: np.ndarray,
    y: np.ndarray,
    global_parameters: np.ndarray,
    positive_weight: float,
    learning_rate: float,
    local_epochs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Perform local full-batch SGD updates and return model parameters only."""
    parameters = np.asarray(global_parameters, dtype=np.float64).copy()
    n_records = len(y)
    for _ in range(local_epochs):
        logits = np.clip(X @ parameters[:-1] + parameters[-1], -40.0, 40.0)
        probabilities = stable_sigmoid(logits)
        residual = probabilities - y
        residual *= np.where(y == 1, positive_weight, 1.0)
        gradient_weights = (X.T @ residual) / n_records
        gradient_bias = float(residual.mean())
        parameters[:-1] -= learning_rate * gradient_weights
        parameters[-1] -= learning_rate * gradient_bias
    return parameters, {
        "node": node_name,
        "status": "success",
        "training_sample_count": int(n_records),
        "post_local_train_weighted_log_loss": weighted_log_loss(X, y, parameters, positive_weight),
        "parameter_update_l2_norm": float(np.linalg.norm(parameters - global_parameters)),
    }


def probability_from_parameters(X: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    return stable_sigmoid(X @ parameters[:-1] + parameters[-1])


def fit_local_lr(X: np.ndarray, y: np.ndarray, pos_weight: float, C: float) -> LogisticRegression:
    model = LogisticRegression(
        C=C,
        class_weight={0: 1.0, 1: float(pos_weight)},
        max_iter=3000,
        solver="lbfgs",
        random_state=SEED,
    )
    model.fit(X, y)
    return model


def baseline_xgb_metrics(node: str) -> dict[str, Any]:
    report_path = LOCAL_RUN_DIR / node / "evaluation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Saved local XGBoost baseline report missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        "model": "local XGBoost baseline",
        "threshold": report["threshold"],
        "test_metrics": report["test_metrics"],
        "source_report": str(report_path.relative_to(PROJECT_DIR)),
        "metrics_reused_from_saved_test_evaluation": True,
    }


def unique_run_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    path = root / stamp
    suffix = 1
    while path.exists():
        path = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=DEFAULT_CONFIG["communication_rounds"])
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_CONFIG["local_epochs"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--local-logistic-C", type=float, default=DEFAULT_CONFIG["local_logistic_C"])
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if min(args.rounds, args.local_epochs) < 1 or args.learning_rate <= 0 or args.local_logistic_C <= 0:
        parser.error("rounds, local-epochs, learning-rate, and local-logistic-C must be positive.")

    run_dir = unique_run_dir(args.output_root)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "training.log", encoding="utf-8")],
        force=True,
    )
    log = logging.getLogger("federated_logistic")
    config = {
        "experiment": "simulated_two_node_federated_binary_logistic_regression",
        "aggregation": "sample-count-weighted FedAvg over model coefficients and intercept",
        "communication_rounds": args.rounds,
        "local_epochs_per_round": args.local_epochs,
        "local_optimizer": "full-batch gradient descent on class-weighted binary cross-entropy",
        "learning_rate": args.learning_rate,
        "local_lr_baseline_C": args.local_logistic_C,
        "threshold_selection": "per-node validation F1 maximization, ties closest to 0.5",
        "seed": SEED,
        "nodes": list(NODE_SPECS),
        "shared_feature_order": SHARED_FEATURES,
        "device": "local process / CPU NumPy; node functions exchange parameters only",
        "privacy_claim": "none; simulated FL only, no secure aggregation, encryption, or differential privacy",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log.info("Federated results: %s", run_dir)
    log.info("Shared feature schema: %s", SHARED_FEATURES)
    log.info("Mapping: Age (years) / 100; BMI (kg/m^2) / 100; common fixed scaling, no fitted scaler.")
    log.info("Nodes: EHR and Pima; target labels stay local and are not model inputs.")
    nodes = {name: load_node(name) for name in NODE_SPECS}
    node_summary: dict[str, Any] = {}
    local_models: dict[str, LogisticRegression] = {}
    local_model_reports: dict[str, Any] = {}
    xgb_baselines: dict[str, Any] = {}
    final_node_parameters: dict[str, Any] = {}

    for name, data in nodes.items():
        train = data["splits"]["train"]
        validation = data["splits"]["validation"]
        y_counts = counts(train["y"])
        positive_weight = y_counts["0"] / y_counts["1"]
        data["positive_weight"] = positive_weight
        log.info("\nNode %s: train n=%d; negatives=%d; positives=%d; positive-class weight=%.8f",
                 name.upper(), len(train["y"]), y_counts["0"], y_counts["1"], positive_weight)
        # A separate single-node baseline is fit only on that node's training partition.
        local_model = fit_local_lr(train["X"], train["y"], positive_weight, args.local_logistic_C)
        local_models[name] = local_model
        validation_prob = local_model.predict_proba(validation["X"])[:, 1]
        threshold, selection = choose_threshold(validation["y"], validation_prob)
        local_model_reports[name] = {
            "threshold": threshold,
            "threshold_selection": selection,
            "validation_metrics": metrics(validation["y"], validation_prob, threshold),
            "training_class_counts": y_counts,
            "scale_pos_weight": positive_weight,
            "training_samples": int(len(train["y"])),
        }
        xgb_baselines[name] = baseline_xgb_metrics(name)
        node_summary[name] = {
            "training_sample_count": int(len(train["y"])),
            "training_class_counts": y_counts,
            "scale_pos_weight": float(positive_weight),
            "feature_mapping": data["feature_mapping"],
            "local_lr_threshold": float(threshold),
            "local_lr_threshold_selection": selection,
            "local_lr_validation_metrics": local_model_reports[name]["validation_metrics"],
        }

    # Both nodes start each federated round with an identical shared logistic-regression vector.
    # Parameter vector layout: [coefficient(age), coefficient(BMI), intercept].
    global_parameters = np.zeros(len(SHARED_FEATURES) + 1, dtype=np.float64)
    round_history: list[dict[str, Any]] = []
    train_sizes = {name: len(data["splits"]["train"]["y"]) for name, data in nodes.items()}
    total_train_size = sum(train_sizes.values())
    for round_number in range(1, args.rounds + 1):
        node_parameters: dict[str, np.ndarray] = {}
        round_nodes: dict[str, Any] = {}
        for name, data in nodes.items():
            local_params, node_log = local_fedavg_train(
                node_name=name,
                X=data["splits"]["train"]["X"],
                y=data["splits"]["train"]["y"],
                global_parameters=global_parameters,
                positive_weight=data["positive_weight"],
                learning_rate=args.learning_rate,
                local_epochs=args.local_epochs,
            )
            node_parameters[name] = local_params
            round_nodes[name] = node_log
        # The server receives model parameters and sample counts, never node record arrays.
        global_parameters = sum(train_sizes[name] * node_parameters[name] for name in nodes) / total_train_size
        round_history.append({
            "round": round_number,
            "aggregation": "sample-count-weighted FedAvg",
            "aggregation_weights": train_sizes,
            "nodes": round_nodes,
            "global_parameters": {"coefficients": global_parameters[:-1].tolist(), "intercept": float(global_parameters[-1])},
        })
        log.info("Round %d/%d: EHR=%s loss=%.6f update_norm=%.6f | Pima=%s loss=%.6f update_norm=%.6f",
                 round_number, args.rounds,
                 round_nodes["ehr"]["status"], round_nodes["ehr"]["post_local_train_weighted_log_loss"], round_nodes["ehr"]["parameter_update_l2_norm"],
                 round_nodes["pima"]["status"], round_nodes["pima"]["post_local_train_weighted_log_loss"], round_nodes["pima"]["parameter_update_l2_norm"])
        final_node_parameters = {name: values.copy() for name, values in node_parameters.items()}

    # Each node chooses an operating threshold on its own validation split. This keeps
    # validation records local and acknowledges that the two datasets have different label
    # prevalence and score distributions, while the underlying model parameters stay shared.
    global_validation_probabilities = {
        name: probability_from_parameters(data["splits"]["validation"]["X"], global_parameters)
        for name, data in nodes.items()
    }
    global_thresholds: dict[str, float] = {}
    global_threshold_selections: dict[str, Any] = {}
    for name, data in nodes.items():
        validation_split = data["splits"]["validation"]
        global_thresholds[name], global_threshold_selections[name] = choose_threshold(
            validation_split["y"], global_validation_probabilities[name]
        )

    evaluation: dict[str, Any] = {}
    for name, data in nodes.items():
        test = data["splits"]["test"]
        local_threshold = local_model_reports[name]["threshold"]
        local_test_probability = local_models[name].predict_proba(test["X"])[:, 1]
        global_test_probability = probability_from_parameters(test["X"], global_parameters)
        local_test_metrics = metrics(test["y"], local_test_probability, local_threshold)
        global_threshold = global_thresholds[name]
        global_test_metrics = metrics(test["y"], global_test_probability, global_threshold)
        local_model_reports[name]["test_metrics"] = local_test_metrics
        xgb_baselines[name]["model"] = "local XGBoost baseline"
        evaluation[name] = {
            "test_sample_count": int(len(test["y"])),
            "test_class_counts": counts(test["y"]),
            "local_xgboost_baseline": xgb_baselines[name],
            "local_logistic_regression": local_model_reports[name],
            "federated_global_logistic_regression": {
                "threshold": global_threshold,
                "threshold_selection": global_threshold_selections[name],
                "validation_metrics": metrics(data["splits"]["validation"]["y"], global_validation_probabilities[name], global_threshold),
                "test_metrics": global_test_metrics,
            },
        }
        log.info("\n%s test metrics:", name.upper())
        for model_name, model_result in (
            ("local XGBoost", evaluation[name]["local_xgboost_baseline"]["test_metrics"]),
            ("local logistic regression", local_test_metrics),
            ("federated global logistic regression", global_test_metrics),
        ):
            log.info("  %-34s acc=%.4f precision=%.4f recall=%.4f F1=%.4f ROC-AUC=%.4f confusion=%s",
                     model_name, model_result["accuracy"], model_result["precision"], model_result["recall"],
                     model_result["f1"], model_result["roc_auc"], model_result["confusion_matrix"])

    feature_mapping = {
        "shared_feature_order": SHARED_FEATURES,
        "features": {
            "age_years_scaled": {
                "EHR_source_column": "age", "Pima_source_column": "Age", "definition": "participant age",
                "unit": "years", "transformation": "divide raw source-scale age by fixed public constant 100.0",
            },
            "bmi_kg_m2_scaled": {
                "EHR_source_column": "bmi", "Pima_source_column": "BMI", "definition": "body mass index",
                "unit": "kg/m^2", "transformation": "divide raw source-scale BMI by fixed public constant 100.0",
            },
        },
        "excluded_nonmatching_features": {
            "EHR": ["gender", "hypertension", "heart_disease", "smoking_history", "HbA1c_level", "blood_glucose_level"],
            "Pima": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction"],
            "reason": "No corresponding feature with a sufficiently equivalent definition exists in both schemas. In particular, Pima Glucose is a 2-hour oral glucose tolerance test measurement, while EHR blood_glucose_level is described as glucose at the time of measurement; these are not treated as interchangeable.",
        },
        "targets": {"ehr": "diabetes (0/1)", "pima": "Outcome (0/1)"},
        "preprocessing": "Reuse processed CSVs with train-only imputation already applied; use fixed shared unit scaling. No fitted transform is learned in this experiment.",
        "unit_compatibility_basis": "Project preprocessing reports list age in years and BMI in kg/m^2 for both datasets; source feature descriptions checked against Pima documentation.",
        "scope_note": "Two shared features are sufficient for a restricted FedAvg technical experiment but omit dataset-specific predictors and do not establish clinical utility.",
    }
    (run_dir / "shared_feature_mapping.json").write_text(json.dumps(feature_mapping, indent=2), encoding="utf-8")
    (run_dir / "round_history.json").write_text(json.dumps(round_history, indent=2), encoding="utf-8")
    (run_dir / "evaluation_report.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    global_checkpoint = {
        "model_type": "binary logistic regression",
        "feature_order": SHARED_FEATURES,
        "coefficients": global_parameters[:-1].tolist(),
        "intercept": float(global_parameters[-1]),
        "thresholds_by_node": {name: float(value) for name, value in global_thresholds.items()},
        "threshold_selection_by_node": global_threshold_selections,
        "aggregation": "sample-count-weighted FedAvg",
        "communication_rounds": args.rounds,
    }
    (run_dir / "global_model_parameters.json").write_text(json.dumps(global_checkpoint, indent=2), encoding="utf-8")
    joblib.dump(global_checkpoint, run_dir / "global_model.joblib")
    checkpoints_dir = run_dir / "node_checkpoints"
    checkpoints_dir.mkdir()
    for name in nodes:
        joblib.dump(local_models[name], checkpoints_dir / f"{name}_local_logistic_regression.joblib")
        np.savez(
            checkpoints_dir / f"{name}_last_round_parameters.npz",
            coefficients=final_node_parameters[name][:-1],
            intercept=final_node_parameters[name][-1],
            feature_order=np.asarray(SHARED_FEATURES),
        )
    (run_dir / "node_summary.json").write_text(json.dumps(node_summary, indent=2), encoding="utf-8")
    (run_dir / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log.info("\nFederated status: SUCCESS")
    log.info("Aggregation: sample-count-weighted FedAvg; rounds=%d; local epochs=%d", args.rounds, args.local_epochs)
    log.info("Final global parameters: coefficients=%s intercept=%.8f", global_parameters[:-1].tolist(), global_parameters[-1])
    log.info("Per-node thresholds selected on local validation: %s", global_thresholds)
    log.info("No raw data was serialized or included in server update artifacts. This is simulated FL without secure aggregation, encryption, or differential privacy.")
    log.info("Results directory: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
