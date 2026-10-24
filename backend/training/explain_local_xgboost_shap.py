r"""Explain saved local EHR and Pima XGBoost models with SHAP.

Run from the project root:
    python backend\training\explain_local_xgboost_shap.py

No model is retrained and no preprocessing object is refitted. SHAP is computed on each
saved model's transformed test features in raw-margin (log-odds) space. For EHR, SHAP values
for one-hot columns are summed back to their original categorical input feature, preserving
the additive explanation while restoring the original feature names.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_RUN_DIR = PROJECT_DIR / "local_model_results" / "run_20261024T003644Z"
PROCESSED_DIR = PROJECT_DIR / "processed"
OUTPUT_ROOT = PROJECT_DIR / "xai_results"
DATASET_NAMES = ("ehr", "pima")


def json_read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def unique_run_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def raw_shap_values(explainer: shap.TreeExplainer, matrix: np.ndarray) -> np.ndarray:
    """Normalize SHAP's version-dependent binary output to (rows, transformed_features)."""
    try:
        values = explainer.shap_values(matrix, check_additivity=False)
    except TypeError:
        values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        # Newer SHAP releases may return (rows, features, outputs).
        values = values[:, :, 1] if values.shape[-1] == 2 else values[:, :, 0]
    if values.ndim != 2 or values.shape[0] != matrix.shape[0]:
        raise ValueError(f"Unexpected SHAP value shape {values.shape}; input shape is {matrix.shape}.")
    return values.astype(np.float64, copy=False)


def group_indices(
    original_features: list[str], transformed_features: list[str]
) -> tuple[list[list[int]], dict[str, list[str]]]:
    groups: list[list[int]] = []
    mapping: dict[str, list[str]] = {}
    assigned: list[int] = []
    for feature in original_features:
        indices = [
            index for index, name in enumerate(transformed_features)
            if (
                name.split("__", 1)[-1] == feature
                or name.split("__", 1)[-1].startswith(feature + "_")
            )
        ]
        if not indices:
            raise ValueError(
                f"Cannot map transformed model columns to original feature {feature!r}. "
                f"Transformed columns: {transformed_features}"
            )
        groups.append(indices)
        assigned.extend(indices)
        mapping[feature] = [transformed_features[index] for index in indices]
    if sorted(assigned) != list(range(len(transformed_features))):
        raise ValueError("Transformed columns were not assigned exactly once to original features.")
    return groups, mapping


def plotting_feature_matrix(
    original_X: pd.DataFrame,
    metadata: dict[str, Any],
    pipeline: Pipeline,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Make numeric plot colors while preserving original categories in local explanations."""
    plot_frame = original_X.copy()
    category_codes: dict[str, dict[str, int]] = {}
    categorical = metadata.get("categorical_columns", [])
    categorical_pipe = pipeline.named_steps["preprocessor"].named_transformers_.get("categorical")
    encoder = categorical_pipe.named_steps.get("encoder") if categorical_pipe is not None else None
    if categorical and encoder is None:
        raise ValueError("Metadata declares categorical features but the saved encoder is unavailable.")
    levels_by_column = dict(zip(categorical, encoder.categories_)) if categorical else {}
    for feature in categorical:
        levels = [str(level) for level in levels_by_column[feature]]
        code_map = {level: index for index, level in enumerate(levels)}
        plot_frame[feature] = original_X[feature].astype(str).map(code_map).fillna(-1).astype(float)
        category_codes[feature] = code_map
    try:
        values = plot_frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Could not create a numeric feature-value matrix for SHAP plotting.") from exc
    return values, category_codes


def choose_cases(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> list[dict[str, Any]]:
    predicted = probabilities >= threshold
    masks = {
        "correctly_predicted_positive": (y_true == 1) & predicted,
        "correctly_predicted_negative": (y_true == 0) & (~predicted),
        "false_negative": (y_true == 1) & (~predicted),
        "false_positive": (y_true == 0) & predicted,
    }
    selected: list[dict[str, Any]] = []
    # Choose confident correctly classified examples and near-threshold errors as readable cases.
    for case_type, mask in masks.items():
        candidate_indices = np.flatnonzero(mask)
        if len(candidate_indices) == 0:
            continue
        if case_type == "correctly_predicted_positive":
            index = int(candidate_indices[np.argmax(probabilities[candidate_indices])])
        elif case_type == "correctly_predicted_negative":
            index = int(candidate_indices[np.argmin(probabilities[candidate_indices])])
        elif case_type == "false_negative":
            index = int(candidate_indices[np.argmax(probabilities[candidate_indices])])
        else:
            index = int(candidate_indices[np.argmin(probabilities[candidate_indices])])
        selected.append({
            "case_type": case_type,
            "test_row_index_zero_based": index,
            "case_id": f"test_row_{index + 1:06d}",
            "actual_label": int(y_true[index]),
            "predicted_label": int(predicted[index]),
            "positive_probability": float(probabilities[index]),
            "threshold": float(threshold),
        })
    return selected


def process_dataset(name: str, output_dir: Path, logger: logging.Logger) -> dict[str, Any]:
    model_dir = MODEL_RUN_DIR / name
    model_path = model_dir / "model.joblib"
    report = json_read(model_dir / "evaluation_report.json")
    training_metadata = json_read(model_dir / "training_metadata.json")
    preprocessing_metadata = json_read(PROCESSED_DIR / name / "preprocessing_metadata.json")

    target = report.get("target")
    features = report.get("features")
    threshold = float(report.get("threshold", report.get("test_metrics", {}).get("threshold")))
    if not target or not isinstance(features, list) or target in features:
        raise ValueError(f"{name}: saved model report has invalid target or feature metadata.")
    if target != preprocessing_metadata.get("target"):
        raise ValueError(f"{name}: target mismatch between model and preprocessing metadata.")
    if training_metadata.get("status") != "success":
        raise ValueError(f"{name}: saved training metadata does not indicate success.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model artifact missing: {model_path}")

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline) or "preprocessor" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
        raise TypeError(f"{name}: expected saved sklearn Pipeline with preprocessor and classifier steps.")
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["classifier"]
    if list(preprocessing_metadata.get("feature_columns", [])) != list(features):
        raise ValueError(f"{name}: feature order does not match saved preprocessing metadata.")

    test_path = PROCESSED_DIR / name / "test.csv"
    test_frame = pd.read_csv(test_path)
    missing = [column for column in [*features, target] if column not in test_frame.columns]
    if missing:
        raise ValueError(f"{name}: test split is missing required columns: {missing}")
    X_original = test_frame.loc[:, features]
    y_true = test_frame[target].astype(int).to_numpy()
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError(f"{name}: test target does not contain both binary labels.")

    # Transform with the saved train-fitted object only. Never call fit/fit_transform here.
    X_transformed = preprocessor.transform(X_original)
    transformed_features = [str(value) for value in preprocessor.get_feature_names_out()]
    transformed = np.asarray(X_transformed)
    if transformed.ndim != 2 or transformed.shape[1] != len(transformed_features):
        raise ValueError(f"{name}: preprocessor output and feature-name dimensions disagree.")
    groups, grouping_metadata = group_indices(features, transformed_features)

    explainer = shap.TreeExplainer(model, model_output="raw", feature_perturbation="tree_path_dependent")
    transformed_shap = raw_shap_values(explainer, transformed)
    if transformed_shap.shape[1] != transformed.shape[1]:
        raise ValueError(f"{name}: SHAP feature dimension does not match transformed model input.")
    # Sum one-hot columns back to their original categorical feature. This preserves additivity.
    grouped_shap = np.column_stack([transformed_shap[:, indices].sum(axis=1) for indices in groups])
    base_values = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
    if len(base_values) != 1:
        raise ValueError(f"{name}: expected one binary positive-class raw-output baseline; got {base_values}.")
    base_value = float(base_values[0])

    # Verify SHAP reconstructs the classifier's actual positive-class raw margin.
    margins = np.asarray(model.predict(transformed, output_margin=True), dtype=np.float64).reshape(-1)
    reconstructed = base_value + grouped_shap.sum(axis=1)
    residual = np.abs(reconstructed - margins)
    max_residual = float(residual.max(initial=0.0))
    mean_residual = float(residual.mean()) if residual.size else 0.0
    tolerance = 1e-4
    if max_residual > tolerance:
        raise AssertionError(
            f"{name}: SHAP additivity check failed: max residual {max_residual:.8g} > {tolerance}."
        )
    model_probabilities = np.asarray(model.predict_proba(transformed)[:, 1], dtype=np.float64)
    margin_probabilities = 1.0 / (1.0 + np.exp(-np.clip(margins, -700, 700)))
    probability_residual = float(np.max(np.abs(model_probabilities - margin_probabilities), initial=0.0))
    if probability_residual > 1e-6:
        raise AssertionError(f"{name}: raw margin does not map to the saved model probability output.")

    importance = pd.DataFrame({
        "feature": features,
        "mean_abs_shap_value": np.mean(np.abs(grouped_shap), axis=0),
        "output_scale": "raw margin (log-odds)",
        "grouped_transformed_columns": [", ".join(grouping_metadata[feature]) for feature in features],
    }).sort_values("mean_abs_shap_value", ascending=False, kind="stable")
    importance_path = output_dir / f"{name}_global_importance.csv"
    importance.to_csv(importance_path, index=False)

    plot_values, category_codes = plotting_feature_matrix(X_original, preprocessing_metadata, pipeline)
    plt.figure(figsize=(10, max(5, 0.55 * len(features))))
    shap.summary_plot(
        grouped_shap,
        features=plot_values,
        feature_names=features,
        max_display=len(features),
        plot_type="dot",
        color_bar_label="Feature value (categorical codes; see metadata)",
        show=False,
    )
    plt.title(f"{name.upper()} SHAP summary — positive-class output (raw margin)")
    plt.tight_layout()
    beeswarm_path = output_dir / f"{name}_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, max(5, 0.55 * len(features))))
    shap.summary_plot(
        grouped_shap,
        feature_names=features,
        max_display=len(features),
        plot_type="bar",
        show=False,
    )
    plt.title(f"{name.upper()} global mean absolute SHAP — raw margin")
    plt.tight_layout()
    bar_path = output_dir / f"{name}_shap_global_bar.png"
    plt.savefig(bar_path, dpi=180, bbox_inches="tight")
    plt.close()

    cases = choose_cases(y_true, model_probabilities, threshold)
    local_dir = output_dir / f"{name}_local_waterfalls"
    local_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        row = case["test_row_index_zero_based"]
        raw_values = X_original.iloc[row].to_numpy(dtype=object)
        explanation = shap.Explanation(
            values=grouped_shap[row],
            base_values=base_value,
            data=raw_values,
            feature_names=features,
        )
        shap.plots.waterfall(explanation, max_display=len(features), show=False)
        plt.title(
            f"{name.upper()} {case['case_id']} ({case['case_type']})\n"
            f"actual={case['actual_label']}, predicted={case['predicted_label']}, "
            f"P(positive)={case['positive_probability']:.4f}, threshold={threshold:.4f}; "
            f"f(x) is raw log-odds"
        )
        figure_path = local_dir / f"{case['case_id']}_{case['case_type']}.png"
        plt.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close()
        case["waterfall_file"] = str(figure_path.relative_to(output_dir))
        case["baseline_raw_margin"] = base_value
        case["final_raw_margin"] = float(margins[row])
        case["shap_reconstructed_raw_margin"] = float(reconstructed[row])
        case["feature_values"] = {feature: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
                                  for feature, value in zip(features, raw_values)}

    metadata_out = {
        "dataset": name,
        "model_artifact": str(model_path.relative_to(PROJECT_DIR)),
        "preprocessor_artifact": str((PROCESSED_DIR / name / "preprocessor.pkl").relative_to(PROJECT_DIR)),
        "evaluation_report": str((model_dir / "evaluation_report.json").relative_to(PROJECT_DIR)),
        "test_split": str(test_path.relative_to(PROJECT_DIR)),
        "target": target,
        "feature_names_original_order": features,
        "transformed_feature_names": transformed_features,
        "transformed_to_original_feature_grouping": grouping_metadata,
        "categorical_plot_value_codes": category_codes,
        "test_rows_explained": int(len(X_original)),
        "classification_threshold": threshold,
        "local_case_ids_are": "1-based row positions in the saved test.csv; no record identifier column was present in the model report",
        "local_cases": cases,
        "explainer": "shap.TreeExplainer",
        "shap_version": shap.__version__,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "output_scale": "raw margin / log-odds for the positive class; SHAP values are additive log-odds contributions, not percentages, risks, causes, or clinical effects",
        "expected_value_baseline_raw_margin": base_value,
        "additivity_check": {
            "passed": max_residual <= tolerance,
            "tolerance": tolerance,
            "max_absolute_residual": max_residual,
            "mean_absolute_residual": mean_residual,
            "compared_against": "saved XGBoost classifier predict(output_margin=True)",
        },
        "probability_mapping_check": {
            "passed": probability_residual <= 1e-6,
            "max_absolute_probability_residual": probability_residual,
            "note": "sigmoid(raw margin) compared with saved classifier predict_proba positive-class probability",
        },
        "global_importance_csv": importance_path.name,
        "beeswarm_plot": beeswarm_path.name,
        "bar_plot": bar_path.name,
        "limitations": [
            "SHAP feature contributions describe this fitted model's output, not causal effects or clinical risk factors.",
            "The binary classifier raw output is log-odds; individual SHAP values are not percentage contributions.",
            "EHR categorical one-hot feature contributions are summed to their original feature; the beeswarm colors encode categories as integer codes documented above.",
            "Local case identifiers are row positions because the saved processed test splits do not provide a stable patient ID.",
        ],
    }
    (output_dir / f"{name}_explanation_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")

    logger.info("\n=== %s ===", name.upper())
    logger.info("Model: %s", model_path)
    logger.info("Test cases explained: %d", len(X_original))
    logger.info("Threshold: %.6f", threshold)
    logger.info("SHAP additivity: PASS (max abs residual %.3g; tolerance %.1e)", max_residual, tolerance)
    logger.info("Probability mapping: PASS (max abs residual %.3g)", probability_residual)
    logger.info("Top 10 features by mean absolute SHAP (raw margin):")
    for rank, row in enumerate(importance.head(10).itertuples(index=False), start=1):
        logger.info("  %2d. %-28s %.8f", rank, row.feature, row.mean_abs_shap_value)
    logger.info("Selected local cases:")
    for case in cases:
        logger.info(
            "  %s: %s actual=%d predicted=%d P(positive)=%.4f",
            case["case_id"], case["case_type"], case["actual_label"],
            case["predicted_label"], case["positive_probability"],
        )
    logger.info("Artifacts: %s", output_dir)
    return metadata_out


def main() -> int:
    run_dir = unique_run_dir(OUTPUT_ROOT)
    log_path = run_dir / "execution.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logger = logging.getLogger("shap_xai")
    logger.info("SHAP explanation run: %s", run_dir)
    logger.info("Existing model run: %s", MODEL_RUN_DIR)
    logger.info("SHAP %s | XGBoost %s | Python %s", shap.__version__, xgboost.__version__, platform.python_version())

    run_status: dict[str, Any] = {
        "run_directory": str(run_dir),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {},
    }
    status = 0
    for name in DATASET_NAMES:
        dataset_out = run_dir / name
        dataset_out.mkdir(parents=True, exist_ok=False)
        try:
            run_status["models"][name] = {"status": "success", **process_dataset(name, dataset_out, logger)}
        except Exception as exc:
            logger.exception("XAI failed for %s: %s", name, exc)
            (dataset_out / "error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            run_status["models"][name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            status = 1
    run_status["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_status["overall_status"] = "success" if status == 0 else "partial_failure"
    (run_dir / "run_metadata.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")
    logger.info("\nOverall status: %s", run_status["overall_status"])
    logger.info("XAI output directory: %s", run_dir)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
