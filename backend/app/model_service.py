"""Load the saved local EHR/Pima XGBoost models once and serve predictions/explanations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from scipy import sparse
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

DATASETS = {
    "ehr": {"target": "diabetes", "model_name": "EHR local XGBoost", "case": "ehr"},
    "pima": {"target": "Outcome", "model_name": "Pima local XGBoost", "case": "pima"},
}
SYNTHETIC_STARTUP_ROWS = {
    "ehr": {
        "gender": "Female", "age": 45.0, "hypertension": 0, "heart_disease": 0,
        "smoking_history": "never", "bmi": 27.5, "HbA1c_level": 5.8,
        "blood_glucose_level": 130.0,
    },
    "pima": {
        "Pregnancies": 1, "Glucose": 120.0, "BloodPressure": 70.0,
        "SkinThickness": 25.0, "Insulin": 90.0, "BMI": 28.0,
        "DiabetesPedigreeFunction": 0.5, "Age": 30.0,
    },
}
PIMA_ZERO_SENTINELS = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


@dataclass
class LoadedModel:
    dataset: str
    target: str
    model_name: str
    features: list[str]
    threshold: float
    pipeline: Pipeline
    classifier: XGBClassifier
    preprocessor: Any
    metadata: dict[str, Any]
    report: dict[str, Any]
    category_groups: list[list[int]]
    transformed_feature_names: list[str]
    shap_explainer: shap.TreeExplainer
    shap_base_value: float


@dataclass
class ModelService:
    model_run_dir: Path
    processed_dir: Path
    models: dict[str, LoadedModel] = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return set(self.models) == set(DATASETS)

    def load(self) -> None:
        """Load and validate both local model bundles; fail startup if either is unusable."""
        self.models.clear()
        for dataset, spec in DATASETS.items():
            try:
                self.models[dataset] = self._load_dataset(dataset, spec)
                logger.info("Loaded %s local XGBoost model and preprocessing artifacts.", dataset)
            except Exception as exc:
                self.models.clear()
                logger.exception("Failed to load %s model bundle.", dataset)
                raise RuntimeError(
                    f"Could not load the required {dataset.upper()} model bundle. "
                    "Verify model, preprocessing, metadata, and runtime dependencies."
                ) from exc

    def _load_dataset(self, dataset: str, spec: dict[str, str]) -> LoadedModel:
        model_dir = self.model_run_dir / dataset
        model_path = model_dir / "model.joblib"
        training_metadata_path = model_dir / "training_metadata.json"
        evaluation_path = model_dir / "evaluation_report.json"
        preprocessing_path = self.processed_dir / dataset / "preprocessor.pkl"
        preprocessing_metadata_path = self.processed_dir / dataset / "preprocessing_metadata.json"
        for path in (model_path, training_metadata_path, evaluation_path, preprocessing_path, preprocessing_metadata_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required {dataset} artifact is missing: {path.name}")

        pipeline = joblib.load(model_path)
        separate_preprocessor = joblib.load(preprocessing_path)
        training_metadata = json.loads(training_metadata_path.read_text(encoding="utf-8"))
        report = json.loads(evaluation_path.read_text(encoding="utf-8"))
        preprocessing_metadata = json.loads(preprocessing_metadata_path.read_text(encoding="utf-8"))
        if not isinstance(pipeline, Pipeline) or set(("preprocessor", "classifier")) - set(pipeline.named_steps):
            raise TypeError("Saved model is not the expected preprocessing/classifier Pipeline.")
        classifier = pipeline.named_steps["classifier"]
        embedded_preprocessor = pipeline.named_steps["preprocessor"]
        if not isinstance(classifier, XGBClassifier):
            raise TypeError("Saved classifier is not an XGBClassifier; no fallback model is used.")
        if training_metadata.get("status") != "success":
            raise ValueError("Saved training metadata does not report a successful local-model run.")

        versions = training_metadata.get("versions", {})
        runtime_versions = {"xgboost": xgboost.__version__, "scikit_learn": sklearn.__version__}
        mismatches = {
            package: {"trained": versions.get(package), "runtime": runtime}
            for package, runtime in runtime_versions.items()
            if versions.get(package) != runtime
        }
        if mismatches:
            raise RuntimeError(f"Model serialization runtime mismatch: {mismatches}")

        target = report.get("target")
        features = report.get("features")
        threshold = report.get("threshold")
        if target != spec["target"] or target != preprocessing_metadata.get("target"):
            raise ValueError("Target column differs across model and preprocessing metadata.")
        if not isinstance(features, list) or not features or target in features:
            raise ValueError("Saved feature order is missing or includes the target.")
        if features != preprocessing_metadata.get("feature_columns"):
            raise ValueError("Model feature order does not match preprocessing feature order.")
        if not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
            raise ValueError("Saved evaluation report has an invalid classification threshold.")
        if not np.array_equal(np.asarray(classifier.classes_), np.asarray([0, 1])):
            raise ValueError("Saved XGBoost classifier does not expose binary classes [0, 1].")

        external_names = [str(value) for value in separate_preprocessor.get_feature_names_out()]
        embedded_names = [str(value) for value in embedded_preprocessor.get_feature_names_out()]
        if external_names != embedded_names:
            raise ValueError("Standalone preprocessor feature names/order differ from model pipeline.")
        if hasattr(separate_preprocessor, "feature_names_in_") and list(separate_preprocessor.feature_names_in_) != features:
            raise ValueError("Standalone preprocessor input feature order differs from model metadata.")
        if getattr(classifier, "n_features_in_", len(external_names)) != len(external_names):
            raise ValueError("Transformed preprocessing dimension differs from classifier input dimension.")

        # Compare both fitted preprocessing objects using a fixed synthetic example. This checks
        # artifact compatibility without reading or using any patient row.
        startup_frame = pd.DataFrame([SYNTHETIC_STARTUP_ROWS[dataset]], columns=features)
        separate_output = self._dense(separate_preprocessor.transform(startup_frame))
        embedded_output = self._dense(embedded_preprocessor.transform(startup_frame))
        if separate_output.shape != embedded_output.shape or not np.allclose(
            separate_output, embedded_output, rtol=0, atol=1e-12
        ):
            raise ValueError("Standalone preprocessing artifact does not match the model's embedded preprocessor.")

        category_groups = self._make_feature_groups(features, external_names)
        explainer = shap.TreeExplainer(
            classifier,
            model_output="raw",
            feature_perturbation="tree_path_dependent",
        )
        startup_shap = explainer.shap_values(separate_output, check_additivity=False)
        if isinstance(startup_shap, list):
            startup_shap = startup_shap[1] if len(startup_shap) == 2 else startup_shap[0]
        startup_shap = np.asarray(startup_shap, dtype=np.float64)
        if startup_shap.ndim == 3:
            startup_shap = startup_shap[:, :, 1] if startup_shap.shape[-1] == 2 else startup_shap[:, :, 0]
        if startup_shap.shape != separate_output.shape:
            raise ValueError("SHAP startup probe dimensions do not match transformed model inputs.")
        base_values = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
        if len(base_values) != 1:
            raise ValueError("Expected a single positive-class raw-margin SHAP baseline.")
        startup_margin = float(np.asarray(classifier.predict(separate_output, output_margin=True)).reshape(-1)[0])
        startup_grouped_sum = sum(float(startup_shap[0, indices].sum()) for indices in category_groups)
        if abs(float(base_values[0]) + startup_grouped_sum - startup_margin) > 1e-4:
            raise ValueError("SHAP startup probe failed to reconstruct the classifier's raw-margin output.")

        return LoadedModel(
            dataset=dataset,
            target=target,
            model_name=spec["model_name"],
            features=features,
            threshold=float(threshold),
            pipeline=pipeline,
            classifier=classifier,
            preprocessor=separate_preprocessor,
            metadata=preprocessing_metadata,
            report=report,
            category_groups=category_groups,
            transformed_feature_names=external_names,
            shap_explainer=explainer,
            shap_base_value=float(base_values[0]),
        )

    @staticmethod
    def _dense(matrix: Any) -> np.ndarray:
        return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)

    @staticmethod
    def _make_feature_groups(features: list[str], transformed_names: list[str]) -> list[list[int]]:
        groups = []
        assigned: list[int] = []
        for feature in features:
            indices = [
                index for index, transformed_name in enumerate(transformed_names)
                if transformed_name.split("__", 1)[-1] == feature
                or transformed_name.split("__", 1)[-1].startswith(feature + "_")
            ]
            if not indices:
                raise ValueError(f"Cannot map transformed columns to original feature {feature!r}.")
            groups.append(indices)
            assigned.extend(indices)
        if sorted(assigned) != list(range(len(transformed_names))):
            raise ValueError("Model transformed columns do not map one-to-one to declared input features.")
        return groups

    def _prepare_row(self, bundle: LoadedModel, features: dict[str, Any]) -> pd.DataFrame:
        if set(features) != set(bundle.features):
            raise ValueError("Input feature names do not exactly match the selected model schema.")
        row = dict(features)
        # These sentinel rules mirror preprocess_diabetes.py. The fitted SimpleImputer then
        # uses only statistics learned from that dataset's training split.
        if bundle.dataset == "ehr" and row.get("smoking_history") == "No Info":
            row["smoking_history"] = np.nan
        if bundle.dataset == "pima":
            for column in PIMA_ZERO_SENTINELS:
                if row.get(column) == 0:
                    row[column] = np.nan
        return pd.DataFrame([row], columns=bundle.features)

    def predict(self, dataset: str, features: dict[str, Any]) -> dict[str, Any]:
        bundle = self.models[dataset]
        frame = self._prepare_row(bundle, features)
        transformed = bundle.preprocessor.transform(frame)
        probability = float(bundle.classifier.predict_proba(transformed)[0, 1])
        return {
            "dataset": dataset,
            "target": bundle.target,
            "model_name": bundle.model_name,
            "predicted_class": int(probability >= bundle.threshold),
            "positive_class_probability": probability,
            "threshold": bundle.threshold,
        }

    def explain(self, dataset: str, features: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
        bundle = self.models[dataset]
        frame = self._prepare_row(bundle, features)
        transformed = self._dense(bundle.preprocessor.transform(frame))
        raw_values = bundle.shap_explainer.shap_values(transformed, check_additivity=False)
        if isinstance(raw_values, list):
            raw_values = raw_values[1] if len(raw_values) == 2 else raw_values[0]
        raw_values = np.asarray(raw_values, dtype=np.float64)
        if raw_values.ndim == 3:
            raw_values = raw_values[:, :, 1] if raw_values.shape[-1] == 2 else raw_values[:, :, 0]
        if raw_values.shape != transformed.shape:
            raise ValueError("SHAP output dimensions do not match transformed model features.")
        grouped = np.asarray([raw_values[0, group].sum() for group in bundle.category_groups], dtype=np.float64)
        # SHAP initializes the expected value lazily on the first shap_values call for
        # XGBoost TreeExplainer. Refresh it after that call rather than using the initial
        # placeholder value (often 0.0).
        base_values = np.asarray(bundle.shap_explainer.expected_value, dtype=np.float64).reshape(-1)
        if len(base_values) != 1:
            raise ValueError("SHAP did not provide a single positive-class baseline.")
        bundle.shap_base_value = float(base_values[0])
        margin = float(np.asarray(bundle.classifier.predict(transformed, output_margin=True)).reshape(-1)[0])
        reconstructed = bundle.shap_base_value + float(grouped.sum())
        reconstruction_error = abs(reconstructed - margin)
        if reconstruction_error > 1e-4:
            raise ValueError("SHAP contributions failed the model-output reconstruction check.")

        ranked = sorted(range(len(bundle.features)), key=lambda index: abs(grouped[index]), reverse=True)[:top_k]
        contributions = []
        for index in ranked:
            value = features[bundle.features[index]]
            contributions.append({
                "feature": bundle.features[index],
                "input_value": value,
                "shap_value": float(grouped[index]),
                "direction": "toward_positive" if grouped[index] > 0 else "toward_negative" if grouped[index] < 0 else "no_effect",
            })
        prediction = self.predict(dataset, features)
        return {
            **prediction,
            "output_scale": "raw margin (log-odds) for positive-class prediction",
            "baseline_raw_margin": bundle.shap_base_value,
            "final_raw_margin": margin,
            "reconstruction_error": reconstruction_error,
            "top_contributions": contributions,
            "interpretation_note": "SHAP values explain this model output in log-odds units. They are not percentages. They are not causal effects or clinical risk estimates.",
        }
