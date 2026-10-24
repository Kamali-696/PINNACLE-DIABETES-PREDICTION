"""API and saved-artifact checks using fabricated request payloads only."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.model_service import ModelService


EHR_INPUT = {
    "gender": "Female",
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 27.5,
    "HbA1c_level": 5.8,
    "blood_glucose_level": 130,
}

PIMA_INPUT = {
    "Pregnancies": 1,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 90,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30,
}


class ModelApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        cls.service = app.state.model_service

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_health_reports_both_models_available(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["models"]["ehr"]["available"])
        self.assertTrue(body["models"]["pima"]["available"])

    def test_model_artifact_loading_and_preprocessor_match(self):
        self.assertTrue(self.service.loaded)
        self.assertEqual(set(self.service.models), {"ehr", "pima"})
        service = ModelService(settings.model_run_dir, settings.processed_dir)
        service.load()
        self.assertTrue(service.loaded)
        self.assertEqual(service.models["ehr"].target, "diabetes")
        self.assertEqual(service.models["pima"].target, "Outcome")

    def test_missing_model_artifact_fails_clearly(self):
        service = ModelService(Path("missing-model-run"), Path("missing-processed"))
        with self.assertRaisesRegex(RuntimeError, "required EHR model bundle"):
            service.load()

    def test_ehr_prediction_and_threshold_application(self):
        response = self.client.post("/api/predict/ehr", json=EHR_INPUT)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["dataset"], "ehr")
        self.assertEqual(result["target"], "diabetes")
        self.assertGreaterEqual(result["positive_class_probability"], 0.0)
        self.assertLessEqual(result["positive_class_probability"], 1.0)
        self.assertEqual(result["predicted_class"], int(result["positive_class_probability"] >= result["threshold"]))

    def test_unified_prediction_uses_primary_pipeline_without_internal_metadata(self):
        response = self.client.post("/api/predict", json=EHR_INPUT)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(set(result), {"predicted_class", "positive_class_probability"})
        bundle = self.service.models["ehr"]
        frame = pd.DataFrame([EHR_INPUT], columns=bundle.features)
        expected = float(bundle.pipeline.predict_proba(frame)[0, 1])
        self.assertAlmostEqual(result["positive_class_probability"], expected, places=6)
        self.assertEqual(result["predicted_class"], int(expected >= bundle.threshold))

    def test_unified_explanation_matches_primary_prediction(self):
        prediction = self.client.post("/api/predict", json=EHR_INPUT)
        explanation = self.client.post("/api/explain", json=EHR_INPUT)
        self.assertEqual(prediction.status_code, 200, prediction.text)
        self.assertEqual(explanation.status_code, 200, explanation.text)
        prediction_body = prediction.json()
        explanation_body = explanation.json()
        self.assertEqual(explanation_body["predicted_class"], prediction_body["predicted_class"])
        self.assertAlmostEqual(
            explanation_body["positive_class_probability"],
            prediction_body["positive_class_probability"],
            places=7,
        )
        self.assertNotIn("dataset", explanation_body)
        self.assertNotIn("model_name", explanation_body)

    def test_pima_prediction_and_threshold_application(self):
        response = self.client.post("/api/predict/pima", json=PIMA_INPUT)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["dataset"], "pima")
        self.assertEqual(result["target"], "Outcome")
        self.assertEqual(result["predicted_class"], int(result["positive_class_probability"] >= result["threshold"]))

    def test_predictions_match_saved_pipeline_directly(self):
        for dataset, payload in (("ehr", EHR_INPUT), ("pima", PIMA_INPUT)):
            with self.subTest(dataset=dataset):
                bundle = self.service.models[dataset]
                frame = pd.DataFrame([payload], columns=bundle.features)
                expected = float(bundle.pipeline.predict_proba(frame)[0, 1])
                actual = self.client.post(f"/api/predict/{dataset}", json=payload).json()
                self.assertAlmostEqual(actual["positive_class_probability"], expected, places=6)
                self.assertEqual(actual["threshold"], bundle.threshold)

    def test_pima_zero_measurement_sentinels_use_saved_imputer(self):
        payload = {**PIMA_INPUT, "Glucose": 0, "BloodPressure": 0, "SkinThickness": 0, "Insulin": 0, "BMI": 0}
        result = self.client.post("/api/predict/pima", json=payload)
        self.assertEqual(result.status_code, 200, result.text)
        bundle = self.service.models["pima"]
        frame = pd.DataFrame([payload], columns=bundle.features)
        for column in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"):
            frame.loc[0, column] = np.nan
        expected = float(bundle.classifier.predict_proba(bundle.preprocessor.transform(frame))[0, 1])
        self.assertAlmostEqual(result.json()["positive_class_probability"], expected, places=6)

    def test_ehr_no_info_category_uses_saved_imputer(self):
        payload = {**EHR_INPUT, "smoking_history": "No Info"}
        response = self.client.post("/api/predict/ehr", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        bundle = self.service.models["ehr"]
        frame = pd.DataFrame([payload], columns=bundle.features)
        frame["smoking_history"] = np.nan
        expected = float(bundle.classifier.predict_proba(bundle.preprocessor.transform(frame))[0, 1])
        self.assertAlmostEqual(response.json()["positive_class_probability"], expected, places=6)

    def test_missing_invalid_and_extra_inputs_are_rejected(self):
        self.assertEqual(self.client.post("/api/predict/ehr", json={k: v for k, v in EHR_INPUT.items() if k != "bmi"}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict/pima", json={**PIMA_INPUT, "Age": "old"}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict/pima", json={**PIMA_INPUT, "Glucose": 10}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict/pima", json={**PIMA_INPUT, "BMI": 5}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict/ehr", json={**EHR_INPUT, "age": 140}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict/ehr", json={**EHR_INPUT, "extra": 1}).status_code, 422)
        self.assertEqual(self.client.post("/api/predict/ehr", json={**EHR_INPUT, "smoking_history": "unknown"}).status_code, 422)

    def test_unknown_dataset_is_rejected(self):
        response = self.client.post("/api/predict/not-a-dataset", json={})
        self.assertEqual(response.status_code, 404)
        self.assertNotIn("D:\\", response.text)

    def test_shap_explanations_reconstruct_raw_margin(self):
        for dataset, payload in (("ehr", EHR_INPUT), ("pima", PIMA_INPUT)):
            with self.subTest(dataset=dataset):
                response = self.client.post(f"/api/explain/{dataset}", json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["output_scale"], "raw margin (log-odds) for positive-class prediction")
                self.assertLessEqual(body["reconstruction_error"], 1e-4)
                self.assertTrue(body["top_contributions"])
                self.assertIn("not percentages", body["interpretation_note"].lower())
                self.assertIn("not causal effects", body["interpretation_note"].lower())


if __name__ == "__main__":
    unittest.main()
