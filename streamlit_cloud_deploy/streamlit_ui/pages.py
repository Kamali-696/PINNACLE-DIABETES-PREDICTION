"""User-facing Streamlit pages and shared display helpers."""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from .api import ApiError, PredictionApi

PROJECT_DIR = Path(__file__).resolve().parents[1]
PRIMARY_EVALUATION = PROJECT_DIR / "local_model_results" / "run_20261024T003644Z" / "ehr" / "evaluation_report.json"
FEDERATED_DIR = PROJECT_DIR / "federated_results" / "run_20261024T012640Z"

FEATURE_LABELS = {
    "gender": "Gender",
    "age": "Age",
    "hypertension": "High blood pressure history",
    "heart_disease": "Heart disease history",
    "smoking_history": "Smoking history",
    "bmi": "BMI",
    "HbA1c_level": "HbA1c level",
    "blood_glucose_level": "Blood glucose level",
}


def _load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path.name)
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    return value


@st.cache_data(show_spinner=False)
def _primary_metrics() -> dict:
    report = _load_json(PRIMARY_EVALUATION)
    metrics = report.get("test_metrics", {})
    required = ("accuracy", "precision", "recall", "f1", "roc_auc")
    if any(not isinstance(metrics.get(key), (int, float)) for key in required):
        raise ValueError("Saved evaluation metrics are incomplete")
    return {key: float(metrics[key]) for key in required}


def _card_metric(label: str, value: float) -> None:
    st.metric(label, f"{value:.1%}")


def render_dashboard() -> None:
    st.title("Dashboard")
    st.write("A research project exploring diabetes prediction, model explanations, and a simulated collaborative-learning experiment.")

    st.subheader("Saved evaluation")
    try:
        metrics = _primary_metrics()
        columns = st.columns(5)
        for column, (key, value) in zip(columns, metrics.items()):
            with column:
                _card_metric({"roc_auc": "ROC-AUC", "f1": "F1 score"}.get(key, key.title()), value)
        st.caption("Values come from the saved held-out evaluation report. They describe a research model and do not establish clinical performance.")
    except (OSError, ValueError, json.JSONDecodeError):
        st.warning("Saved evaluation metrics are currently unavailable.")

    st.subheader("Explore the application")
    columns = st.columns(3)
    for column, title, text in zip(
        columns,
        ("Diabetes Prediction", "Model Explainability", "Federated Learning Experiment"),
        (
            "Submit health-related values for one model-generated estimate.",
            "Review contributions for a prediction made in this session.",
            "See results from a separate two-node research simulation.",
        ),
    ):
        with column:
            st.markdown(f"#### {title}")
            st.write(text)
    st.info("For educational and research use only. Outputs are not diagnoses or calibrated clinical risk estimates.")


def _validated_payload(values: dict) -> tuple[dict | None, list[str]]:
    errors = []
    required = (
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
    )
    if not values.get("gender"):
        errors.append("Select a gender value.")
    if not values.get("smoking_history"):
        errors.append("Select a smoking history value.")
    for key in required:
        if values.get(key) is None:
            errors.append(f"Enter {FEATURE_LABELS[key].lower()}.")
    ranges = {
        "age": (0, 120),
        "bmi": (10, 100),
        "HbA1c_level": (3, 20),
        "blood_glucose_level": (20, 1000),
    }
    for key, (low, high) in ranges.items():
        value = values.get(key)
        if value is not None and not low <= value <= high:
            errors.append(f"{FEATURE_LABELS[key]} must be between {low} and {high}.")
    if errors:
        return None, errors
    return {
        "gender": values["gender"],
        "age": float(values["age"]),
        "hypertension": int(values["hypertension"]),
        "heart_disease": int(values["heart_disease"]),
        "smoking_history": values["smoking_history"],
        "bmi": float(values["bmi"]),
        "HbA1c_level": float(values["HbA1c_level"]),
        "blood_glucose_level": float(values["blood_glucose_level"]),
    }, []


def render_prediction(api: PredictionApi) -> None:
    st.title("Diabetes Prediction")
    st.write("Enter the requested health-related values to receive one model-generated estimate.")

    with st.form("diabetes_prediction_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=None, placeholder="Choose an option")
            age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=None, step=1.0)
            hypertension = st.radio("High blood pressure history", ["No", "Yes"], horizontal=True, index=None)
            heart_disease = st.radio("Heart disease history", ["No", "Yes"], horizontal=True, index=None)
            smoking_history = st.selectbox(
                "Smoking history",
                ["never", "former", "current", "not current", "ever", "No Info"],
                index=None,
                placeholder="Choose an option",
                format_func=lambda value: "No information" if value == "No Info" else value.title(),
            )
        with col_b:
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=100.0, value=None, step=0.1, format="%.1f")
            hba1c = st.number_input("HbA1c level (%)", min_value=3.0, max_value=20.0, value=None, step=0.1, format="%.1f")
            glucose = st.number_input("Blood glucose level (mg/dL)", min_value=20.0, max_value=1000.0, value=None, step=1.0, format="%.1f")
        submitted = st.form_submit_button("Get prediction", type="primary", use_container_width=True)

    if submitted:
        form_values = {
            "gender": gender,
            "age": age,
            "hypertension": None if hypertension is None else int(hypertension == "Yes"),
            "heart_disease": None if heart_disease is None else int(heart_disease == "Yes"),
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose,
        }
        payload, errors = _validated_payload(form_values)
        if errors:
            for message in errors:
                st.error(message)
        else:
            try:
                with st.spinner("Generating prediction…"):
                    result = api.predict(payload)
                result["feature_values"] = payload
                st.session_state["latest_prediction"] = result
                st.session_state.pop("latest_explanation", None)
                st.success("Prediction completed.")
            except ApiError as exc:
                st.error(str(exc))

    result = st.session_state.get("latest_prediction")
    if result:
        st.divider()
        positive = int(result["predicted_class"]) == 1
        st.subheader("Prediction result")
        st.markdown(f"### Predicted outcome: **{'Positive' if positive else 'Negative'}**")
        probability = float(result["positive_class_probability"])
        st.metric("Model-generated probability score", f"{probability:.4f}")
        st.caption(f"Equivalent to {probability:.1%}. This is not a confirmed diagnosis or calibrated clinical risk.")
        st.info("Educational and research use only. A qualified health professional should interpret results in context.")
        st.caption("You can view feature contributions for this prediction on the Model Explainability page.")


def _explanation_chart(contributions: list[dict]) -> None:
    values = [float(item["shap_value"]) for item in contributions]
    max_abs = max((abs(value) for value in values), default=0.0) or 1.0
    rows = []
    for item, value in zip(contributions, values):
        feature = FEATURE_LABELS.get(item.get("feature"), item.get("feature", "Feature"))
        safe_feature = html.escape(str(feature))
        color = "#138a72" if value >= 0 else "#5b75d1"
        width = max(2.0, min(100.0, abs(value) / max_abs * 100.0))
        direction = "toward positive" if value > 0 else "toward negative" if value < 0 else "no contribution"
        rows.append(
            f'<div class="contribution-row"><div class="contribution-head"><span>{safe_feature}</span>'
            f'<span>{value:+.4f} · {direction}</span></div>'
            f'<div class="contribution-track"><div style="width:{width:.1f}%;background:{color};"></div></div></div>'
        )
    st.markdown("""<style>
    .contribution-row{margin:0.9rem 0 1.1rem}.contribution-head{display:flex;justify-content:space-between;gap:1rem;
    font-size:.92rem;color:#233447}.contribution-track{height:10px;background:#e9eef4;border-radius:8px;margin-top:.38rem;overflow:hidden}
    .contribution-track div{height:100%;border-radius:8px}
    </style>""" + "".join(rows), unsafe_allow_html=True)


def render_explainability(api: PredictionApi) -> None:
    st.title("Model Explainability")
    st.write("Feature contributions show how the model output moved for the prediction made most recently in this session.")
    prediction = st.session_state.get("latest_prediction")
    if not prediction:
        st.info("Make a prediction first. Its submitted values will be used for the explanation.")
        return
    if st.button("Generate explanation", type="primary"):
        try:
            with st.spinner("Calculating feature contributions…"):
                explanation = api.explain(prediction["feature_values"])
            same_result = (
                explanation.get("predicted_class") == prediction.get("predicted_class")
                and abs(float(explanation.get("positive_class_probability", -1)) - float(prediction["positive_class_probability"])) < 1e-6
            )
            if not same_result:
                raise ApiError("The explanation did not match the current prediction. Please submit the form again.")
            st.session_state["latest_explanation"] = explanation
        except (ApiError, TypeError, ValueError) as exc:
            st.error(str(exc))

    explanation = st.session_state.get("latest_explanation")
    if explanation:
        st.caption(f"Current prediction: {'Positive' if explanation['predicted_class'] else 'Negative'} · {float(explanation['positive_class_probability']):.1%} model-generated probability")
        contributions = explanation.get("top_contributions", [])
        if not contributions:
            st.info("No feature contributions were returned.")
            return
        _explanation_chart(contributions)
        st.info("Positive values move the model output toward a positive prediction; negative values move it toward a negative prediction. SHAP contributions describe model behavior, not causal effects or medical risk factors.")
        st.caption("Contribution values are in raw model-output (log-odds) units, not percentages.")


def _percent(value: float) -> str:
    return f"{float(value):.1%}"


def render_federated_experiment() -> None:
    st.title("Federated Learning Experiment")
    st.write("This page summarizes a saved two-node simulation. It is a separate research experiment and does not power the prediction form.")
    try:
        report = _load_json(FEDERATED_DIR / "evaluation_report.json")
        config = _load_json(FEDERATED_DIR / "configuration.json")
        mapping = _load_json(FEDERATED_DIR / "shared_feature_mapping.json")
        history = _load_json(FEDERATED_DIR / "round_history.json")
        node_keys = list(config.get("nodes", []))
        if (
            not isinstance(mapping, dict)
            or mapping.get("shared_feature_order") != config.get("shared_feature_order")
            or len(node_keys) != 2
            or not isinstance(history, list)
            or len(history) != int(config.get("communication_rounds", -1))
        ):
            raise ValueError("Saved experiment structure is incomplete")
        shared = config.get("shared_feature_order", [])
        st.subheader("Experiment setup")
        st.write(f"**Shared feature space:** {', '.join('Age (years)' if 'age' in item else 'BMI (kg/m²)' if 'bmi' in item else str(item) for item in shared)}")
        st.write(f"**Communication rounds:** {len(history)}")
        st.write(f"**Local training passes per round:** {int(config['local_epochs_per_round'])}")
        st.caption("The run was simulated locally using parameter averaging. The participating nodes are shown with neutral labels.")

        for index, internal_node in enumerate(node_keys):
            label = f"Node {'A' if index == 0 else 'B'}"
            node_report = report[internal_node]
            local = node_report["local_logistic_regression"]["test_metrics"]
            shared_result = node_report["federated_global_logistic_regression"]["test_metrics"]
            st.subheader(label)
            st.caption(f"Held-out evaluation rows: {int(node_report['test_sample_count']):,}")
            table = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 score", "ROC-AUC"],
                "Local result": [_percent(local[key]) for key in ("accuracy", "precision", "recall", "f1", "roc_auc")],
                "Global result": [_percent(shared_result[key]) for key in ("accuracy", "precision", "recall", "f1", "roc_auc")],
            }
            st.dataframe(table, hide_index=True, width="stretch")
        st.subheader("Experiment limitations")
        st.write("This is a simulation, not a deployed multi-institution system. It does not use secure aggregation, encryption, or differential privacy. The shared feature space contains only age and BMI; the results are research observations and are not clinically validated.")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        st.warning("Saved federated experiment results are incomplete or unavailable.")


