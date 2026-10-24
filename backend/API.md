# Pinnacle local-model API

The API loads the independently trained EHR and Pima XGBoost models from
`local_model_results/run_20261024T003644Z/`. It loads the separately saved
train-fitted preprocessors from `processed/ehr/` and `processed/pima/`, then verifies
their transformed columns match the preprocessing embedded in each model pipeline.
The API fails startup if an artifact is missing or the model's XGBoost/scikit-learn
runtime versions do not match its saved training metadata. It never trains or refits.

## Endpoints

- `POST /api/predict` — unified public prediction using the fixed primary form schema; returns only predicted class and positive-class probability.
- `POST /api/explain` — SHAP contributions for the same primary pipeline and input schema.

The unified public routes use the EHR-shaped schema because it is the schema used by the existing primary prediction form and matching saved pipeline. This is a fixed internal choice, not a guess based on submitted health values. The unified response models omit internal source and model identifiers. The Streamlit UI sends raw values; the saved backend preprocessor and threshold remain authoritative.

- `GET /api/health` — reports availability of both required models.
- `POST /api/predict/ehr` — EHR probability and predicted class using the saved threshold.
- `POST /api/predict/pima` — Pima probability and predicted class using the saved threshold.
- `POST /api/explain/ehr` — top single-record EHR SHAP contributions.
- `POST /api/explain/pima` — top single-record Pima SHAP contributions.

For EHR, `smoking_history: "No Info"` is treated as missing and passed through the saved
training-fitted imputer. For Pima, zeros in `Glucose`, `BloodPressure`, `SkinThickness`,
`Insulin`, and `BMI` are treated as missing, matching the preprocessing script; zero
`Pregnancies` remains valid. These sentinels are handled before the saved imputer runs.

The explanation endpoint reports raw-margin/log-odds contributions. SHAP describes the
fitted model's output; its values are not percentages, causal effects, or clinical risk
estimates.

## Example request bodies

`POST /api/predict/ehr` (the explanation endpoint accepts the same body):

```json
{
  "gender": "Female",
  "age": 45,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "never",
  "bmi": 27.5,
  "HbA1c_level": 5.8,
  "blood_glucose_level": 130
}
```

`POST /api/predict/pima` (the explanation endpoint accepts the same body):

```json
{
  "Pregnancies": 1,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 25,
  "Insulin": 90,
  "BMI": 28.0,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 30
}
```

Unknown fields, missing required fields, invalid categories, and out-of-range values receive
HTTP 422. Unsupported dataset paths receive HTTP 404. Internal prediction failures return a
generic HTTP 500 message; stack traces and artifact paths are not included in API responses.

## CMD commands

Run from the project directory:

```cmd
cd /d D:\PINNACLE_DIABETES_PREDICTION
python -m venv backend\.venv
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
```

Start the backend:

```cmd
cd /d D:\PINNACLE_DIABETES_PREDICTION\backend
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```

Run backend tests (fabricated request payloads only):

```cmd
cd /d D:\PINNACLE_DIABETES_PREDICTION\backend
.venv\Scripts\python -m unittest discover -s tests -v
```
