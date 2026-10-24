# PINNACLE Streamlit application

The active user interface is the Streamlit application. The React source under `frontend/` is retained for reference and is not part of the run workflow.

## Start on Windows CMD

From the project directory:

```cmd
cd /d D:\PINNACLE_DIABETES_PREDICTION
uv venv .streamlit-venv --python 3.14
uv pip install --python .streamlit-venv\Scripts\python.exe -r streamlit_requirements.txt
```

In one CMD window, start the existing backend from a Python environment installed with `backend/requirements.txt` (the environment must match the saved model runtime versions):

```cmd
cd /d D:\PINNACLE_DIABETES_PREDICTION\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another CMD window, start Streamlit:

```cmd
cd /d D:\PINNACLE_DIABETES_PREDICTION
.streamlit-venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Set `PINNACLE_API_BASE_URL` if the API is hosted elsewhere; the default is `http://127.0.0.1:8000/api`.

## Unified API contract

`POST /api/predict` and `POST /api/explain` use the exact schema of the primary saved pipeline, determined by the existing React form and saved artifacts. These routes do not infer a source from health values or return internal model/source identifiers. Existing explicit internal API routes remain available to backend clients. The backend applies the saved preprocessing and threshold; Streamlit sends raw form values without additional preprocessing.

The explainability page only explains the most recent prediction in the session and checks that the returned class and probability match it. SHAP values are in raw model-output units and are not causal effects.

The dashboard and federated experiment page read saved JSON reports. The federated experiment is presented as a separate simulation and is not used by the prediction route.
