# PINNACLE Diabetes Detection

The active user interface is the Streamlit application. The React files under `frontend/` are retained as source history and are not used in the active workflow.

> **Research software only.** Predictions are educational/research outputs, not diagnoses or calibrated clinical risk estimates. The project has not been clinically validated.

## Run the application on Windows CMD

Follow [the Streamlit setup guide](STREAMLIT_README.md). In short, install the frontend dependencies, start the existing FastAPI service, then run:

```cmd
cd /d D:\PINNACLE_DIABETES_PREDICTION
.streamlit-venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The backend defaults to `http://127.0.0.1:8000`. Set `PINNACLE_API_BASE_URL` to override its API base URL.

## Project artifacts

- The prediction API uses the saved primary pipeline and train-fitted preprocessing artifacts. It does not train or refit when serving requests.
- Saved evaluation reports are in `local_model_results/`.
- The separate simulated federated-learning results are in `federated_results/`. They are not used by the prediction form.
- Backend routes, schemas, and example payloads are documented in [`backend/API.md`](backend/API.md).

The Streamlit prediction and explanation pages use the unified `POST /api/predict` and `POST /api/explain` routes. These are fixed to the existing primary form schema; the UI does not infer a model from health values or ask the user to select an internal source.

## Backend checks

Run the backend tests from Windows CMD:

```cmd
cd /d D:\PINNACLE_DIABETES_PREDICTION\backend
python -m unittest discover -s tests -v
```
