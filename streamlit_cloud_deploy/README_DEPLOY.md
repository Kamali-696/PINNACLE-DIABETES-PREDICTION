# PINNACLE public deployment package

This folder is a sanitized deployment bundle for Streamlit Community Cloud (UI) and Render (API). It contains the serving code, saved model/preprocessor artifacts, and JSON reports required by the existing application. It intentionally excludes all source datasets and processed train/validation/test CSVs.

## Publish the API

1. Push this folder as a GitHub repository (or its contents at the repository root).
2. In Render, create a Blueprint from that repository and select `render.yaml`.
3. Wait for the `pinnacle-diabetes-api` health check to pass. Its public API base URL is the service URL plus `/api`.

The free Render service may sleep when idle, so the first request after inactivity can be slow. Render may also require a larger instance if its available memory is insufficient for the model and explanation dependencies.

## Publish the UI

1. In Streamlit Community Cloud, deploy `streamlit_app.py` from the same repository.
2. Set the Streamlit app secret `PINNACLE_API_BASE_URL` to `https://<your-render-service>.onrender.com/api`.
3. Redeploy the app and check that the sidebar reports the prediction service is connected.

The application continues to use one unified prediction endpoint. The API currently routes predictions to the primary saved model internally; this package does not change model routing, model artifacts, or preprocessing.

## Local package smoke commands

From this directory, install the root `requirements.txt` for the UI and `backend/requirements.txt` for the API in separate Python 3.14.3 environments. Start the API from `backend` with `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`, then start the UI from this directory with `python -m streamlit run streamlit_app.py`.

No public deployment has been created from this package yet.
