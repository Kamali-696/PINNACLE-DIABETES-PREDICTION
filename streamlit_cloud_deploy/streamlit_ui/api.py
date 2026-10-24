"""Small client for the unified prediction API contract."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ApiError(RuntimeError):
    """Safe, user-displayable API error."""


class PredictionApi:
    def __init__(self, base_url: str | None = None, timeout: float = 45.0):
        configured_url = base_url or os.getenv("PINNACLE_API_BASE_URL")
        if not configured_url:
            # Streamlit Community Cloud stores deployment configuration in st.secrets;
            # local runs continue to use the localhost default when no secret is set.
            try:
                import streamlit as st

                configured_url = st.secrets.get("PINNACLE_API_BASE_URL")
            except Exception:
                configured_url = None
        self.base_url = (configured_url or "http://127.0.0.1:8000/api").rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Accept": "application/json", **({"Content-Type": "application/json"} if data else {})},
            method="GET" if data is None else "POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Do not forward backend internals or model/source identifiers to the UI.
            if exc.code == 422:
                raise ApiError("Some values were rejected. Check the form ranges and required fields.") from exc
            raise ApiError("The prediction service could not complete the request. Please try again later.") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise ApiError("The prediction service is unavailable. Check that the backend is running.") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError("The prediction service returned an unreadable response.") from exc

    def health(self) -> bool:
        try:
            return self._request("/health").get("status") == "ok"
        except ApiError:
            return False

    def predict(self, features: dict) -> dict:
        result = self._request("/predict", features)
        if set(result) != {"predicted_class", "positive_class_probability"}:
            raise ApiError("The prediction service returned an unexpected response.")
        return result

    def explain(self, features: dict) -> dict:
        return self._request("/explain", features)
