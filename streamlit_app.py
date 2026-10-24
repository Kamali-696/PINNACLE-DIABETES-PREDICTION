"""PINNACLE Streamlit application entry point."""

import streamlit as st

from streamlit_ui.api import PredictionApi
from streamlit_ui.pages import (
    render_dashboard,
    render_explainability,
    render_federated_experiment,
    render_prediction,
)

st.set_page_config(page_title="PINNACLE Diabetes Detection", page_icon="🩺", layout="wide")
st.markdown(
    """<style>
    .block-container{max-width:1120px;padding-top:2rem;padding-bottom:3rem}
    [data-testid="stMetric"]{background:#f5f8fb;border:1px solid #e4ebf2;padding:1rem;border-radius:12px}
    [data-testid="stMetric"] *{color:#203247!important}
    </style>""",
    unsafe_allow_html=True,
)

api = PredictionApi()
with st.sidebar:
    st.markdown("## PINNACLE")
    st.caption("Diabetes Detection Research")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Diabetes Prediction", "Model Explainability", "Federated Learning Experiment"],
        label_visibility="collapsed",
    )
    st.divider()
    if api.health():
        st.success("Prediction service connected")
    else:
        st.warning("Prediction service unavailable")
    st.caption("Research prototype · Not a medical device")

if page == "Dashboard":
    render_dashboard()
elif page == "Diabetes Prediction":
    render_prediction(api)
elif page == "Model Explainability":
    render_explainability(api)
elif page == "Federated Learning Experiment":
    render_federated_experiment()
