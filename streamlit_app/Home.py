from __future__ import annotations

import requests
import streamlit as st

from services.api_client import get_health


st.set_page_config(
    page_title="Wine Quality MLOps",
    page_icon="🍷",
    layout="wide",
)


with st.sidebar:
    st.title("🍷 Navigation")

    st.page_link(
        "Home.py",
        label="Tableau de bord",
        icon="🏠",
    )
    st.page_link(
        "pages/1_Prediction.py",
        label="Prédiction",
        icon="🔮",
    )
    st.page_link(
        "pages/2_Feedback.py",
        label="Feedback",
        icon="💬",
    )
    st.page_link(
        "pages/3_Monitoring.py",
        label="Monitoring",
        icon="📊",
    )


st.title("🍷 Wine Quality MLOps")
st.caption(
    "Plateforme de prédiction et de supervision "
    "de la qualité du vin."
)

st.header("État du système")

try:
    health = get_health()

    first, second, third = st.columns(3)

    first.metric(
        "API FastAPI",
        health.get("status", "inconnu"),
    )
    second.metric(
        "MLflow",
        health.get("mlflow_status", "inconnu"),
    )
    third.metric(
        "Modèle",
        health.get("model_status", "inconnu"),
    )

    st.success("Les services principaux sont disponibles.")

except requests.RequestException as error:
    st.error(f"API indisponible : {error}")


st.divider()
st.header("Accès rapide")

first, second, third = st.columns(3)

with first:
    st.page_link(
        "pages/1_Prediction.py",
        label="Lancer une prédiction",
        icon="🔮",
        use_container_width=True,
    )

with second:
    st.page_link(
        "pages/2_Feedback.py",
        label="Envoyer un feedback",
        icon="💬",
        use_container_width=True,
    )

with third:
    st.page_link(
        "pages/3_Monitoring.py",
        label="Voir le monitoring",
        icon="📊",
        use_container_width=True,
    )
