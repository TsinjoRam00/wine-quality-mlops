from __future__ import annotations

import os

import requests
import streamlit as st

from services.api_client import get_health


st.set_page_config(
    page_title="Monitoring",
    page_icon="📊",
    layout="wide",
)


with st.sidebar:
    st.title("🍷 Navigation")

    st.page_link("Home.py", label="Tableau de bord", icon="🏠")
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


st.title("📊 Monitoring MLOps")

try:
    health = get_health()

    first, second, third = st.columns(3)

    first.metric(
        "API",
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

except requests.RequestException as error:
    st.error(f"Impossible de joindre l’API : {error}")


st.divider()
st.subheader("Outils de supervision")

grafana_url = os.getenv(
    "GRAFANA_PUBLIC_URL",
    "http://localhost:3000",
)
prometheus_url = os.getenv(
    "PROMETHEUS_PUBLIC_URL",
    "http://localhost:9090",
)
mlflow_url = os.getenv(
    "MLFLOW_PUBLIC_URL",
    "http://localhost:5000",
)
jenkins_url = os.getenv(
    "JENKINS_PUBLIC_URL",
    "http://localhost:8080",
)
alertmanager_url = os.getenv(
    "ALERTMANAGER_PUBLIC_URL",
    "http://localhost:9093",
)

first, second, third = st.columns(3)

with first:
    st.link_button(
        "Ouvrir Grafana",
        grafana_url,
        use_container_width=True,
    )
    st.link_button(
        "Ouvrir Prometheus",
        prometheus_url,
        use_container_width=True,
    )

with second:
    st.link_button(
        "Ouvrir MLflow",
        mlflow_url,
        use_container_width=True,
    )
    st.link_button(
        "Ouvrir Jenkins",
        jenkins_url,
        use_container_width=True,
    )

with third:
    st.link_button(
        "Ouvrir Alertmanager",
        alertmanager_url,
        use_container_width=True,
    )
    st.link_button(
        "Documentation API",
        "http://localhost:8000/docs",
        use_container_width=True,
    )
