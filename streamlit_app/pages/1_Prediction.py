from __future__ import annotations

import requests
import streamlit as st

from services.api_client import predict_wine


st.set_page_config(
    page_title="Prédiction",
    page_icon="🔮",
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


st.title("🔮 Nouvelle prédiction")
st.caption(
    "Renseignez les caractéristiques physico-chimiques du vin."
)


with st.form("prediction_form"):
    left, middle, right = st.columns(3)

    with left:
        fixed_acidity = st.number_input(
            "Fixed acidity",
            min_value=0.0,
            value=7.4,
            step=0.1,
        )
        volatile_acidity = st.number_input(
            "Volatile acidity",
            min_value=0.0,
            value=0.70,
            step=0.01,
        )
        citric_acid = st.number_input(
            "Citric acid",
            min_value=0.0,
            value=0.00,
            step=0.01,
        )
        residual_sugar = st.number_input(
            "Residual sugar",
            min_value=0.0,
            value=1.9,
            step=0.1,
        )

    with middle:
        chlorides = st.number_input(
            "Chlorides",
            min_value=0.0,
            value=0.076,
            format="%.3f",
        )
        free_sulfur_dioxide = st.number_input(
            "Free sulfur dioxide",
            min_value=0.0,
            value=11.0,
            step=1.0,
        )
        total_sulfur_dioxide = st.number_input(
            "Total sulfur dioxide",
            min_value=0.0,
            value=34.0,
            step=1.0,
        )
        density = st.number_input(
            "Density",
            min_value=0.0,
            value=0.9978,
            format="%.4f",
        )

    with right:
        ph = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=3.51,
            step=0.01,
        )
        sulphates = st.number_input(
            "Sulphates",
            min_value=0.0,
            value=0.56,
            step=0.01,
        )
        alcohol = st.number_input(
            "Alcohol",
            min_value=0.0,
            value=9.4,
            step=0.1,
        )

    submitted = st.form_submit_button(
        "Lancer la prédiction",
        use_container_width=True,
    )


if submitted:
    payload = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "ph": ph,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }

    try:
        result = predict_wine(payload)

        st.session_state["last_prediction"] = result
        st.session_state["last_payload"] = payload

        predicted_label = result.get(
            "predicted_label",
            "inconnue",
        )

        st.success(
            f"Qualité prédite : {predicted_label}"
        )

        probabilities = result.get("probabilities", {})

        first, second, third = st.columns(3)

        first.metric(
            "Faible",
            f"{probabilities.get('low', 0) * 100:.2f} %",
        )
        second.metric(
            "Moyenne",
            f"{probabilities.get('medium', 0) * 100:.2f} %",
        )
        third.metric(
            "Élevée",
            f"{probabilities.get('high', 0) * 100:.2f} %",
        )

        prediction_id = result.get("prediction_id")

        if prediction_id:
            st.info(f"Prediction ID : {prediction_id}")
        else:
            st.warning(
                "La prédiction a réussi, mais elle n’a pas "
                "été enregistrée dans PostgreSQL."
            )

        st.write(
            "**Modèle :**",
            result.get("model_name"),
        )
        st.write(
            "**Version :**",
            result.get("model_version"),
        )
        st.write(
            "**Alias :**",
            result.get("model_alias"),
        )

        st.page_link(
            "pages/2_Feedback.py",
            label="Envoyer un feedback",
            icon="💬",
        )

    except requests.RequestException as error:
        st.error(f"Erreur de prédiction : {error}")
