from __future__ import annotations

import requests
import streamlit as st

from services.api_client import send_feedback


st.set_page_config(
    page_title="Feedback",
    page_icon="💬",
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


st.title("💬 Feedback")
st.caption(
    "Indiquez la qualité réelle du vin afin "
    "d’évaluer les performances du modèle."
)


last_prediction = st.session_state.get(
    "last_prediction",
    {},
)

saved_prediction_id = last_prediction.get(
    "prediction_id",
    "",
)

prediction_id = st.text_input(
    "Prediction ID",
    value=saved_prediction_id or "",
    placeholder="UUID de la prédiction",
)

if last_prediction:
    st.write(
        "**Classe prédite :**",
        last_prediction.get("predicted_label"),
    )
    st.write(
        "**Version du modèle :**",
        last_prediction.get("model_version"),
    )


labels = {
    "Faible": 0,
    "Moyenne": 1,
    "Élevée": 2,
}

actual_label = st.selectbox(
    "Qualité réelle",
    options=list(labels.keys()),
)

comment = st.text_area(
    "Commentaire facultatif",
    max_chars=500,
)

if st.button(
    "Envoyer le feedback",
    type="primary",
    use_container_width=True,
):
    if not prediction_id:
        st.warning(
            "Effectuez d’abord une prédiction "
            "ou saisissez un Prediction ID."
        )
    else:
        try:
            response = send_feedback(
                prediction_id=prediction_id,
                actual_class=labels[actual_label],
                comment=comment,
            )

            st.success("Feedback enregistré avec succès.")
            st.json(response)

        except requests.RequestException as error:
            st.error(f"Erreur d’envoi : {error}")
