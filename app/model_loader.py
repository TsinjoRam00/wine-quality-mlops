# app/model_loader.py

# Importation de logging pour afficher des logs propres.
import logging

# Importation de Path pour manipuler les chemins de fichiers.
from pathlib import Path

# Importation de joblib pour charger les objets de preprocessing.
import joblib

# Importation de NumPy pour manipuler les tableaux numériques.
import numpy as np

# Importation de Pandas pour créer les DataFrames.
import pandas as pd

# Importation de MLflow.
import mlflow

# Importation du client MLflow.
from mlflow.tracking import MlflowClient

# Importation du flavor sklearn MLflow.
import mlflow.sklearn

# Importation du flavor XGBoost MLflow.
import mlflow.xgboost

# Importation de la configuration.
from app.config import CLASS_LABELS
from app.config import FEATURE_COLUMNS
from app.config import MLFLOW_MODEL_ALIAS
from app.config import MLFLOW_MODEL_NAME
from app.config import MLFLOW_REGISTRY_URI
from app.config import MLFLOW_TRACKING_URI
from app.config import PREPROCESSING_OBJECTS_PATH


# Création du logger.
logger = logging.getLogger(__name__)


# Classe responsable du chargement du modèle et du preprocessing.
class ModelLoader:
    # Initialisation du loader.
    def __init__(self):
        # Modèle natif sklearn ou xgboost.
        self.native_model = None

        # Modèle pyfunc MLflow.
        self.pyfunc_model = None

        # Objets de preprocessing.
        self.preprocessing_objects = None

        # URI du modèle chargé.
        self.model_uri = None

        # Version MLflow du modèle.
        self.model_version = None

        # Alias MLflow utilisé.
        self.model_alias = None

    # Configuration de MLflow.
    def configure_mlflow(self) -> None:
        # Définition du tracking URI.
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Définition du registry URI.
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)

    # Récupération de l'URI du meilleur modèle.
    def resolve_model_uri(self) -> str:
        # Configuration MLflow.
        self.configure_mlflow()

        # Création du client MLflow.
        client = MlflowClient()

        # Tentative de chargement via alias champion.
        try:
            # Récupération de la version correspondant à l'alias.
            model_version = client.get_model_version_by_alias(
                name=MLFLOW_MODEL_NAME,
                alias=MLFLOW_MODEL_ALIAS,
            )

            # Sauvegarde de la version.
            self.model_version = str(model_version.version)

            # Sauvegarde de l'alias.
            self.model_alias = MLFLOW_MODEL_ALIAS

            # Retour de l'URI avec alias.
            return f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"

        # Si l'alias n'existe pas, on prend la dernière version.
        except Exception as error:
            # Log d'un avertissement non bloquant.
            logger.warning("Alias MLflow indisponible, fallback dernière version : %s", error)

        # Recherche de toutes les versions du modèle.
        versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")

        # Si aucune version n'existe, on bloque.
        if not versions:
            # Levée d'une erreur claire.
            raise RuntimeError(
                f"Aucune version trouvée dans MLflow pour le modèle {MLFLOW_MODEL_NAME}."
            )

        # Tri des versions par numéro décroissant.
        latest_version = sorted(
            versions,
            key=lambda item: int(item.version),
            reverse=True,
        )[0]

        # Sauvegarde de la version.
        self.model_version = str(latest_version.version)

        # Aucun alias utilisé dans ce cas.
        self.model_alias = None

        # Retour de l'URI avec version.
        return f"models:/{MLFLOW_MODEL_NAME}/{latest_version.version}"

    # Chargement des objets de preprocessing.
    def load_preprocessing_objects(self) -> None:
        # Conversion du chemin en Path.
        preprocessing_path = Path(PREPROCESSING_OBJECTS_PATH)

        # Vérification de l'existence du fichier.
        if not preprocessing_path.exists():
            # Levée d'une erreur claire.
            raise FileNotFoundError(
                f"Fichier preprocessing introuvable : {preprocessing_path}"
            )

        # Chargement du fichier joblib.
        self.preprocessing_objects = joblib.load(preprocessing_path)

    # Chargement du modèle MLflow.
    def load_model(self) -> None:
        # Résolution de l'URI du meilleur modèle.
        self.model_uri = self.resolve_model_uri()

        # Chargement du modèle pyfunc.
        self.pyfunc_model = mlflow.pyfunc.load_model(self.model_uri)

        # Tentative de chargement en flavor sklearn.
        try:
            # Chargement sklearn.
            self.native_model = mlflow.sklearn.load_model(self.model_uri)

        # Si ce n'est pas un modèle sklearn, tentative XGBoost.
        except Exception:
            try:
                # Chargement XGBoost.
                self.native_model = mlflow.xgboost.load_model(self.model_uri)

            # Si aucun flavor natif n'est disponible.
            except Exception:
                # On garde seulement pyfunc.
                self.native_model = None

    # Chargement complet du modèle et du preprocessing.
    def load(self) -> None:
        # Log de début.
        logger.info("Chargement du preprocessing...")

        # Chargement des objets de preprocessing.
        self.load_preprocessing_objects()

        # Log de début modèle.
        logger.info("Chargement du modèle depuis MLflow...")

        # Chargement du modèle.
        self.load_model()

        # Log de succès.
        logger.info("Modèle chargé : %s", self.model_uri)

    # Vérification si le modèle est chargé.
    def is_loaded(self) -> bool:
        # Retourne True si au moins un modèle est disponible.
        return self.pyfunc_model is not None

    # Préparation des features avant prédiction.
    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        # Création d'un DataFrame avec une seule ligne.
        df = pd.DataFrame([input_data])

        # Réorganisation des colonnes dans le bon ordre.
        X = df[FEATURE_COLUMNS].copy()

        # Récupération de l'imputer.
        imputer = self.preprocessing_objects["imputer"]

        # Récupération des bornes IQR.
        iqr_bounds = self.preprocessing_objects["iqr_bounds"]

        # Récupération du scaler.
        scaler = self.preprocessing_objects["scaler"]

        # Application de l'imputer.
        X_imputed = pd.DataFrame(
            imputer.transform(X),
            columns=FEATURE_COLUMNS,
        )

        # Application du clipping IQR.
        for column, limits in iqr_bounds.items():
            # Limitation des valeurs extrêmes.
            X_imputed[column] = X_imputed[column].clip(
                lower=limits["lower"],
                upper=limits["upper"],
            )

        # Application du scaler si disponible.
        if scaler is not None:
            # Transformation avec le scaler appris pendant le preprocessing.
            X_processed = pd.DataFrame(
                scaler.transform(X_imputed),
                columns=FEATURE_COLUMNS,
            )

        # Si aucun scaler n'existe.
        else:
            # On garde les données imputées et clippées.
            X_processed = X_imputed

        # Retour des features transformées.
        return X_processed

    # Prédiction.
    def predict(self, input_data: dict) -> dict:
        # Vérification du chargement.
        if not self.is_loaded():
            # Erreur si le modèle n'est pas chargé.
            raise RuntimeError("Le modèle n'est pas chargé.")

        # Préparation des données.
        X_processed = self.preprocess_input(input_data)

        # Utilisation du modèle natif si disponible.
        model_for_prediction = self.native_model if self.native_model else self.pyfunc_model

        # Prédiction de la classe.
        prediction = model_for_prediction.predict(X_processed)

        # Conversion de la classe prédite en int.
        predicted_class = int(np.asarray(prediction)[0])

        # Récupération du label texte.
        predicted_label = CLASS_LABELS.get(predicted_class, "unknown")

        # Initialisation des probabilités.
        probabilities = None

        # Si le modèle natif supporte predict_proba.
        if self.native_model is not None and hasattr(self.native_model, "predict_proba"):
            # Calcul des probabilités.
            proba = self.native_model.predict_proba(X_processed)[0]

            # Conversion en dictionnaire lisible.
            probabilities = {
                CLASS_LABELS[index]: float(value)
                for index, value in enumerate(proba)
            }

        # Retour de la prédiction complète.
        return {
            "model_name": MLFLOW_MODEL_NAME,
            "model_version": self.model_version,
            "model_alias": self.model_alias,
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "probabilities": probabilities,
        }


# Instance globale du loader.
model_loader = ModelLoader()
