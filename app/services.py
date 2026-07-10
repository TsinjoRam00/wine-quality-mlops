# app/services.py

# Importation de logging.
import logging

# Importation de Path.
from pathlib import Path

# Importation de MLflow.
import mlflow

# Importation du client MLflow.
from mlflow.tracking import MlflowClient

# Importation de la configuration.
from app.config import DATA_PROCESSED_DIR
from app.config import FIGURES_DIR
from app.config import MLFLOW_EXPERIMENT_NAME
from app.config import MLFLOW_MODEL_ALIAS
from app.config import MLFLOW_MODEL_NAME
from app.config import MLFLOW_REGISTRY_URI
from app.config import MLFLOW_TRACKING_URI
from app.config import MODELS_DIR
from app.config import REPORTS_DIR

# Importation du training.
from src.training.train import run_training


# Création du logger.
logger = logging.getLogger(__name__)

# Statut simple de l'entraînement.
training_status = {
    "status": "idle",
    "message": "Aucun entraînement en cours.",
}


# Configuration MLflow.
def configure_mlflow() -> None:
    # Définition du tracking URI.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Définition du registry URI.
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)


# Tâche d'entraînement exécutée en arrière-plan.
def training_job(grid_size: str, cv: int, n_jobs: int, register_model: bool) -> None:
    # Mise à jour du statut.
    training_status["status"] = "running"

    # Mise à jour du message.
    training_status["message"] = "Entraînement en cours."

    # Log de début.
    logger.info("Début de l'entraînement depuis l'API.")

    try:
        # Lancement du pipeline de training existant.
        run_training(
            processed_dir=Path(DATA_PROCESSED_DIR),
            models_dir=Path(MODELS_DIR),
            reports_dir=Path(REPORTS_DIR),
            figures_dir=Path(FIGURES_DIR),
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=MLFLOW_EXPERIMENT_NAME,
            registered_model_name=MLFLOW_MODEL_NAME,
            cv=cv,
            n_jobs=n_jobs,
            random_state=42,
            grid_size=grid_size,
            register_model=register_model,
        )

        # Mise à jour du statut si succès.
        training_status["status"] = "success"

        # Message de succès.
        training_status["message"] = "Entraînement terminé avec succès."

        # Log de succès.
        logger.info("Entraînement terminé avec succès.")

    except Exception as error:
        # Mise à jour du statut si erreur.
        training_status["status"] = "failed"

        # Message d'erreur.
        training_status["message"] = str(error)

        # Log de l'erreur.
        logger.exception("Erreur pendant l'entraînement : %s", error)


# Liste les modèles enregistrés dans MLflow.
def list_registered_models() -> list[dict]:
    # Configuration MLflow.
    configure_mlflow()

    # Création du client.
    client = MlflowClient()

    # Récupération des modèles enregistrés.
    registered_models = client.search_registered_models()

    # Liste de sortie.
    results = []

    # Boucle sur chaque modèle.
    for registered_model in registered_models:
        # Récupération des versions.
        versions = [
            str(version.version)
            for version in registered_model.latest_versions
        ]

        # Ajout au résultat.
        results.append(
            {
                "name": registered_model.name,
                "versions": versions,
            }
        )

    # Retour de la liste.
    return results


# Récupère les métriques du meilleur modèle.
def get_best_model_metrics() -> dict:
    # Configuration MLflow.
    configure_mlflow()

    # Création du client.
    client = MlflowClient()

    # Initialisation de la version modèle.
    model_version = None

    # Tentative avec alias champion.
    try:
        # Récupération de la version par alias.
        model_version = client.get_model_version_by_alias(
            name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_MODEL_ALIAS,
        )

    # Si alias indisponible.
    except Exception:
        # Recherche des versions.
        versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")

        # Si aucune version n'est disponible.
        if not versions:
            return {
                "model_name": MLFLOW_MODEL_NAME,
                "model_version": None,
                "run_id": None,
                "metrics": {},
            }

        # Récupération de la dernière version.
        model_version = sorted(
            versions,
            key=lambda item: int(item.version),
            reverse=True,
        )[0]

    # Récupération du run associé au modèle.
    run = client.get_run(model_version.run_id)

    # Retour des métriques.
    return {
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": str(model_version.version),
        "run_id": model_version.run_id,
        "metrics": run.data.metrics,
    }
