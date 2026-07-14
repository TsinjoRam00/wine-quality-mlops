# app/services.py

# Gestion des logs de l'application.
import logging

# Gestion des variables d'environnement transmises au sous-processus.
import os

# Permet de lancer le script d'entraînement comme une commande externe.
import subprocess

# Donne accès à l'interpréteur Python actuellement utilisé.
import sys

# Gestion des dates du statut d'entraînement.
from datetime import datetime, timezone

# Gestion propre du chemin racine du projet.
from pathlib import Path

# Évite de lancer plusieurs entraînements simultanément.
from threading import Lock

# Bibliothèque MLflow.
import mlflow

# Client permettant d'interroger MLflow et le Model Registry.
from mlflow.tracking import MlflowClient

# Configuration MLflow de l'application.
from app.config import MLFLOW_MODEL_ALIAS
from app.config import MLFLOW_MODEL_NAME
from app.config import MLFLOW_REGISTRY_URI
from app.config import MLFLOW_TRACKING_URI


# Racine du projet.
#
# app/services.py
#      └── parent : app
#             └── parent : racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Logger du module.
logger = logging.getLogger(__name__)


# Verrou empêchant deux entraînements de fonctionner en même temps.
training_lock = Lock()


# Statut global consultable par l'API.
training_status = {
    "status": "idle",
    "message": "Aucun entraînement en cours.",
    "started_at": None,
    "finished_at": None,
}


def utc_now() -> str:
    """
    Retourne la date et l'heure actuelles au format ISO 8601 en UTC.
    """

    return datetime.now(timezone.utc).isoformat()


def configure_mlflow() -> None:
    """
    Configure l'adresse du serveur MLflow Tracking et du Model Registry.
    """

    # Serveur utilisé pour les expériences, les runs et les métriques.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Serveur utilisé pour les modèles enregistrés.
    mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)


def training_job(
    grid_size: str,
    cv: int,
    n_jobs: int,
    register_model: bool,
) -> None:
    """
    Lance l'entraînement Machine Learning dans un sous-processus Python.

    Cette fonction est appelée en arrière-plan par la route POST /train.
    """

    # Empêche deux entraînements de commencer au même moment.
    lock_acquired = training_lock.acquire(blocking=False)

    if not lock_acquired:
        logger.warning("Un entraînement est déjà en cours.")
        return

    try:
        # Mise à jour du statut.
        training_status["status"] = "running"
        training_status["message"] = "Entraînement en cours."
        training_status["started_at"] = utc_now()
        training_status["finished_at"] = None

        logger.info("Début de l'entraînement depuis l'API.")

        # Construction de la commande.
        #
        # sys.executable représente le Python actuellement utilisé :
        # - Python du venv en local ;
        # - Python du conteneur dans Docker.
        command = [
            sys.executable,
            "-m",
            "src.training.train",
            "--grid-size",
            grid_size,
            "--cv",
            str(cv),
            "--n-jobs",
            str(n_jobs),
        ]

        # Ajout optionnel de l'enregistrement dans MLflow Registry.
        if register_model:
            command.append("--register-model")

        # Copie des variables d'environnement du processus FastAPI.
        environment = os.environ.copy()

        # Garantit que Python peut retrouver les packages app et src.
        environment["PYTHONPATH"] = str(PROJECT_ROOT)

        logger.info("Commande d'entraînement : %s", " ".join(command))

        # Exécution du script d'entraînement.
        #
        # check=True provoque une exception si la commande se termine
        # avec un code différent de zéro.
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )

        # Après un entraînement réussi, tentative de rechargement
        # du nouveau meilleur modèle dans FastAPI.
        try:
            # Import local pour éviter une dépendance circulaire au démarrage.
            from app.model_loader import model_loader

            model_loader.load()

            logger.info(
                "Le meilleur modèle MLflow a été rechargé dans l'API."
            )

        except Exception as reload_error:
            # Le training reste considéré comme réussi même si le
            # rechargement immédiat échoue.
            logger.warning(
                "Entraînement réussi, mais rechargement du modèle impossible : %s",
                reload_error,
            )

        # Statut final en cas de succès.
        training_status["status"] = "success"
        training_status["message"] = "Entraînement terminé avec succès."
        training_status["finished_at"] = utc_now()

        logger.info("Entraînement terminé avec succès.")

    except subprocess.CalledProcessError as error:
        # Erreur spécifique lorsque le script d'entraînement retourne
        # un code de sortie différent de zéro.
        training_status["status"] = "failed"
        training_status["message"] = (
            f"Le script d'entraînement a échoué avec le code "
            f"{error.returncode}."
        )
        training_status["finished_at"] = utc_now()

        logger.exception(
            "Échec du script d'entraînement avec le code %s.",
            error.returncode,
        )

    except Exception as error:
        # Gestion de toute autre erreur.
        training_status["status"] = "failed"
        training_status["message"] = str(error)
        training_status["finished_at"] = utc_now()

        logger.exception(
            "Erreur pendant l'entraînement : %s",
            error,
        )

    finally:
        # Libération du verrou dans tous les cas.
        training_lock.release()


def list_registered_models() -> list[dict]:
    """
    Retourne la liste des modèles enregistrés dans MLflow Registry,
    ainsi que toutes leurs versions.
    """

    # Configuration de MLflow.
    configure_mlflow()

    # Création du client MLflow.
    client = MlflowClient()

    # Recherche des modèles enregistrés.
    registered_models = client.search_registered_models()

    # Liste qui contiendra la réponse finale.
    results = []

    # Parcours des modèles.
    for registered_model in registered_models:
        # Recherche de toutes les versions de ce modèle.
        model_versions = client.search_model_versions(
            f"name='{registered_model.name}'"
        )

        # Tri des versions de la plus récente à la plus ancienne.
        sorted_versions = sorted(
            model_versions,
            key=lambda model_version: int(model_version.version),
            reverse=True,
        )

        # Préparation des informations du modèle.
        results.append(
            {
                "name": registered_model.name,
                "versions": [
                    str(model_version.version)
                    for model_version in sorted_versions
                ],
            }
        )

    return results


def get_best_model_metrics() -> dict:
    """
    Récupère les métriques du modèle possédant l'alias champion.

    Si l'alias champion n'existe pas, la version la plus récente
    du modèle est utilisée.
    """

    # Configuration de MLflow.
    configure_mlflow()

    # Création du client MLflow.
    client = MlflowClient()

    model_version = None

    try:
        # Recherche prioritaire de la version possédant l'alias champion.
        model_version = client.get_model_version_by_alias(
            name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_MODEL_ALIAS,
        )

    except Exception as alias_error:
        logger.warning(
            "Alias '%s' indisponible pour '%s' : %s",
            MLFLOW_MODEL_ALIAS,
            MLFLOW_MODEL_NAME,
            alias_error,
        )

        # Recherche de toutes les versions existantes.
        versions = client.search_model_versions(
            f"name='{MLFLOW_MODEL_NAME}'"
        )

        # Aucun modèle enregistré.
        if not versions:
            return {
                "model_name": MLFLOW_MODEL_NAME,
                "model_version": None,
                "run_id": None,
                "metrics": {},
            }

        # Sélection de la version numérique la plus récente.
        model_version = max(
            versions,
            key=lambda version: int(version.version),
        )

    # Récupération du run ayant produit cette version.
    run = client.get_run(model_version.run_id)

    # Conversion explicite des métriques en float.
    metrics = {
        metric_name: float(metric_value)
        for metric_name, metric_value in run.data.metrics.items()
    }

    return {
        "model_name": MLFLOW_MODEL_NAME,
        "model_version": str(model_version.version),
        "run_id": model_version.run_id,
        "metrics": metrics,
    }