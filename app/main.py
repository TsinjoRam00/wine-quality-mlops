# app/main.py

# Importation du module logging.
import logging

# Importation du context manager asynchrone pour gérer le démarrage de l'API.
from contextlib import asynccontextmanager

# Importation de FastAPI.
from fastapi import FastAPI

# Importation de BackgroundTasks pour lancer un entraînement sans bloquer la réponse.
from fastapi import BackgroundTasks

# Importation de HTTPException pour retourner des erreurs propres.
from fastapi import HTTPException

# Importation du client MLflow.
from mlflow.tracking import MlflowClient

# Importation de la configuration.
from app.config import MLFLOW_MODEL_NAME
from app.config import MLFLOW_TRACKING_URI

# Importation du loader de modèle.
from app.model_loader import model_loader

# Importation des schémas.
from app.schemas import HealthResponse
from app.schemas import MetricsResponse
from app.schemas import PredictionResponse
from app.schemas import TrainRequest
from app.schemas import TrainResponse
from app.schemas import WineFeatures

# Importation des services.
from app.services import configure_mlflow
from app.services import get_best_model_metrics
from app.services import list_registered_models
from app.services import training_job
from app.services import training_status


# Configuration basique des logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

# Création du logger.
logger = logging.getLogger(__name__)


# Fonction exécutée au démarrage et à l'arrêt de l'API.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Log de démarrage.
    logger.info("Démarrage de l'API Wine Quality MLOps.")

    # Tentative de chargement automatique du meilleur modèle.
    try:
        # Chargement du modèle.
        model_loader.load()

    # Si le modèle n'est pas encore disponible.
    except Exception as error:
        # Log non bloquant.
        logger.warning("Modèle non chargé au démarrage : %s", error)

    # L'API démarre même si le modèle n'est pas encore disponible.
    yield

    # Log d'arrêt.
    logger.info("Arrêt de l'API Wine Quality MLOps.")


# Création de l'application FastAPI.
app = FastAPI(
    title="Wine Quality MLOps API",
    description="API REST professionnelle pour entraîner et prédire la qualité du vin avec MLflow.",
    version="1.0.0",
    lifespan=lifespan,
)


# Route racine.
@app.get("/")
def root():
    # Retour des informations principales.
    return {
        "message": "Bienvenue sur l'API Wine Quality MLOps.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "train": "/train",
        "models": "/models",
        "metrics": "/metrics",
    }


# Route de santé.
@app.get("/health", response_model=HealthResponse)
def health():
    # Statut MLflow par défaut.
    mlflow_status = "unknown"

    # Tentative de connexion à MLflow.
    try:
        # Configuration MLflow.
        configure_mlflow()

        # Création du client.
        client = MlflowClient()

        # Appel simple pour vérifier MLflow.
        client.search_experiments()

        # Si tout va bien.
        mlflow_status = "ok"

    # Si MLflow ne répond pas.
    except Exception:
        # Statut erreur.
        mlflow_status = "error"

    # Statut modèle.
    model_status = "loaded" if model_loader.is_loaded() else "not_loaded"

    # Statut global.
    global_status = "ok" if mlflow_status == "ok" else "degraded"

    # Retour de santé.
    return {
        "status": global_status,
        "mlflow_status": mlflow_status,
        "model_status": model_status,
        "model_name": MLFLOW_MODEL_NAME,
    }


# Route pour lancer un entraînement.
@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    # Si un entraînement est déjà en cours.
    if training_status["status"] == "running":
        # Retourne une erreur 409.
        raise HTTPException(
            status_code=409,
            detail="Un entraînement est déjà en cours.",
        )

    # Ajout de la tâche d'entraînement en arrière-plan.
    background_tasks.add_task(
        training_job,
        request.grid_size,
        request.cv,
        request.n_jobs,
        request.register_model,
    )

    # Réponse immédiate.
    return {
        "status": "started",
        "message": "L'entraînement a été lancé en arrière-plan.",
    }


# Route de prédiction.
@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures):
    # Si le modèle n'est pas chargé.
    if not model_loader.is_loaded():
        # Tentative de rechargement.
        try:
            # Chargement du modèle.
            model_loader.load()

        # Si impossible de charger.
        except Exception as error:
            # Erreur propre.
            raise HTTPException(
                status_code=503,
                detail=f"Modèle indisponible : {error}",
            )

    # Conversion des features Pydantic en dictionnaire.
    input_data = features.model_dump()

    # Prédiction.
    result = model_loader.predict(input_data)

    # Retour du résultat.
    return result


# Route pour lister les modèles MLflow.
@app.get("/models")
def models():
    # Tentative de récupération des modèles.
    try:
        # Retour des modèles.
        return {
            "models": list_registered_models(),
            "training_status": training_status,
        }

    # Si erreur MLflow.
    except Exception as error:
        # Erreur propre.
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de récupérer les modèles : {error}",
        )


# Route pour afficher les métriques du meilleur modèle.
@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    # Tentative de récupération des métriques.
    try:
        # Retour des métriques.
        return get_best_model_metrics()

    # Si erreur.
    except Exception as error:
        # Erreur propre.
        raise HTTPException(
            status_code=503,
            detail=f"Impossible de récupérer les métriques : {error}",
        )
