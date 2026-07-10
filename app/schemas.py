# app/schemas.py

# Importation de BaseModel pour créer des schémas de données.
from pydantic import BaseModel

# Importation de Field pour ajouter des validations et exemples.
from pydantic import Field

# Importation de Literal pour limiter certaines valeurs possibles.
from typing import Literal

# Importation d'Optional pour les champs optionnels.
from typing import Optional

# Importation de Dict pour les dictionnaires typés.
from typing import Dict

# Importation de List pour les listes typées.
from typing import List


# Schéma d'entrée pour une prédiction.
class WineFeatures(BaseModel):
    # Acidité fixe du vin.
    fixed_acidity: float = Field(..., example=7.4)

    # Acidité volatile du vin.
    volatile_acidity: float = Field(..., example=0.70)

    # Acide citrique.
    citric_acid: float = Field(..., example=0.00)

    # Sucre résiduel.
    residual_sugar: float = Field(..., example=1.9)

    # Chlorures.
    chlorides: float = Field(..., example=0.076)

    # Dioxyde de soufre libre.
    free_sulfur_dioxide: float = Field(..., example=11.0)

    # Dioxyde de soufre total.
    total_sulfur_dioxide: float = Field(..., example=34.0)

    # Densité.
    density: float = Field(..., example=0.9978)

    # Potentiel hydrogène.
    ph: float = Field(..., ge=0, le=14, example=3.51)

    # Sulfates.
    sulphates: float = Field(..., example=0.56)

    # Alcool.
    alcohol: float = Field(..., example=9.4)


# Réponse de prédiction.
class PredictionResponse(BaseModel):
    # Nom du modèle utilisé.
    model_name: str

    # Version du modèle MLflow.
    model_version: Optional[str]

    # Alias MLflow utilisé.
    model_alias: Optional[str]

    # Classe prédite sous forme numérique.
    predicted_class: int

    # Classe prédite sous forme texte.
    predicted_label: str

    # Probabilités par classe si disponibles.
    probabilities: Optional[Dict[str, float]] = None


# Requête pour lancer un entraînement.
class TrainRequest(BaseModel):
    # Taille de la grille d'hyperparamètres.
    grid_size: Literal["fast", "full"] = "fast"

    # Nombre de folds pour la validation croisée.
    cv: int = 3

    # Nombre de jobs parallèles.
    n_jobs: int = 1

    # Enregistrer ou non le modèle dans MLflow Model Registry.
    register_model: bool = True


# Réponse de lancement d'entraînement.
class TrainResponse(BaseModel):
    # Statut de la tâche.
    status: str

    # Message explicatif.
    message: str


# Réponse santé API.
class HealthResponse(BaseModel):
    # Statut global.
    status: str

    # Statut MLflow.
    mlflow_status: str

    # Statut du modèle.
    model_status: str

    # Nom du modèle.
    model_name: str


# Réponse de liste des modèles.
class ModelInfo(BaseModel):
    # Nom du modèle.
    name: str

    # Versions disponibles.
    versions: List[str]


# Réponse des métriques.
class MetricsResponse(BaseModel):
    # Nom du modèle.
    model_name: str

    # Version du modèle.
    model_version: Optional[str]

    # Run ID MLflow.
    run_id: Optional[str]

    # Métriques du run.
    metrics: Dict[str, float]
