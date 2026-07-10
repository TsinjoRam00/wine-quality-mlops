# app/config.py

# Importation du module os pour lire les variables d'environnement.
import os

# URI du serveur MLflow.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# URI du registre MLflow.
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)

# Nom du modèle enregistré dans MLflow Model Registry.
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "WineQualityClassifier")

# Alias du meilleur modèle.
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")

# Chemin local vers les objets de preprocessing.
PREPROCESSING_OBJECTS_PATH = os.getenv(
    "PREPROCESSING_OBJECTS_PATH",
    "artifacts/preprocessing/preprocessing_objects.joblib",
)

# Dossier des données préparées.
DATA_PROCESSED_DIR = os.getenv("DATA_PROCESSED_DIR", "data/processed")

# Dossier local des modèles.
MODELS_DIR = os.getenv("MODELS_DIR", "artifacts/models")

# Dossier local des rapports.
REPORTS_DIR = os.getenv("REPORTS_DIR", "artifacts/reports")

# Dossier local des figures.
FIGURES_DIR = os.getenv("FIGURES_DIR", "artifacts/figures")

# Nom de l'expérience MLflow.
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "wine-quality-classification",
)

# Liste des colonnes attendues par le modèle.
FEATURE_COLUMNS = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

# Mapping numérique vers texte.
CLASS_LABELS = {
    0: "low",
    1: "medium",
    2: "high",
}
