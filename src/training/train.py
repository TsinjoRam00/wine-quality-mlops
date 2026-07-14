# src/training/train.py

# Importation du module argparse pour lire les arguments en ligne de commande.
import argparse

# Importation du module json pour sauvegarder certains résultats.
import json

# Importation de Path pour gérer les chemins proprement.
from pathlib import Path

# Importation de joblib pour sauvegarder le meilleur modèle localement.
import joblib

# Importation de NumPy pour les calculs numériques.
import numpy as np

# Importation de Pandas pour lire les datasets préparés.
import pandas as pd

# Importation de MLflow pour tracker les expériences.
import mlflow

# Importation du module MLflow sklearn pour sauvegarder les modèles sklearn.
import mlflow.sklearn
import mlflow.xgboost
# Importation du client MLflow pour gérer le Model Registry.
from mlflow.tracking import MlflowClient

# Importation de StratifiedKFold pour la validation croisée stratifiée.
from sklearn.model_selection import StratifiedKFold

# Importation de GridSearchCV pour optimiser les hyperparamètres.
from sklearn.model_selection import GridSearchCV

# Importation de LogisticRegression.
from sklearn.linear_model import LogisticRegression

# Importation de DecisionTreeClassifier.
from sklearn.tree import DecisionTreeClassifier

# Importation de RandomForestClassifier.
from sklearn.ensemble import RandomForestClassifier

# Importation de XGBClassifier depuis XGBoost.
from xgboost import XGBClassifier

# Importation de la fonction d'évaluation personnalisée.
from src.evaluation.evaluate import evaluate_and_save_model

# Importation de la fonction pour sauvegarder le graphique comparatif.
from src.evaluation.evaluate import save_model_comparison_plot

# Importation de la fonction pour sauvegarder du JSON.
from src.evaluation.evaluate import save_json


# Définition des colonnes features.
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

# Définition de la colonne cible.
TARGET_COLUMN = "quality_class"

# Définition du nom du modèle dans le registre MLflow.
DEFAULT_REGISTERED_MODEL_NAME = "WineQualityClassifier"


# Fonction pour créer les dossiers nécessaires.
def ensure_directories(models_dir: Path, reports_dir: Path, figures_dir: Path) -> None:
    # Création du dossier des modèles si nécessaire.
    models_dir.mkdir(parents=True, exist_ok=True)

    # Création du dossier des rapports si nécessaire.
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Création du dossier des figures si nécessaire.
    figures_dir.mkdir(parents=True, exist_ok=True)


# Fonction pour lire un fichier CSV préparé.
def load_processed_split(file_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    # Lecture du fichier CSV.
    df = pd.read_csv(file_path)

    # Vérification que la cible existe.
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La colonne cible {TARGET_COLUMN} est absente dans {file_path}")

    # Sélection des features.
    X = df[FEATURE_COLUMNS].copy()

    # Sélection de la cible.
    y = df[TARGET_COLUMN].astype(int).copy()

    # Retour des features et de la cible.
    return X, y


# Fonction pour charger train, validation et test.
def load_processed_datasets(processed_dir: Path):
    # Chargement du jeu d'entraînement.
    X_train, y_train = load_processed_split(processed_dir / "train.csv")

    # Chargement du jeu de validation.
    X_val, y_val = load_processed_split(processed_dir / "validation.csv")

    # Chargement du jeu de test.
    X_test, y_test = load_processed_split(processed_dir / "test.csv")

    # Retour des trois jeux de données.
    return X_train, X_val, X_test, y_train, y_val, y_test


# Fonction pour définir les modèles et leurs hyperparamètres.
def build_models_and_param_grids(random_state: int, grid_size: str) -> dict:
    # Création du modèle Logistic Regression.
    logistic_regression = LogisticRegression(
        max_iter=3000,
        random_state=random_state,
    )

    # Création du modèle Decision Tree.
    decision_tree = DecisionTreeClassifier(
        random_state=random_state,
    )

    # Création du modèle Random Forest.
    random_forest = RandomForestClassifier(
        random_state=random_state,
        n_jobs=1,
    )

    # Création du modèle XGBoost.
    xgboost = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )

    # Si on veut une grille rapide pour ordinateur personnel.
    if grid_size == "fast":
        # Définition des grilles rapides.
        param_grids = {
            "logistic_regression": {
                "C": [0.1, 1.0, 10.0],
                "class_weight": [None, "balanced"],
            },
            "decision_tree": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "class_weight": [None, "balanced"],
            },
            "random_forest": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "class_weight": [None, "balanced"],
            },
            "xgboost": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        }

    # Si on veut une grille plus complète.
    else:
        # Définition des grilles plus larges.
        param_grids = {
            "logistic_regression": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "class_weight": [None, "balanced"],
            },
            "decision_tree": {
                "max_depth": [3, 5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"],
                "class_weight": [None, "balanced"],
            },
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": [None, "balanced"],
            },
            "xgboost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
                "reg_lambda": [1.0, 2.0],
            },
        }

    # Association des noms de modèles aux objets modèles.
    models = {
        "logistic_regression": logistic_regression,
        "decision_tree": decision_tree,
        "random_forest": random_forest,
        "xgboost": xgboost,
    }

    # Création du dictionnaire final.
    model_configs = {}

    # Boucle sur chaque modèle.
    for model_name, model in models.items():
        # Ajout du modèle et de sa grille.
        model_configs[model_name] = {
            "model": model,
            "param_grid": param_grids[model_name],
        }

    # Retour des configurations.
    return model_configs


# Fonction pour définir les métriques utilisées pendant GridSearchCV.
def build_scoring_metrics() -> dict:
    # Création du dictionnaire des métriques.
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    }

    # Retour des métriques.
    return scoring


# Fonction pour sauvegarder les résultats de validation croisée.
def save_cv_results(grid_search: GridSearchCV, output_path: Path) -> None:
    # Création du dossier parent.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Conversion des résultats de GridSearchCV en DataFrame.
    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    # Tri des résultats par le rang du F1-score macro.
    cv_results_df = cv_results_df.sort_values("rank_test_f1_macro")

    # Sauvegarde des résultats en CSV.
    cv_results_df.to_csv(output_path, index=False)


# Fonction pour logger les métriques dans MLflow avec un préfixe.
def log_metrics_with_prefix(metrics: dict, prefix: str) -> None:
    # Boucle sur chaque métrique.
    for metric_name, metric_value in metrics.items():
        # Log de la métrique dans MLflow.
        mlflow.log_metric(f"{prefix}_{metric_name}", float(metric_value))


# Fonction pour logger les meilleurs hyperparamètres dans MLflow.
def log_best_params(best_params: dict) -> None:
    # Boucle sur chaque hyperparamètre.
    for param_name, param_value in best_params.items():
        # Log du paramètre dans MLflow.
        mlflow.log_param(f"best_{param_name}", param_value)


# Fonction pour entraîner un modèle avec GridSearchCV.
def train_one_model(
    model_name: str,
    model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int,
    n_jobs: int,
    random_state: int,
    reports_dir: Path,
    figures_dir: Path,
) -> dict:
    # Affichage du modèle en cours.
    print(f"\nEntraînement du modèle : {model_name}")

    # Création de la validation croisée stratifiée.
    cross_validator = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    # Création des métriques de scoring.
    scoring = build_scoring_metrics()

    # Création de GridSearchCV.
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cross_validator,
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=1,
    )

    # Entraînement avec recherche d'hyperparamètres.
    grid_search.fit(X_train, y_train)

    # Récupération du meilleur modèle.
    best_model = grid_search.best_estimator_

    # Évaluation sur validation.
    validation_report = evaluate_and_save_model(
        model=best_model,
        X=X_val,
        y=y_val,
        model_name=model_name,
        split_name="validation",
        reports_dir=reports_dir,
        figures_dir=figures_dir,
    )

    # Évaluation sur test.
    test_report = evaluate_and_save_model(
        model=best_model,
        X=X_test,
        y=y_test,
        model_name=model_name,
        split_name="test",
        reports_dir=reports_dir,
        figures_dir=figures_dir,
    )

    # Définition du chemin des résultats de validation croisée.
    cv_results_path = reports_dir / f"{model_name}_cv_results.csv"

    # Sauvegarde des résultats de validation croisée.
    save_cv_results(grid_search, cv_results_path)

    # Création du résumé du modèle.
    summary = {
        "model_name": model_name,
        "best_estimator": best_model,
        "best_params": grid_search.best_params_,
        "best_cv_f1_macro": float(grid_search.best_score_),
        "validation_metrics": validation_report["metrics"],
        "test_metrics": test_report["metrics"],
        "validation_report_path": validation_report["report_path"],
        "test_report_path": test_report["report_path"],
        "validation_confusion_matrix_path": validation_report["confusion_matrix_path"],
        "test_confusion_matrix_path": test_report["confusion_matrix_path"],
        "validation_roc_curve_path": validation_report["roc_curve_path"],
        "test_roc_curve_path": test_report["roc_curve_path"],
        "cv_results_path": str(cv_results_path),
    }

    # Retour du résumé.
    return summary


# Fonction pour enregistrer un modèle dans le registry MLflow.
def register_best_model(best_run_id: str, registered_model_name: str) -> None:
    # Création de l'URI du modèle depuis le run MLflow.
    model_uri = f"runs:/{best_run_id}/model"

    # Affichage de l'action.
    print(f"\nEnregistrement du meilleur modèle dans MLflow Model Registry : {registered_model_name}")

    # Enregistrement du modèle.
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    # Création du client MLflow.
    client = MlflowClient()

    # Tentative d'ajout de l'alias champion.
    try:
        # Attribution de l'alias champion à la version enregistrée.
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="champion",
            version=model_version.version,
        )

        # Affichage du succès.
        print(f"Alias 'champion' ajouté à la version {model_version.version}.")

    # Si l'alias n'est pas supporté par la version MLflow utilisée.
    except Exception as error:
        # Affichage d'un avertissement non bloquant.
        print(f"Alias non ajouté, mais le modèle est enregistré. Détail : {error}")


# Fonction principale d'entraînement.
def run_training(
    processed_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    mlflow_tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    cv: int,
    n_jobs: int,
    random_state: int,
    grid_size: str,
    register_model: bool,
) -> None:
    # Création des dossiers de sortie.
    ensure_directories(models_dir, reports_dir, figures_dir)

    # Chargement des données préparées.
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_datasets(processed_dir)

    # Configuration du tracking URI MLflow.
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Configuration du registry URI MLflow.
    mlflow.set_registry_uri(mlflow_tracking_uri)

    # Définition ou création de l'expérience MLflow.
    mlflow.set_experiment(experiment_name)

    # Création des modèles et grilles.
    model_configs = build_models_and_param_grids(
        random_state=random_state,
        grid_size=grid_size,
    )

    # Liste pour stocker les résultats de comparaison.
    comparison_rows = []

    # Variable pour stocker le meilleur modèle global.
    best_global_model = None

    # Variable pour stocker le nom du meilleur modèle global.
    best_global_model_name = None

    # Variable pour stocker le meilleur F1-score de validation.
    best_global_validation_f1 = -1.0

    # Variable pour stocker le run id MLflow du meilleur modèle.
    best_global_run_id = None

    # Boucle sur chaque modèle.
    for model_name, config in model_configs.items():
        # Ouverture d'un run MLflow.
        with mlflow.start_run(run_name=model_name) as run:
            # Log du nom du modèle.
            mlflow.log_param("model_name", model_name)

            # Log de la taille de grille.
            mlflow.log_param("grid_size", grid_size)

            # Log du nombre de folds.
            mlflow.log_param("cv", cv)

            # Log du random state.
            mlflow.log_param("random_state", random_state)

            # Entraînement du modèle.
            summary = train_one_model(
                model_name=model_name,
                model=config["model"],
                param_grid=config["param_grid"],
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                cv=cv,
                n_jobs=n_jobs,
                random_state=random_state,
                reports_dir=reports_dir,
                figures_dir=figures_dir,
            )

            # Log des meilleurs hyperparamètres.
            log_best_params(summary["best_params"])

            # Log du meilleur score CV.
            mlflow.log_metric("best_cv_f1_macro", summary["best_cv_f1_macro"])

            # Log des métriques de validation.
            log_metrics_with_prefix(summary["validation_metrics"], "validation")

            # Log des métriques de test.
            log_metrics_with_prefix(summary["test_metrics"], "test")

            # Log du rapport validation.
            mlflow.log_artifact(summary["validation_report_path"], artifact_path="reports")

            # Log du rapport test.
            mlflow.log_artifact(summary["test_report_path"], artifact_path="reports")

            # Log des résultats de CV.
            mlflow.log_artifact(summary["cv_results_path"], artifact_path="reports")

            # Log de la matrice de confusion validation.
            mlflow.log_artifact(summary["validation_confusion_matrix_path"], artifact_path="figures")

            # Log de la matrice de confusion test.
            mlflow.log_artifact(summary["test_confusion_matrix_path"], artifact_path="figures")

            # Log de la courbe ROC validation.
            mlflow.log_artifact(summary["validation_roc_curve_path"], artifact_path="figures")

            # Log de la courbe ROC test.
            mlflow.log_artifact(summary["test_roc_curve_path"], artifact_path="figures")

            # Création d'un exemple d'entrée pour MLflow.
            input_example = X_train.head(5)

            # Log du modèle dans MLflow.
            if model_name == "xgboost":
                mlflow.xgboost.log_model(
                    xgb_model=summary["best_estimator"],
                    artifact_path="model",
                    input_example=input_example,
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=summary["best_estimator"],
                    artifact_path="model",
                    input_example=input_example,
                )

            # Création d'une ligne de comparaison.
            comparison_row = {
                "model_name": model_name,
                "best_cv_f1_macro": summary["best_cv_f1_macro"],
                "validation_accuracy": summary["validation_metrics"]["accuracy"],
                "validation_precision_macro": summary["validation_metrics"]["precision_macro"],
                "validation_recall_macro": summary["validation_metrics"]["recall_macro"],
                "validation_f1_macro": summary["validation_metrics"]["f1_macro"],
                "validation_roc_auc_ovr_weighted": summary["validation_metrics"]["roc_auc_ovr_weighted"],
                "test_accuracy": summary["test_metrics"]["accuracy"],
                "test_precision_macro": summary["test_metrics"]["precision_macro"],
                "test_recall_macro": summary["test_metrics"]["recall_macro"],
                "test_f1_macro": summary["test_metrics"]["f1_macro"],
                "test_roc_auc_ovr_weighted": summary["test_metrics"]["roc_auc_ovr_weighted"],
                "best_params": json.dumps(summary["best_params"]),
                "mlflow_run_id": run.info.run_id,
            }

            # Ajout de la ligne au tableau de comparaison.
            comparison_rows.append(comparison_row)

            # Récupération du F1-score de validation.
            validation_f1 = summary["validation_metrics"]["f1_macro"]

            # Si ce modèle est le meilleur jusqu'ici.
            if validation_f1 > best_global_validation_f1:
                # Mise à jour du meilleur F1.
                best_global_validation_f1 = validation_f1

                # Mise à jour du meilleur modèle.
                best_global_model = summary["best_estimator"]

                # Mise à jour du nom du meilleur modèle.
                best_global_model_name = model_name

                # Mise à jour du run id.
                best_global_run_id = run.info.run_id

    # Conversion des résultats en DataFrame.
    comparison_df = pd.DataFrame(comparison_rows)

    # Tri par F1-score validation décroissant.
    comparison_df = comparison_df.sort_values("validation_f1_macro", ascending=False)

    # Définition du chemin du fichier de comparaison.
    comparison_path = reports_dir / "model_comparison.csv"

    # Sauvegarde de la comparaison.
    comparison_df.to_csv(comparison_path, index=False)

    # Définition du chemin du graphique de comparaison.
    comparison_plot_path = figures_dir / "model_comparison.png"

    # Sauvegarde du graphique de comparaison.
    save_model_comparison_plot(comparison_df, comparison_plot_path)

    # Sauvegarde locale du meilleur modèle.
    best_model_path = models_dir / "best_model.joblib"

    # Sauvegarde du meilleur modèle avec joblib.
    joblib.dump(best_global_model, best_model_path)

    # Création du résumé final.
    final_summary = {
        "best_model_name": best_global_model_name,
        "best_validation_f1_macro": best_global_validation_f1,
        "best_model_path": str(best_model_path),
        "best_mlflow_run_id": best_global_run_id,
        "comparison_path": str(comparison_path),
        "comparison_plot_path": str(comparison_plot_path),
    }

    # Sauvegarde du résumé final.
    save_json(final_summary, reports_dir / "training_summary.json")

    # Enregistrement dans le Model Registry si demandé.
    if register_model and best_global_run_id is not None:
        register_best_model(
            best_run_id=best_global_run_id,
            registered_model_name=registered_model_name,
        )

    # Affichage du tableau de comparaison.
    print("\nComparaison des modèles :")
    print(comparison_df)

    # Affichage du meilleur modèle.
    print(f"\nMeilleur modèle : {best_global_model_name}")

    # Affichage du meilleur F1-score.
    print(f"Meilleur F1 validation : {best_global_validation_f1:.4f}")

    # Affichage du chemin du meilleur modèle.
    print(f"Modèle sauvegardé dans : {best_model_path}")

    # Affichage du chemin du rapport de comparaison.
    print(f"Comparaison sauvegardée dans : {comparison_path}")


# Fonction pour lire les arguments CLI.
def parse_args() -> argparse.Namespace:
    # Création du parser.
    parser = argparse.ArgumentParser(description="Training ML complet pour Wine Quality.")

    # Argument du dossier des données préparées.
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))

    # Argument du dossier des modèles.
    parser.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))

    # Argument du dossier des rapports.
    parser.add_argument("--reports-dir", type=Path, default=Path("artifacts/reports"))

    # Argument du dossier des figures.
    parser.add_argument("--figures-dir", type=Path, default=Path("artifacts/figures"))

    # Argument de l'URI MLflow.
    parser.add_argument("--mlflow-tracking-uri", type=str, default="http://localhost:5000")

    # Argument du nom d'expérience MLflow.
    parser.add_argument("--experiment-name", type=str, default="wine-quality-classification")

    # Argument du nom du modèle enregistré.
    parser.add_argument("--registered-model-name", type=str, default=DEFAULT_REGISTERED_MODEL_NAME)

    # Argument du nombre de folds CV.
    parser.add_argument("--cv", type=int, default=3)

    # Argument du nombre de jobs parallèles.
    parser.add_argument("--n-jobs", type=int, default=-1)

    # Argument du random state.
    parser.add_argument("--random-state", type=int, default=42)

    # Argument de la taille de grille.
    parser.add_argument("--grid-size", type=str, default="fast", choices=["fast", "full"])

    # Argument pour enregistrer le meilleur modèle dans MLflow Model Registry.
    parser.add_argument("--register-model", action="store_true")

    # Retour des arguments.
    return parser.parse_args()


# Point d'entrée du script.
if __name__ == "__main__":
    # Lecture des arguments.
    args = parse_args()

    # Lancement de l'entraînement.
    run_training(
        processed_dir=args.processed_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        grid_size=args.grid_size,
        register_model=args.register_model,
    )
