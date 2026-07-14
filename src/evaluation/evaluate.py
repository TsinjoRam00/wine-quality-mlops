# src/evaluation/evaluate.py

# Importation du module json pour sauvegarder les rapports au format JSON.
import json

# Importation de Path pour gérer proprement les chemins de fichiers.
from pathlib import Path

# Importation de NumPy pour les calculs numériques.
import numpy as np

# Importation de Pandas pour manipuler les tableaux de résultats.
import pandas as pd

# Importation de matplotlib pour créer et sauvegarder les graphiques.
import matplotlib

# Utilisation du backend Agg pour générer des figures sans interface graphique.
matplotlib.use("Agg")

# Importation de pyplot pour tracer les graphiques.
import matplotlib.pyplot as plt

# Importation des métriques de classification depuis scikit-learn.
from sklearn.metrics import accuracy_score

# Importation de precision_score pour calculer la précision.
from sklearn.metrics import precision_score

# Importation de recall_score pour calculer le rappel.
from sklearn.metrics import recall_score

# Importation de f1_score pour calculer le score F1.
from sklearn.metrics import f1_score

# Importation de classification_report pour créer un rapport détaillé.
from sklearn.metrics import classification_report

# Importation de confusion_matrix pour créer la matrice de confusion.
from sklearn.metrics import confusion_matrix

# Importation de ConfusionMatrixDisplay pour afficher la matrice de confusion.
from sklearn.metrics import ConfusionMatrixDisplay

# Importation de roc_curve pour calculer les courbes ROC.
from sklearn.metrics import roc_curve

# Importation de roc_auc_score pour calculer l'AUC ROC.
from sklearn.metrics import roc_auc_score

# Importation de label_binarize pour transformer la cible multiclasses en format binaire.
from sklearn.preprocessing import label_binarize


# Définition des labels numériques utilisés par le modèle.
TARGET_LABELS = [0, 1, 2]

# Définition des noms lisibles des classes.
TARGET_NAMES = ["low", "medium", "high"]


# Fonction utilitaire pour rendre les objets compatibles JSON.
def make_json_serializable(value):
    # Si la valeur est un type NumPy entier, on la convertit en int Python.
    if isinstance(value, np.integer):
        return int(value)

    # Si la valeur est un type NumPy flottant, on la convertit en float Python.
    if isinstance(value, np.floating):
        return float(value)

    # Si la valeur est un tableau NumPy, on le convertit en liste Python.
    if isinstance(value, np.ndarray):
        return value.tolist()

    # Sinon, on retourne la valeur telle quelle.
    return value


# Fonction pour sauvegarder un dictionnaire dans un fichier JSON.
def save_json(data: dict, output_path: Path) -> None:
    # Création du dossier parent si nécessaire.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ouverture du fichier JSON en écriture.
    with open(output_path, "w", encoding="utf-8") as file:
        # Écriture des données avec indentation.
        json.dump(data, file, indent=4, ensure_ascii=False, default=make_json_serializable)


# Fonction pour calculer les métriques principales.
def compute_classification_metrics(y_true, y_pred, y_proba) -> dict:
    # Calcul de l'accuracy.
    accuracy = accuracy_score(y_true, y_pred)

    # Calcul de la précision macro.
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)

    # Calcul du rappel macro.
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Calcul du F1-score macro.
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Calcul de la précision weighted.
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)

    # Calcul du rappel weighted.
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    # Calcul du F1-score weighted.
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Calcul de l'AUC ROC multiclasses avec stratégie One-vs-Rest.
    roc_auc_ovr_weighted = roc_auc_score(
        y_true,
        y_proba,
        multi_class="ovr",
        average="weighted",
    )

    # Création du dictionnaire des métriques.
    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc_ovr_weighted": roc_auc_ovr_weighted,
    }

    # Retour des métriques.
    return metrics


# Fonction pour créer et sauvegarder la matrice de confusion.
def save_confusion_matrix_plot(y_true, y_pred, model_name: str, output_path: Path) -> None:
    # Création du dossier de sortie.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calcul de la matrice de confusion.
    matrix = confusion_matrix(y_true, y_pred, labels=TARGET_LABELS)

    # Création de l'affichage de la matrice.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=TARGET_NAMES,
    )

    # Création d'une figure.
    fig, ax = plt.subplots(figsize=(7, 6))

    # Affichage de la matrice.
    display.plot(ax=ax, values_format="d")

    # Ajout du titre.
    ax.set_title(f"Confusion Matrix - {model_name}")

    # Ajustement de la mise en page.
    plt.tight_layout()

    # Sauvegarde de la figure.
    plt.savefig(output_path)

    # Fermeture de la figure pour éviter d'utiliser trop de mémoire.
    plt.close(fig)


# Fonction pour créer et sauvegarder les courbes ROC multiclasses.
def save_roc_curve_plot(y_true, y_proba, model_name: str, output_path: Path) -> None:
    # Création du dossier de sortie.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Transformation de y_true en format binaire One-vs-Rest.
    y_true_binarized = label_binarize(y_true, classes=TARGET_LABELS)

    # Création d'une figure.
    fig, ax = plt.subplots(figsize=(8, 6))

    # Boucle sur chaque classe.
    for class_index, class_name in enumerate(TARGET_NAMES):
        # Calcul de la courbe ROC pour cette classe.
        fpr, tpr, _ = roc_curve(
            y_true_binarized[:, class_index],
            y_proba[:, class_index],
        )

        # Calcul de l'AUC pour cette classe.
        auc_score = roc_auc_score(
            y_true_binarized[:, class_index],
            y_proba[:, class_index],
        )

        # Affichage de la courbe ROC.
        ax.plot(fpr, tpr, label=f"{class_name} AUC={auc_score:.3f}")

    # Ajout de la diagonale de référence.
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")

    # Ajout du titre.
    ax.set_title(f"ROC Curves - {model_name}")

    # Ajout du label de l'axe X.
    ax.set_xlabel("False Positive Rate")

    # Ajout du label de l'axe Y.
    ax.set_ylabel("True Positive Rate")

    # Ajout de la légende.
    ax.legend()

    # Ajout d'une grille.
    ax.grid(True)

    # Ajustement de la mise en page.
    plt.tight_layout()

    # Sauvegarde de la figure.
    plt.savefig(output_path)

    # Fermeture de la figure.
    plt.close(fig)


# Fonction complète pour évaluer un modèle et sauvegarder ses résultats.
def evaluate_and_save_model(
    model,
    X,
    y,
    model_name: str,
    split_name: str,
    reports_dir: Path,
    figures_dir: Path,
) -> dict:
    # Prédiction des classes.
    y_pred = model.predict(X)

    # Prédiction des probabilités de chaque classe.
    y_proba = model.predict_proba(X)

    # Calcul des métriques.
    metrics = compute_classification_metrics(y, y_pred, y_proba)

    # Création du rapport de classification sous forme de dictionnaire.
    report = classification_report(
        y,
        y_pred,
        labels=TARGET_LABELS,
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # Création de la matrice de confusion.
    matrix = confusion_matrix(y, y_pred, labels=TARGET_LABELS)

    # Préparation du rapport complet.
    full_report = {
        "model_name": model_name,
        "split_name": split_name,
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    # Définition du chemin JSON du rapport.
    report_path = reports_dir / f"{model_name}_{split_name}_report.json"

    # Sauvegarde du rapport JSON.
    save_json(full_report, report_path)

    # Définition du chemin de la figure de matrice de confusion.
    confusion_matrix_path = figures_dir / f"{model_name}_{split_name}_confusion_matrix.png"

    # Sauvegarde de la matrice de confusion.
    save_confusion_matrix_plot(y, y_pred, model_name, confusion_matrix_path)

    # Définition du chemin de la figure ROC.
    roc_curve_path = figures_dir / f"{model_name}_{split_name}_roc_curve.png"

    # Sauvegarde de la courbe ROC.
    save_roc_curve_plot(y, y_proba, model_name, roc_curve_path)

    # Ajout des chemins dans le rapport retourné.
    full_report["report_path"] = str(report_path)

    # Ajout du chemin de la matrice de confusion.
    full_report["confusion_matrix_path"] = str(confusion_matrix_path)

    # Ajout du chemin de la courbe ROC.
    full_report["roc_curve_path"] = str(roc_curve_path)

    # Retour du rapport.
    return full_report


# Fonction pour sauvegarder un graphique comparatif des modèles.
def save_model_comparison_plot(comparison_df: pd.DataFrame, output_path: Path) -> None:
    # Création du dossier de sortie.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sélection des colonnes de métriques importantes.
    metric_columns = [
        "validation_accuracy",
        "validation_precision_macro",
        "validation_recall_macro",
        "validation_f1_macro",
        "validation_roc_auc_ovr_weighted",
    ]

    # Création d'une figure.
    fig, ax = plt.subplots(figsize=(12, 7))

    # Création du graphique en barres.
    comparison_df.set_index("model_name")[metric_columns].plot(kind="bar", ax=ax)

    # Ajout du titre.
    ax.set_title("Comparaison des modèles sur le jeu de validation")

    # Ajout du label de l'axe X.
    ax.set_xlabel("Modèle")

    # Ajout du label de l'axe Y.
    ax.set_ylabel("Score")

    # Rotation des noms de modèles.
    plt.xticks(rotation=30, ha="right")

    # Ajout d'une grille.
    ax.grid(True, axis="y")

    # Ajustement de la mise en page.
    plt.tight_layout()

    # Sauvegarde de la figure.
    plt.savefig(output_path)

    # Fermeture de la figure.
    plt.close(fig)
