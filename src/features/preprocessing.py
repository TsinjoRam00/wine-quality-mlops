# src/features/preprocessing.py

# Importation du module argparse pour lire les arguments passés en ligne de commande.
import argparse

# Importation du module json pour sauvegarder les rapports de preprocessing au format JSON.
import json

# Importation de Path pour manipuler les chemins de fichiers proprement.
from pathlib import Path

# Importation de joblib pour sauvegarder les objets Python comme les scalers et imputers.
import joblib

# Importation de NumPy pour les opérations numériques.
import numpy as np

# Importation de Pandas pour charger, nettoyer et transformer les données tabulaires.
import pandas as pd

# Importation du train_test_split pour séparer les données en train, validation et test.
from sklearn.model_selection import train_test_split

# Importation de SimpleImputer pour remplacer les valeurs manquantes.
from sklearn.impute import SimpleImputer

# Importation de StandardScaler pour standardiser les variables.
from sklearn.preprocessing import StandardScaler

# Importation de MinMaxScaler pour normaliser les variables.
from sklearn.preprocessing import MinMaxScaler

# Définition des colonnes attendues dans les fichiers CSV originaux.
ORIGINAL_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]

# Définition des noms de colonnes propres utilisés dans notre projet.
CLEAN_COLUMNS = [
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
    "quality",
]

# Définition des colonnes utilisées comme features numériques du modèle.
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

# Définition du mapping entre les labels texte et les classes numériques.
TARGET_MAPPING = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


# Définition d'une fonction pour créer les dossiers nécessaires.
def ensure_directories(processed_dir: Path, artifacts_dir: Path) -> None:
    # Création du dossier des données transformées si celui-ci n'existe pas.
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Création du dossier des artefacts si celui-ci n'existe pas.
    artifacts_dir.mkdir(parents=True, exist_ok=True)


# Définition d'une fonction pour charger un fichier CSV Wine Quality.
def load_wine_file(file_path: Path, wine_type: str) -> pd.DataFrame:
    # Lecture du fichier CSV avec le séparateur officiel du dataset, qui est le point-virgule.
    df = pd.read_csv(file_path, sep=";")

    # Ajout d'une colonne permettant de savoir si le vin est rouge ou blanc.
    df["wine_type"] = wine_type

    # Retour du DataFrame chargé.
    return df


# Définition d'une fonction pour valider les colonnes du dataset brut.
def validate_raw_columns(df: pd.DataFrame) -> None:
    # Création d'une liste contenant les colonnes manquantes.
    missing_columns = [column for column in ORIGINAL_COLUMNS if column not in df.columns]

    # Si une ou plusieurs colonnes sont manquantes, on bloque le pipeline.
    if missing_columns:
        # Levée d'une erreur explicite avec les colonnes manquantes.
        raise ValueError(f"Colonnes manquantes dans le dataset brut : {missing_columns}")


# Définition d'une fonction pour renommer les colonnes proprement.
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Création d'un dictionnaire associant les anciens noms aux nouveaux noms.
    rename_mapping = dict(zip(ORIGINAL_COLUMNS, CLEAN_COLUMNS))

    # Renommage des colonnes du DataFrame.
    df = df.rename(columns=rename_mapping)

    # Retour du DataFrame avec les colonnes renommées.
    return df


# Définition d'une fonction pour forcer les types numériques.
def enforce_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    # Boucle sur chaque colonne numérique attendue.
    for column in FEATURE_COLUMNS + ["quality"]:
        # Conversion de la colonne en numérique, avec NaN si une valeur est invalide.
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Conversion de la qualité en type entier nullable pour accepter temporairement les NaN.
    df["quality"] = df["quality"].astype("Int64")

    # Retour du DataFrame typé.
    return df


# Définition d'une fonction pour créer la cible catégorielle.
def create_quality_label(df: pd.DataFrame) -> pd.DataFrame:
    # Création des conditions pour séparer les vins faibles, moyens et élevés.
    conditions = [
        df["quality"] <= 5,
        df["quality"] == 6,
        df["quality"] >= 7,
    ]

    # Définition des labels associés aux conditions.
    choices = [
        "low",
        "medium",
        "high",
    ]

    # Création de la colonne quality_label avec NumPy select.
    df["quality_label"] = np.select(conditions, choices, default=None)

    # Encodage numérique de la cible pour les modèles ML.
    df["quality_class"] = df["quality_label"].map(TARGET_MAPPING)

    # Conversion de quality_class en entier nullable.
    df["quality_class"] = df["quality_class"].astype("Int64")

    # Retour du DataFrame avec les cibles créées.
    return df


# Définition d'une fonction pour supprimer les lignes invalides.
def remove_invalid_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Suppression des lignes où la qualité originale est absente.
    df = df.dropna(subset=["quality"])

    # Suppression des lignes où la classe créée est absente.
    df = df.dropna(subset=["quality_label", "quality_class"])

    # Conversion définitive de quality en entier classique.
    df["quality"] = df["quality"].astype(int)

    # Conversion définitive de quality_class en entier classique.
    df["quality_class"] = df["quality_class"].astype(int)

    # Retour du DataFrame nettoyé.
    return df


# Définition d'une fonction pour supprimer les doublons.
def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    # Calcul du nombre de lignes avant suppression des doublons.
    before = len(df)

    # Suppression des lignes exactement dupliquées.
    df = df.drop_duplicates()

    # Calcul du nombre de lignes après suppression.
    after = len(df)

    # Calcul du nombre de doublons supprimés.
    removed = before - after

    # Retour du DataFrame et du nombre de doublons supprimés.
    return df, removed


# Définition d'une fonction pour séparer les features et la cible.
def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Sélection des colonnes features.
    X = df[FEATURE_COLUMNS].copy()

    # Sélection de la cible encodée.
    y = df["quality_class"].copy()

    # Retour des features et de la cible.
    return X, y


# Définition d'une fonction pour créer train, validation et test.
def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    # Séparation initiale entre train temporaire et test final.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Calcul de la taille relative de validation à l'intérieur du train temporaire.
    validation_relative_size = validation_size / (1.0 - test_size)

    # Séparation du train temporaire en train final et validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_relative_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    # Retour des trois ensembles.
    return X_train, X_val, X_test, y_train, y_val, y_test


# Définition d'une fonction pour calculer les bornes IQR sur le train uniquement.
def fit_iqr_bounds(X_train: pd.DataFrame, multiplier: float = 1.5) -> dict:
    # Création d'un dictionnaire vide pour stocker les bornes.
    bounds = {}

    # Boucle sur chaque feature numérique.
    for column in X_train.columns:
        # Calcul du premier quartile.
        q1 = X_train[column].quantile(0.25)

        # Calcul du troisième quartile.
        q3 = X_train[column].quantile(0.75)

        # Calcul de l'intervalle interquartile.
        iqr = q3 - q1

        # Calcul de la borne inférieure.
        lower = q1 - multiplier * iqr

        # Calcul de la borne supérieure.
        upper = q3 + multiplier * iqr

        # Sauvegarde des bornes pour cette colonne.
        bounds[column] = {
            "lower": float(lower),
            "upper": float(upper),
        }

    # Retour des bornes.
    return bounds


# Définition d'une fonction pour appliquer le clipping des outliers.
def apply_iqr_clipping(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    # Création d'une copie pour éviter de modifier l'objet original.
    X_clipped = X.copy()

    # Boucle sur chaque colonne et ses bornes.
    for column, limits in bounds.items():
        # Application du clipping entre la borne inférieure et la borne supérieure.
        X_clipped[column] = X_clipped[column].clip(
            lower=limits["lower"],
            upper=limits["upper"],
        )

    # Retour des données sans valeurs extrêmes excessives.
    return X_clipped


# Définition d'une fonction pour choisir le scaler.
def build_scaler(scaling: str):
    # Si l'utilisateur demande la standardisation.
    if scaling == "standard":
        # Retour d'un StandardScaler.
        return StandardScaler()

    # Si l'utilisateur demande la normalisation.
    if scaling == "minmax":
        # Retour d'un MinMaxScaler.
        return MinMaxScaler()

    # Si aucun scaling n'est demandé.
    if scaling == "none":
        # Retour de None.
        return None

    # Si la valeur est invalide, on bloque le pipeline.
    raise ValueError("scaling doit être 'standard', 'minmax' ou 'none'.")


# Définition d'une fonction pour transformer train, validation et test.
def fit_transform_preprocessing(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaling: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    # Création d'un imputer qui remplace les valeurs manquantes par la médiane du train.
    imputer = SimpleImputer(strategy="median")

    # Entraînement de l'imputer sur le train puis transformation du train.
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    # Transformation de la validation avec l'imputer appris sur le train.
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )

    # Transformation du test avec l'imputer appris sur le train.
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # Calcul des bornes d'outliers sur le train uniquement.
    iqr_bounds = fit_iqr_bounds(X_train_imputed)

    # Application du clipping sur le train.
    X_train_clipped = apply_iqr_clipping(X_train_imputed, iqr_bounds)

    # Application du clipping sur la validation.
    X_val_clipped = apply_iqr_clipping(X_val_imputed, iqr_bounds)

    # Application du clipping sur le test.
    X_test_clipped = apply_iqr_clipping(X_test_imputed, iqr_bounds)

    # Construction du scaler demandé.
    scaler = build_scaler(scaling)

    # Si un scaler est demandé.
    if scaler is not None:
        # Entraînement du scaler sur le train puis transformation du train.
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_clipped),
            columns=X_train.columns,
            index=X_train.index,
        )

        # Transformation de la validation avec le scaler appris sur le train.
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val_clipped),
            columns=X_val.columns,
            index=X_val.index,
        )

        # Transformation du test avec le scaler appris sur le train.
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_clipped),
            columns=X_test.columns,
            index=X_test.index,
        )
    else:
        # Si aucun scaler n'est demandé, le train reste simplement clippé.
        X_train_scaled = X_train_clipped

        # Si aucun scaler n'est demandé, la validation reste simplement clippée.
        X_val_scaled = X_val_clipped

        # Si aucun scaler n'est demandé, le test reste simplement clippé.
        X_test_scaled = X_test_clipped

    # Création d'un dictionnaire contenant les objets et paramètres de preprocessing.
    preprocessing_objects = {
        "imputer": imputer,
        "iqr_bounds": iqr_bounds,
        "scaler": scaler,
        "scaling": scaling,
        "feature_columns": FEATURE_COLUMNS,
        "target_mapping": TARGET_MAPPING,
    }

    # Retour des données transformées et des objets appris.
    return X_train_scaled, X_val_scaled, X_test_scaled, preprocessing_objects


# Définition d'une fonction pour sauvegarder les datasets transformés.
def save_processed_datasets(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    processed_dir: Path,
) -> None:
    # Création du dataset train avec features et cible.
    train_df = X_train.copy()

    # Ajout de la cible au train.
    train_df["quality_class"] = y_train.values

    # Création du dataset validation avec features et cible.
    val_df = X_val.copy()

    # Ajout de la cible à la validation.
    val_df["quality_class"] = y_val.values

    # Création du dataset test avec features et cible.
    test_df = X_test.copy()

    # Ajout de la cible au test.
    test_df["quality_class"] = y_test.values

    # Sauvegarde du train au format CSV.
    train_df.to_csv(processed_dir / "train.csv", index=False)

    # Sauvegarde de la validation au format CSV.
    val_df.to_csv(processed_dir / "validation.csv", index=False)

    # Sauvegarde du test au format CSV.
    test_df.to_csv(processed_dir / "test.csv", index=False)


# Définition d'une fonction pour créer un rapport de preprocessing.
def build_report(
    df_before_duplicates: pd.DataFrame,
    df_after_duplicates: pd.DataFrame,
    duplicates_removed: int,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    scaling: str,
) -> dict:
    # Création du rapport sous forme de dictionnaire.
    report = {
        "rows_before_duplicate_removal": int(len(df_before_duplicates)),
        "rows_after_duplicate_removal": int(len(df_after_duplicates)),
        "duplicates_removed": int(duplicates_removed),
        "missing_values_after_cleaning": df_after_duplicates.isna().sum().to_dict(),
        "target_distribution_full": df_after_duplicates["quality_label"].value_counts().to_dict(),
        "target_distribution_train": y_train.value_counts().to_dict(),
        "target_distribution_validation": y_val.value_counts().to_dict(),
        "target_distribution_test": y_test.value_counts().to_dict(),
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "test_shape": list(X_test.shape),
        "scaling": scaling,
    }

    # Retour du rapport.
    return report


# Définition de la fonction principale du pipeline de preprocessing.
def run_preprocessing(
    red_path: Path,
    white_path: Path,
    processed_dir: Path,
    artifacts_dir: Path,
    test_size: float,
    validation_size: float,
    random_state: int,
    scaling: str,
) -> None:
    # Création des dossiers nécessaires.
    ensure_directories(processed_dir, artifacts_dir)

    # Chargement du dataset des vins rouges.
    red_df = load_wine_file(red_path, "red")

    # Chargement du dataset des vins blancs.
    white_df = load_wine_file(white_path, "white")

    # Validation des colonnes du dataset rouge.
    validate_raw_columns(red_df)

    # Validation des colonnes du dataset blanc.
    validate_raw_columns(white_df)

    # Fusion des deux datasets en un seul DataFrame.
    df = pd.concat([red_df, white_df], axis=0, ignore_index=True)

    # Renommage propre des colonnes.
    df = rename_columns(df)

    # Conversion des colonnes en types numériques.
    df = enforce_numeric_types(df)

    # Création des labels de classification.
    df = create_quality_label(df)

    # Suppression des lignes avec cible invalide.
    df = remove_invalid_targets(df)

    # Sauvegarde du DataFrame avant suppression des doublons pour le rapport.
    df_before_duplicates = df.copy()

    # Suppression des doublons.
    df, duplicates_removed = remove_duplicates(df)

    # Séparation des features et de la cible.
    X, y = split_features_target(df)

    # Séparation des données en train, validation et test.
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(
        X=X,
        y=y,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    # Application de l'imputation, du clipping des outliers et du scaling.
    X_train_processed, X_val_processed, X_test_processed, preprocessing_objects = fit_transform_preprocessing(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        scaling=scaling,
    )

    # Sauvegarde des datasets transformés.
    save_processed_datasets(
        X_train=X_train_processed,
        X_val=X_val_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        processed_dir=processed_dir,
    )

    # Sauvegarde des objets de preprocessing.
    joblib.dump(preprocessing_objects, artifacts_dir / "preprocessing_objects.joblib")

    # Création du rapport de preprocessing.
    report = build_report(
        df_before_duplicates=df_before_duplicates,
        df_after_duplicates=df,
        duplicates_removed=duplicates_removed,
        X_train=X_train_processed,
        X_val=X_val_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaling=scaling,
    )

    # Sauvegarde du rapport au format JSON.
    with open(artifacts_dir / "preprocessing_report.json", "w", encoding="utf-8") as file:
        # Écriture du rapport avec indentation.
        json.dump(report, file, indent=4, ensure_ascii=False)

    # Affichage d'un message de succès.
    print("Preprocessing terminé avec succès.")

    # Affichage du dossier des données transformées.
    print(f"Données sauvegardées dans : {processed_dir}")

    # Affichage du dossier des artefacts.
    print(f"Artefacts sauvegardés dans : {artifacts_dir}")


# Définition d'une fonction pour lire les arguments en ligne de commande.
def parse_args() -> argparse.Namespace:
    # Création du parser d'arguments.
    parser = argparse.ArgumentParser(description="Pipeline de preprocessing Wine Quality.")

    # Argument pour le fichier des vins rouges.
    parser.add_argument("--red-path", type=Path, default=Path("data/raw/winequality-red.csv"))

    # Argument pour le fichier des vins blancs.
    parser.add_argument("--white-path", type=Path, default=Path("data/raw/winequality-white.csv"))

    # Argument pour le dossier de sortie des données transformées.
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))

    # Argument pour le dossier de sortie des artefacts.
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/preprocessing"))

    # Argument pour la taille du test set.
    parser.add_argument("--test-size", type=float, default=0.15)

    # Argument pour la taille du validation set.
    parser.add_argument("--validation-size", type=float, default=0.15)

    # Argument pour la reproductibilité.
    parser.add_argument("--random-state", type=int, default=42)

    # Argument pour choisir le type de scaling.
    parser.add_argument("--scaling", type=str, default="standard", choices=["standard", "minmax", "none"])

    # Retour des arguments parsés.
    return parser.parse_args()


# Point d'entrée du script.
if __name__ == "__main__":
    # Lecture des arguments.
    args = parse_args()

    # Lancement du preprocessing avec les arguments fournis.
    run_preprocessing(
        red_path=args.red_path,
        white_path=args.white_path,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        scaling=args.scaling,
    )
