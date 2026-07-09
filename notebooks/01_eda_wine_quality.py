# %% [markdown]
# # EDA - Wine Quality Dataset
# Ce notebook analyse le dataset Wine Quality de l'UCI.
# Il étudie les vins rouges et blancs, les statistiques, les distributions, les corrélations, les valeurs manquantes, les doublons et les outliers.

# %%
# Importation de Pandas pour manipuler les données tabulaires.
import pandas as pd

# Importation de NumPy pour les calculs numériques.
import numpy as np

# Importation de Matplotlib pour créer les graphiques.
import matplotlib.pyplot as plt

# Définition du chemin vers le fichier des vins rouges.
red_path = "../data/raw/winequality-red.csv"

# Définition du chemin vers le fichier des vins blancs.
white_path = "../data/raw/winequality-white.csv"

# Lecture du dataset des vins rouges avec le séparateur officiel.
red_df = pd.read_csv(red_path, sep=";")

# Lecture du dataset des vins blancs avec le séparateur officiel.
white_df = pd.read_csv(white_path, sep=";")

# Ajout d'une colonne pour identifier les vins rouges.
red_df["wine_type"] = "red"

# Ajout d'une colonne pour identifier les vins blancs.
white_df["wine_type"] = "white"

# Fusion des deux datasets dans un seul DataFrame.
df = pd.concat([red_df, white_df], axis=0, ignore_index=True)

# Affichage des dimensions du dataset combiné.
print("Dimensions du dataset combiné :", df.shape)

# Affichage des cinq premières lignes.
df.head()

# %% [markdown]
# ## 1. Renommage des colonnes
# On remplace les espaces par des underscores pour avoir des noms plus propres en Python.

# %%
# Création d'un dictionnaire pour renommer les colonnes.
rename_mapping = {
    "fixed acidity": "fixed_acidity",
    "volatile acidity": "volatile_acidity",
    "citric acid": "citric_acid",
    "residual sugar": "residual_sugar",
    "free sulfur dioxide": "free_sulfur_dioxide",
    "total sulfur dioxide": "total_sulfur_dioxide",
    "pH": "ph",
}

# Application du renommage des colonnes.
df = df.rename(columns=rename_mapping)

# Affichage des colonnes après renommage.
df.columns

# %% [markdown]
# ## 2. Informations générales
# On vérifie les types de colonnes et le nombre de valeurs non nulles.

# %%
# Affichage des informations générales sur le dataset.
df.info()

# %%
# Affichage des types de variables.
df.dtypes

# %% [markdown]
# ## 3. Description statistique
# On calcule les statistiques principales : moyenne, écart-type, min, quartiles et max.

# %%
# Affichage des statistiques descriptives des variables numériques.
df.describe().T

# %% [markdown]
# ## 4. Valeurs manquantes
# L'UCI indique qu'il n'y a pas de valeurs manquantes, mais on le vérifie quand même.

# %%
# Calcul du nombre de valeurs manquantes par colonne.
missing_values = df.isna().sum()

# Affichage des valeurs manquantes.
missing_values

# %%
# Calcul du pourcentage de valeurs manquantes par colonne.
missing_percent = df.isna().mean() * 100

# Affichage du pourcentage de valeurs manquantes.
missing_percent

# %% [markdown]
# ## 5. Doublons
# On vérifie si certaines lignes sont exactement identiques.

# %%
# Calcul du nombre de doublons exacts.
duplicates_count = df.duplicated().sum()

# Affichage du nombre de doublons.
print("Nombre de doublons :", duplicates_count)

# %%
# Affichage des lignes dupliquées si elles existent.
df[df.duplicated()].head()

# %% [markdown]
# ## 6. Distribution de la variable cible quality
# La variable quality est la note sensorielle du vin.

# %%
# Calcul du nombre d'observations par note de qualité.
quality_counts = df["quality"].value_counts().sort_index()

# Affichage de la distribution de quality.
quality_counts

# %%
# Création d'un graphique de distribution de quality.
quality_counts.plot(kind="bar")

# Ajout du titre du graphique.
plt.title("Distribution de la qualité du vin")

# Ajout du label de l'axe X.
plt.xlabel("Quality")

# Ajout du label de l'axe Y.
plt.ylabel("Nombre d'observations")

# Affichage du graphique.
plt.show()

# %% [markdown]
# ## 7. Création de quality_label
# On transforme la note en classes : low, medium, high.

# %%
# Définition des conditions pour les classes.
conditions = [
    df["quality"] <= 5,
    df["quality"] == 6,
    df["quality"] >= 7,
]

# Définition des labels associés.
choices = [
    "low",
    "medium",
    "high",
]

# Création de la colonne quality_label.
df["quality_label"] = np.select(conditions, choices, default=None)

# Affichage des premières lignes avec la nouvelle cible.
df[["quality", "quality_label"]].head()

# %%
# Distribution de la nouvelle cible.
df["quality_label"].value_counts()

# %%
# Graphique de la distribution de quality_label.
df["quality_label"].value_counts().plot(kind="bar")

# Ajout du titre.
plt.title("Distribution de quality_label")

# Ajout du label X.
plt.xlabel("Classe")

# Ajout du label Y.
plt.ylabel("Nombre d'observations")

# Affichage du graphique.
plt.show()

# %% [markdown]
# ## 8. Distribution par type de vin
# On vérifie le nombre de vins rouges et blancs.

# %%
# Calcul du nombre de vins par type.
wine_type_counts = df["wine_type"].value_counts()

# Affichage des résultats.
wine_type_counts

# %%
# Graphique de la distribution des types de vin.
wine_type_counts.plot(kind="bar")

# Ajout du titre.
plt.title("Distribution des types de vin")

# Ajout du label X.
plt.xlabel("Type de vin")

# Ajout du label Y.
plt.ylabel("Nombre d'observations")

# Affichage du graphique.
plt.show()

# %% [markdown]
# ## 9. Distribution des variables numériques
# On visualise chaque variable numérique avec un histogramme.

# %%
# Définition des colonnes numériques.
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Suppression de quality de cette liste pour analyser les features séparément.
feature_columns = [column for column in numeric_columns if column != "quality"]

# Affichage de la liste des features numériques.
feature_columns

# %%
# Boucle sur chaque feature numérique.
for column in feature_columns:
    # Création d'une nouvelle figure pour chaque variable.
    plt.figure()

    # Création de l'histogramme.
    df[column].hist(bins=30)

    # Ajout du titre.
    plt.title(f"Distribution de {column}")

    # Ajout du label X.
    plt.xlabel(column)

    # Ajout du label Y.
    plt.ylabel("Fréquence")

    # Affichage du graphique.
    plt.show()

# %% [markdown]
# ## 10. Corrélations
# On calcule les corrélations entre variables numériques.

# %%
# Calcul de la matrice de corrélation.
correlation_matrix = df[numeric_columns].corr()

# Affichage de la matrice de corrélation.
correlation_matrix

# %%
# Création d'une figure pour la heatmap.
plt.figure(figsize=(12, 8))

# Affichage de la matrice sous forme d'image.
plt.imshow(correlation_matrix, cmap="viridis", aspect="auto")

# Ajout d'une barre de couleur.
plt.colorbar()

# Ajout des labels de l'axe X.
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)

# Ajout des labels de l'axe Y.
plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)

# Ajout du titre.
plt.title("Matrice de corrélation")

# Ajustement de la mise en page.
plt.tight_layout()

# Affichage du graphique.
plt.show()

# %%
# Corrélation de chaque variable avec quality.
quality_correlations = correlation_matrix["quality"].sort_values(ascending=False)

# Affichage des corrélations avec quality.
quality_correlations

# %% [markdown]
# ## 11. Analyse des outliers
# On utilise la méthode IQR pour compter les valeurs extrêmes.

# %%
# Définition d'une fonction pour compter les outliers d'une colonne.
def count_outliers_iqr(series):
    # Calcul du premier quartile.
    q1 = series.quantile(0.25)

    # Calcul du troisième quartile.
    q3 = series.quantile(0.75)

    # Calcul de l'intervalle interquartile.
    iqr = q3 - q1

    # Calcul de la borne inférieure.
    lower = q1 - 1.5 * iqr

    # Calcul de la borne supérieure.
    upper = q3 + 1.5 * iqr

    # Comptage des valeurs inférieures à la borne inférieure ou supérieures à la borne supérieure.
    count = ((series < lower) | (series > upper)).sum()

    # Retour du nombre d'outliers.
    return count

# %%
# Création d'un dictionnaire avec le nombre d'outliers par variable.
outlier_counts = {column: count_outliers_iqr(df[column]) for column in feature_columns}

# Conversion du dictionnaire en Series Pandas.
outlier_counts = pd.Series(outlier_counts).sort_values(ascending=False)

# Affichage du nombre d'outliers.
outlier_counts

# %%
# Graphique du nombre d'outliers par variable.
outlier_counts.plot(kind="bar")

# Ajout du titre.
plt.title("Nombre d'outliers par variable")

# Ajout du label X.
plt.xlabel("Variable")

# Ajout du label Y.
plt.ylabel("Nombre d'outliers")

# Ajustement de la mise en page.
plt.tight_layout()

# Affichage du graphique.
plt.show()

# %% [markdown]
# ## 12. Boxplots des variables numériques
# Les boxplots permettent de visualiser les valeurs extrêmes.

# %%
# Boucle sur chaque feature numérique.
for column in feature_columns:
    # Création d'une nouvelle figure.
    plt.figure()

    # Création du boxplot.
    plt.boxplot(df[column].dropna(), vert=False)

    # Ajout du titre.
    plt.title(f"Boxplot de {column}")

    # Ajout du label X.
    plt.xlabel(column)

    # Affichage du graphique.
    plt.show()

# %% [markdown]
# ## 13. Relation entre alcohol et quality
# L'alcool est souvent une variable importante dans ce dataset.

# %%
# Création d'une figure.
plt.figure()

# Création d'un scatter plot entre alcohol et quality.
plt.scatter(df["alcohol"], df["quality"], alpha=0.3)

# Ajout du titre.
plt.title("Relation entre alcohol et quality")

# Ajout du label X.
plt.xlabel("Alcohol")

# Ajout du label Y.
plt.ylabel("Quality")

# Affichage du graphique.
plt.show()

# %% [markdown]
# ## 14. Relation entre volatile_acidity et quality
# L'acidité volatile peut être négativement liée à la qualité.

# %%
# Création d'une figure.
plt.figure()

# Création du scatter plot entre volatile_acidity et quality.
plt.scatter(df["volatile_acidity"], df["quality"], alpha=0.3)

# Ajout du titre.
plt.title("Relation entre volatile_acidity et quality")

# Ajout du label X.
plt.xlabel("Volatile acidity")

# Ajout du label Y.
plt.ylabel("Quality")

# Affichage du graphique.
plt.show()

# %% [markdown]
# ## 15. Sauvegarde du rapport EDA
# On sauvegarde quelques résultats utiles dans un fichier JSON.

# %%
# Importation de json pour écrire un rapport.
import json

# Création du dossier de rapport.
report_dir = "../artifacts/reports"

# Création du dossier s'il n'existe pas.
import os

# Création effective du dossier.
os.makedirs(report_dir, exist_ok=True)

# Création du rapport EDA.
eda_report = {
    "shape": df.shape,
    "missing_values": df.isna().sum().to_dict(),
    "duplicates_count": int(duplicates_count),
    "quality_distribution": df["quality"].value_counts().sort_index().to_dict(),
    "quality_label_distribution": df["quality_label"].value_counts().to_dict(),
    "wine_type_distribution": df["wine_type"].value_counts().to_dict(),
    "outlier_counts": outlier_counts.to_dict(),
    "quality_correlations": quality_correlations.to_dict(),
}

# Ouverture du fichier JSON en écriture.
with open(f"{report_dir}/eda_report.json", "w", encoding="utf-8") as file:
    # Écriture du rapport dans le fichier JSON.
    json.dump(eda_report, file, indent=4, ensure_ascii=False)

# Affichage d'un message de succès.
print("Rapport EDA sauvegardé dans artifacts/reports/eda_report.json")
