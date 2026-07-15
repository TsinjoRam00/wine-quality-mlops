# Wine Quality — tests et observabilité

## Installation

```bash
cd /mnt/d/Projects/wine-quality-mlops
pip install -r requirements-dev.txt
```

## Tests

Rapides :

```bash
pytest -m unit -v
```

Infrastructure et API avec Docker démarré :

```bash
pytest -m integration -v
```

Tous les tests avec couverture :

```bash
pytest -v --cov=app --cov=src --cov-report=term-missing
```

## Intégration des logs et métriques

Au début de `app/main.py` :

```python
from app.logging_config import configure_logging
configure_logging()
```

Après `app = FastAPI(...)` :

```python
from app.observability import install_observability
install_observability(app)
```

Les métriques seront disponibles sur :

```text
http://localhost:8000/metrics/prometheus
```

Dans `/predict`, incrémente la métrique après une prédiction :

```python
from app.observability import PREDICTIONS

PREDICTIONS.labels(
    model_name="WineQualityClassifier",
    predicted_class=str(prediction),
).inc()
```

## Monitoring

```bash
docker compose -f docker-compose.monitoring.yml config
docker compose -f docker-compose.monitoring.yml up -d
```

Interfaces :

- Prometheus : http://localhost:9090
- Alertmanager : http://localhost:9093
- Grafana : http://localhost:3000

Dans Grafana, ajoute Prometheus avec l'URL interne :

```text
http://prometheus:9090
```

## Drift

Le script compare la référence `data/processed/train.csv` avec
`data/monitoring/production_features.csv`.

```bash
python -m src.monitoring.drift
```

Il retourne :

- code 0 : pas de drift ou moins de 100 observations ;
- code 2 : drift global détecté.

Le fichier `production_features.csv` doit être alimenté par les requêtes
réelles de `/predict`. En production, stocke plutôt chaque requête dans
PostgreSQL, puis exporte une fenêtre récente pour le contrôle.

## Feedback loop

Chaque prédiction doit retourner un `prediction_id`. Une route
`POST /feedback` reçoit ensuite la vraie qualité. Enregistre dans
PostgreSQL :

- features ;
- prédiction ;
- modèle/version ;
- date ;
- vraie qualité ;
- exactitude.

Avec au moins 50 feedbacks, calcule une accuracy/F1 glissante. Le drift
mesure les entrées ; le feedback mesure la performance réelle.

## Réentraînement automatique

Politique conseillée :

1. Au moins 100 prédictions et 30 % des variables en drift ; ou
2. Au moins 50 feedbacks et accuracy glissante < 0,55.
3. Jenkins entraîne un candidat.
4. Le candidat passe Pytest et les contrôles de données.
5. Comparaison avec `WineQualityClassifier@champion`.
6. Promotion seulement si le candidat est acceptable.
7. Déploiement, smoke test, rollback en cas d'échec.

Le patch Jenkins est dans `Jenkinsfile.monitoring.patch.txt`.


## PostgreSQL et feedback loop

Crée les tables :

```bash
docker exec -i wine_postgres psql -U <USER> -d <DATABASE>       < monitoring/sql/001_feedback_monitoring.sql
```

Dans `app/main.py` :

```python
from app.feedback_router import router as feedback_router
app.include_router(feedback_router)
```

Dans `/predict`, appelle `save_prediction(...)` et retourne aussi le
`prediction_id`. Le client envoie ensuite la vraie qualité à
`POST /feedback`.

Contrôle de performance :

```bash
python -m src.monitoring.performance
```

Décision de réentraînement :

```bash
python -m src.monitoring.retraining_policy
```

Le réentraînement produit un candidat. Ne change l'alias
`WineQualityClassifier@champion` qu'après tests, comparaison des
métriques et smoke test.
