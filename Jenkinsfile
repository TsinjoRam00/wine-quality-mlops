pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    parameters {
        booleanParam(
            name: 'RUN_TRAINING',
            defaultValue: true,
            description: 'Exécuter le preprocessing et entraîner les modèles'
        )

        booleanParam(
            name: 'PUSH_DOCKER_IMAGE',
            defaultValue: true,
            description: 'Envoyer l’image vers Docker Hub'
        )

        booleanParam(
            name: 'DEPLOY_API',
            defaultValue: true,
            description: 'Déployer automatiquement l’API avec Docker Compose'
        )
    }

    environment {
        DOCKER_IMAGE = '00znz/wine-quality-api'

        PYTHONPATH = '.'

        /*
         * Jenkins, MLflow et MinIO communiquent
         * à travers le réseau Docker.
         */
        MLFLOW_TRACKING_URI = 'http://mlflow:5000'
        MLFLOW_REGISTRY_URI = 'http://mlflow:5000'
        MLFLOW_S3_ENDPOINT_URL = 'http://minio:9000'
        MINIO_ENDPOINT = 'http://minio:9000'
        AWS_ACCESS_KEY_ID = 'minioadmin'
        AWS_SECRET_ACCESS_KEY = 'minioadmin123'
        AWS_DEFAULT_REGION = 'us-east-1'
        MLFLOW_ARTIFACT_BUCKET = 'mlflow-artifacts'

        POSTGRES_HOST = 'postgres'
        POSTGRES_PORT = '5432'
        POSTGRES_DB = 'mlflow_db'
        POSTGRES_USER = 'mlops_user'
        POSTGRES_PASSWORD = 'mlops_password'
        AWS_EC2_METADATA_DISABLED = 'true'
    }

    stages {
        stage('Checkout GitHub') {
            steps {
                checkout scm
            }
        }

        stage('System Check') {
            steps {
                sh '''
                    echo "===== Git ====="
                    git --version

                    echo "===== Python ====="
                    python3 --version

                    echo "===== Docker ====="
                    docker --version

                    echo "===== Docker Compose ====="
                    docker compose version

                    echo "===== Ansible ====="
                    ansible --version
                '''
            }
        }

        stage('Python Environment') {
            steps {
                sh '''
                    rm -rf .venv

                    python3 -m venv .venv
                    . .venv/bin/activate

                    python -m pip install --upgrade pip setuptools wheel
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Check MLOps Services') {
            steps {
                sh '''
                    echo "Vérification de MLflow..."
                    curl -fsS http://mlflow:5000 > /dev/null

                    echo "Vérification de MinIO..."
                    curl -fsS http://minio:9000/minio/health/live > /dev/null

                    echo "MLflow et MinIO répondent correctement."
                '''
            }
        }

        stage('Download Dataset') {
            when {
                expression {
                    return params.RUN_TRAINING
                }
            }

            steps {
                sh '''
                    mkdir -p data/raw

                    if [ ! -f data/raw/winequality-red.csv ]; then
                        curl -L \
                          -o data/raw/winequality-red.csv \
                          https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
                    fi

                    if [ ! -f data/raw/winequality-white.csv ]; then
                        curl -L \
                          -o data/raw/winequality-white.csv \
                          https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
                    fi

                    ls -lh data/raw
                '''
            }
        }

        stage('Preprocessing') {
            when {
                expression {
                    return params.RUN_TRAINING
                }
            }

            steps {
                sh '''
                    . .venv/bin/activate
                    python src/features/preprocessing.py
                '''
            }
        }

        stage('Tests') {
            steps {
                sh '''
                    mkdir -p artifacts/reports

                    . .venv/bin/activate

                    pytest -v \
                      --junitxml=artifacts/reports/pytest-report.xml
                '''
            }
        }

        stage('Training and MLflow Registry') {
            when {
                expression {
                    return params.RUN_TRAINING
                }
            }

            steps {
                sh '''
                    . .venv/bin/activate

                    python -m src.training.train \
                      --grid-size fast \
                      --cv 3 \
                      --n-jobs 1 \
                      --register-model
                '''
            }
        }

        stage('Check Docker Build Files') {
            steps {
                sh '''
                    echo "Vérification des fichiers nécessaires à l’image..."

                    if [ ! -f artifacts/preprocessing/preprocessing_objects.joblib ]; then
                        echo "ERREUR : preprocessing_objects.joblib est absent."
                        echo "Relance le pipeline avec RUN_TRAINING coché."
                        exit 1
                    fi

                    if [ ! -d data/processed ]; then
                        echo "ERREUR : le dossier data/processed est absent."
                        echo "Relance le pipeline avec RUN_TRAINING coché."
                        exit 1
                    fi

                    echo "Les fichiers nécessaires sont présents."
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    echo "Construction de l’image ${DOCKER_IMAGE}:${BUILD_NUMBER}..."

                    docker build \
                      -t ${DOCKER_IMAGE}:${BUILD_NUMBER} \
                      -t ${DOCKER_IMAGE}:latest \
                      -f docker/Dockerfile.api \
                      .

                    echo "Image Docker construite :"
                    docker image inspect \
                      ${DOCKER_IMAGE}:${BUILD_NUMBER} \
                      --format='{{.RepoTags}}'
                '''
            }
        }

        stage('Push Docker Hub') {
            when {
                expression {
                    return params.PUSH_DOCKER_IMAGE
                }
            }

            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'dockerhub-credentials',
                        usernameVariable: 'DOCKERHUB_USERNAME',
                        passwordVariable: 'DOCKERHUB_TOKEN'
                    )
                ]) {
                    sh '''
                        set -eu

                        trap 'docker logout || true' EXIT

                        echo "$DOCKERHUB_TOKEN" |
                          docker login \
                            --username "$DOCKERHUB_USERNAME" \
                            --password-stdin

                        echo "Envoi du tag ${BUILD_NUMBER}..."
                        docker push ${DOCKER_IMAGE}:${BUILD_NUMBER}

                        echo "Envoi du tag latest..."
                        docker push ${DOCKER_IMAGE}:latest
                    '''
                }
            }
        }

        stage('Deploy API with Docker Compose') {
            when {
                expression {
                    return params.PUSH_DOCKER_IMAGE && params.DEPLOY_API
                }
            }

            steps {
                sh '''
                    set -eu

                    echo "===== Vérification du fichier Compose ====="

                    test -f docker-compose.deploy.yml

                    docker compose \
                      -f docker-compose.deploy.yml \
                      config > /dev/null

                    echo "===== Vérification du réseau Docker ====="

                    docker network inspect \
                      wine_mlops_network > /dev/null

                    echo "===== Téléchargement de l’image ====="

                    docker pull \
                      ${DOCKER_IMAGE}:${BUILD_NUMBER}

                    echo "===== Suppression de l’ancienne API ====="

                    docker rm -f wine_api || true

                    echo "===== Déploiement de la nouvelle API ====="

                    IMAGE_TAG=${BUILD_NUMBER} \
                      docker compose \
                        -f docker-compose.deploy.yml \
                        up -d \
                        --force-recreate \
                        api

                    echo "===== État des services ====="

                    IMAGE_TAG=${BUILD_NUMBER} \
                      docker compose \
                        -f docker-compose.deploy.yml \
                        ps
                '''
            }
        }

        stage('Smoke Test API') {
            when {
                expression {
                    return params.PUSH_DOCKER_IMAGE && params.DEPLOY_API
                }
            }

            steps {
                sh '''
                    echo "Attente du démarrage de FastAPI..."

                    for attempt in $(seq 1 24); do
                        HEALTH_STATUS=$(
                            docker inspect \
                              --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' \
                              wine_api 2>/dev/null || echo "absent"
                        )

                        echo "Tentative ${attempt}/24 — état : ${HEALTH_STATUS}"

                        if docker exec wine_api \
                            curl -fsS http://localhost:8000/health
                        then
                            echo ""
                            echo "API déployée et fonctionnelle."
                            exit 0
                        fi

                        if [ "$HEALTH_STATUS" = "unhealthy" ]; then
                            echo "Le conteneur est devenu unhealthy."
                            docker logs --tail=150 wine_api
                            exit 1
                        fi

                        sleep 5
                    done

                    echo "L’API ne répond pas après 120 secondes."

                    echo "===== État du conteneur ====="
                    docker inspect \
                      wine_api \
                      --format='{{json .State.Health}}' || true

                    echo "===== Logs de l’API ====="
                    docker logs --tail=150 wine_api || true

                    exit 1
                '''
            }
        }
    }

    post {
        always {
            junit(
                testResults: 'artifacts/reports/pytest-report.xml',
                allowEmptyResults: true
            )

            archiveArtifacts(
                artifacts: 'artifacts/reports/**/*,artifacts/figures/**/*',
                allowEmptyArchive: true
            )
        }

        success {
            echo 'Pipeline Wine Quality MLOps terminé avec succès.'
        }

        failure {
            echo 'Pipeline échoué. Consulte la sortie console Jenkins.'
        }
    }
}
