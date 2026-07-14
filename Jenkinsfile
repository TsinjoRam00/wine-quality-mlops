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
            defaultValue: false,
            description: 'Envoyer l’image vers Docker Hub'
        )
    }

    environment {
        /*
         *
         */
        DOCKER_IMAGE = '00znz/wine-quality-api'

        PYTHONPATH = '.'

        /*
         * Jenkins, MLflow et MinIO sont dans le même réseau Docker.
         * Jenkins utilise donc les noms de services Docker.
         */
        MLFLOW_TRACKING_URI = 'http://mlflow:5000'
        MLFLOW_REGISTRY_URI = 'http://mlflow:5000'
        MLFLOW_S3_ENDPOINT_URL = 'http://minio:9000'

        AWS_ACCESS_KEY_ID = 'minioadmin'
        AWS_SECRET_ACCESS_KEY = 'minioadmin123'
        AWS_DEFAULT_REGION = 'us-east-1'
        AWS_EC2_METADATA_DISABLED = 'true'
    }

    stages {
        stage('Checkout GitHub') {
            steps {
                /*
                 * Jenkins utilise le dépôt configuré dans le job.
                 */
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

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build \
                      -t ${DOCKER_IMAGE}:${BUILD_NUMBER} \
                      -t ${DOCKER_IMAGE}:latest \
                      -f docker/Dockerfile.api \
                      .
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
                        echo "$DOCKERHUB_TOKEN" |
                          docker login \
                            --username "$DOCKERHUB_USERNAME" \
                            --password-stdin

                        docker push ${DOCKER_IMAGE}:${BUILD_NUMBER}
                        docker push ${DOCKER_IMAGE}:latest

                        docker logout
                    '''
                }
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
