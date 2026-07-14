pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    parameters {
        string(
            name: 'GIT_BRANCH',
            defaultValue: 'main',
            description: 'Branche GitHub à construire'
        )

        booleanParam(
            name: 'RUN_TRAINING',
            defaultValue: true,
            description: 'Lancer le preprocessing et le training ML'
        )

        booleanParam(
            name: 'PUSH_DOCKER_IMAGE',
            defaultValue: true,
            description: 'Envoyer l’image API vers Docker Hub'
        )

        booleanParam(
            name: 'DEPLOY_LOCAL_COMPOSE',
            defaultValue: true,
            description: 'Déployer avec docker compose sur la machine Jenkins'
        )
    }

    environment {
        GITHUB_REPO = 'git@github.com:TsinjoRam00/wine-quality-mlops.git'

        DOCKER_IMAGE = '00znz/wine-quality-api'

        PYTHONPATH = '.'

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
                git branch: "${params.GIT_BRANCH}",
                    credentialsId: 'github-ssh-key',
                    url: "${env.GITHUB_REPO}"
            }
        }

        stage('System Check') {
            steps {
                sh '''
                    echo "=== Docker ==="
                    docker --version

                    echo "=== Docker Compose ==="
                    docker compose version

                    echo "=== Python ==="
                    python3 --version

                    echo "=== Ansible ==="
                    ansible --version
                '''
            }
        }

        stage('Install Python Dependencies') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate

                    python -m pip install --upgrade pip setuptools wheel
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Start MLOps Services') {
            steps {
                sh '''
                    docker compose up -d postgres minio create-bucket mlflow
                    sleep 30
                    docker compose ps
                '''
            }
        }

        stage('Download Dataset') {
            when {
                expression { return params.RUN_TRAINING }
            }
            steps {
                sh '''
                    mkdir -p data/raw

                    if [ ! -f data/raw/winequality-red.csv ]; then
                        curl -L -o data/raw/winequality-red.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
                    fi

                    if [ ! -f data/raw/winequality-white.csv ]; then
                        curl -L -o data/raw/winequality-white.csv https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
                    fi

                    ls -lh data/raw
                '''
            }
        }

        stage('Preprocessing') {
            when {
                expression { return params.RUN_TRAINING }
            }
            steps {
                sh '''
                    . .venv/bin/activate
                    python src/features/preprocessing.py
                '''
            }
        }

        stage('Run Tests') {
            steps {
                sh '''
                    mkdir -p artifacts/reports

                    . .venv/bin/activate
                    pytest -v --junitxml=artifacts/reports/pytest-report.xml
                '''
            }
        }

        stage('Train Model and Register MLflow') {
            when {
                expression { return params.RUN_TRAINING }
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
                        -f docker/Dockerfile.api .
                '''
            }
        }

        stage('Push Docker Hub') {
            when {
                expression { return params.PUSH_DOCKER_IMAGE }
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
                        echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

                        docker push ${DOCKER_IMAGE}:${BUILD_NUMBER}
                        docker push ${DOCKER_IMAGE}:latest

                        docker logout
                    '''
                }
            }
        }

        stage('Deploy with Docker Compose') {
            when {
                expression { return params.DEPLOY_LOCAL_COMPOSE }
            }
            steps {
                sh '''
                    docker compose up -d --build
                    docker compose ps
                '''
            }
        }

        stage('Smoke Test API') {
            when {
                expression { return params.DEPLOY_LOCAL_COMPOSE }
            }
            steps {
                sh '''
                    sleep 20

                    curl -f http://api:8000/health || curl -f http://localhost:8000/health
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'artifacts/reports/**/*, artifacts/figures/**/*', allowEmptyArchive: true
            junit allowEmptyResults: true, testResults: 'artifacts/reports/pytest-report.xml'
        }

        success {
            echo 'Pipeline terminé avec succès.'
        }

        failure {
            echo 'Pipeline échoué. Vérifie la console Jenkins.'
        }
    }
}
