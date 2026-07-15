from __future__ import annotations

import json
import os
import subprocess
import uuid
from urllib.parse import urlparse

import boto3
import mlflow
import psycopg2
import pytest
from botocore.config import Config
from mlflow import MlflowClient


def command(*args: str):
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.mark.integration
@pytest.mark.mlflow
def test_mlflow_experiment_and_champion(require_mlflow, mlflow_tracking_uri):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("wine-quality-classification")
    assert experiment is not None

    champion = client.get_model_version_by_alias(
        "WineQualityClassifier",
        "champion",
    )
    assert champion is not None
    assert int(champion.version) >= 1


def postgres_kwargs() -> dict:
    uri = os.getenv("POSTGRES_TEST_URI") or os.getenv("DATABASE_URL")
    if uri and uri.startswith(("postgresql://", "postgres://")):
        parsed = urlparse(uri)
        return {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "dbname": parsed.path.lstrip("/"),
            "user": parsed.username,
            "password": parsed.password,
            "connect_timeout": 5,
        }
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", "mlflow"),
        "user": os.getenv("POSTGRES_USER", "mlflow"),
        "password": os.getenv("POSTGRES_PASSWORD", "mlflowpassword"),
        "connect_timeout": 5,
    }


@pytest.mark.integration
@pytest.mark.postgres
def test_postgres_read_write():
    try:
        connection = psycopg2.connect(**postgres_kwargs())
    except psycopg2.Error as error:
        pytest.skip(f"PostgreSQL indisponible: {error}")

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
            cursor.execute("CREATE TEMP TABLE pytest_probe (value text)")
            cursor.execute("INSERT INTO pytest_probe VALUES ('ok')")
            cursor.execute("SELECT value FROM pytest_probe")
            assert cursor.fetchone()[0] == "ok"
    finally:
        connection.rollback()
        connection.close()


@pytest.mark.integration
@pytest.mark.minio
def test_minio_bucket_and_roundtrip():
    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(signature_version="s3v4"),
    )
    bucket = os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")
    names = {item["Name"] for item in client.list_buckets()["Buckets"]}
    assert bucket in names

    key = f"pytest/{uuid.uuid4().hex}.txt"
    try:
        client.put_object(Bucket=bucket, Key=key, Body=b"ok")
        response = client.get_object(Bucket=bucket, Key=key)
        assert response["Body"].read() == b"ok"
    finally:
        client.delete_object(Bucket=bucket, Key=key)


@pytest.mark.integration
@pytest.mark.docker
def test_docker_compose_and_health(require_docker, project_root):
    config = command(
        "docker",
        "compose",
        "-f",
        str(project_root / "docker-compose.yml"),
        "config",
        "--quiet",
    )
    assert config.returncode == 0, config.stderr

    for container in (
        "wine_api",
        "wine_mlflow",
        "wine_minio",
        "wine_postgres",
        "wine_jenkins",
    ):
        inspect = command(
            "docker",
            "inspect",
            container,
            "--format",
            "{{json .State}}",
        )
        assert inspect.returncode == 0, inspect.stderr
        state = json.loads(inspect.stdout)
        assert state["Running"] is True
        if state.get("Health"):
            assert state["Health"]["Status"] == "healthy"


@pytest.mark.integration
@pytest.mark.docker
def test_deployed_api_image_is_versioned(require_docker):
    """Valide l'image uniquement lorsqu'un déploiement est attendu."""
    expected_prefix = os.getenv("EXPECTED_API_IMAGE_PREFIX")

    if not expected_prefix:
        pytest.skip(
            "Validation de l'image réservée à l'étape post-déploiement."
        )

    inspect = command(
        "docker",
        "inspect",
        "wine_api",
        "--format",
        "{{.Config.Image}}",
    )

    assert inspect.returncode == 0, inspect.stderr

    image = inspect.stdout.strip()

    assert image.startswith(expected_prefix)
    assert not image.endswith(":deploy-test")
