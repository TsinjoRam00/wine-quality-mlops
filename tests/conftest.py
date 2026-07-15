from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def processed_dir(project_root: Path) -> Path:
    return project_root / "data" / "processed"


@pytest.fixture(scope="session")
def api_base_url() -> str:
    return os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


@pytest.fixture(scope="session")
def mlflow_tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000").rstrip("/")


@pytest.fixture
def valid_payload() -> dict[str, float]:
    return {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.70,
        "citric_acid": 0.00,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }


def service_up(url: str) -> bool:
    try:
        return requests.get(url, timeout=4).status_code < 500
    except requests.RequestException:
        return False


@pytest.fixture(scope="session")
def require_api(api_base_url: str) -> Iterator[None]:
    if not service_up(f"{api_base_url}/health"):
        pytest.skip(f"FastAPI indisponible sur {api_base_url}")
    yield


@pytest.fixture(scope="session")
def require_mlflow(mlflow_tracking_uri: str) -> Iterator[None]:
    if not service_up(mlflow_tracking_uri):
        pytest.skip(f"MLflow indisponible sur {mlflow_tracking_uri}")
    yield


@pytest.fixture(scope="session")
def require_docker() -> Iterator[None]:
    if shutil.which("docker") is None:
        pytest.skip("Docker absent")
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if result.returncode != 0:
        pytest.skip(result.stderr.strip())
    yield


def load_split(processed_dir: Path, name: str) -> pd.DataFrame:
    path = processed_dir / f"{name}.csv"
    assert path.is_file(), f"Fichier absent: {path}"
    return pd.read_csv(path)


def target_column(frame: pd.DataFrame) -> str:
    for name in (
        "quality_class",
        "quality_category",
        "quality_label",
        "target",
        "label",
        "quality",
    ):
        if name in frame.columns:
            return name
    raise AssertionError(f"Colonne cible introuvable: {list(frame.columns)}")
