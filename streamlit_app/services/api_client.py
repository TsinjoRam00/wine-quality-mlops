from __future__ import annotations

import os
from typing import Any

import requests


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")


def get_health() -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def predict_wine(payload: dict[str, float]) -> dict[str, Any]:
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def send_feedback(
    prediction_id: str,
    actual_class: int,
    comment: str | None = None,
) -> dict[str, Any]:
    payload = {
        "prediction_id": prediction_id,
        "actual_class": actual_class,
        "comment": comment or None,
    }

    response = requests.post(
        f"{API_BASE_URL}/feedback",
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    return response.json()
