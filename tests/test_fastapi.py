from __future__ import annotations

import pytest
import requests


@pytest.mark.integration
@pytest.mark.api
def test_health(require_api, api_base_url):
    response = requests.get(f"{api_base_url}/health", timeout=10)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["mlflow_status"] == "ok"
    assert body["model_status"] == "loaded"


@pytest.mark.integration
@pytest.mark.api
@pytest.mark.parametrize("route", ["/models", "/metrics", "/openapi.json"])
def test_read_routes(require_api, api_base_url, route):
    response = requests.get(f"{api_base_url}{route}", timeout=10)
    assert response.status_code == 200
    assert response.json() is not None


@pytest.mark.integration
@pytest.mark.api
def test_predict_success(require_api, api_base_url, valid_payload):
    response = requests.post(
        f"{api_base_url}/predict",
        json=valid_payload,
        timeout=20,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, dict)
    assert any(
        key in body
        for key in ("prediction", "predicted_class", "predicted_quality", "quality")
    )


@pytest.mark.integration
@pytest.mark.api
def test_predict_validation_error(require_api, api_base_url, valid_payload):
    invalid = valid_payload.copy()
    invalid.pop("alcohol")
    response = requests.post(
        f"{api_base_url}/predict",
        json=invalid,
        timeout=10,
    )
    assert response.status_code == 422
