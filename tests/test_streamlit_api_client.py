from __future__ import annotations

from typing import Any

from streamlit_app.services import api_client


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.status_checked = False

    def raise_for_status(self) -> None:
        self.status_checked = True

    def json(self) -> dict[str, Any]:
        return self.payload


def test_get_health(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, timeout: int) -> FakeResponse:
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "status": "ok",
                "model_status": "loaded",
            }
        )

    monkeypatch.setattr(api_client.requests, "get", fake_get)

    result = api_client.get_health()

    assert captured["url"].endswith("/health")
    assert captured["timeout"] == 10
    assert result["status"] == "ok"
    assert result["model_status"] == "loaded"


def test_predict_wine(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str,
        json: dict[str, float],
        timeout: int,
    ) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout

        return FakeResponse(
            {
                "predicted_label": "medium",
                "prediction_id": "prediction-test-123",
            }
        )

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    payload = {
        "fixed_acidity": 7.4,
        "alcohol": 9.4,
    }

    result = api_client.predict_wine(payload)

    assert captured["url"].endswith("/predict")
    assert captured["json"] == payload
    assert captured["timeout"] == 30
    assert result["predicted_label"] == "medium"
    assert result["prediction_id"] == "prediction-test-123"


def test_send_feedback(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str,
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout

        return FakeResponse(
            {
                "status": "accepted",
                "prediction_id": json["prediction_id"],
            }
        )

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    result = api_client.send_feedback(
        prediction_id="prediction-test-123",
        actual_class=1,
        comment="Test Jenkins",
    )

    assert captured["url"].endswith("/feedback")
    assert captured["timeout"] == 15
    assert captured["json"] == {
        "prediction_id": "prediction-test-123",
        "actual_class": 1,
        "comment": "Test Jenkins",
    }
    assert result["status"] == "accepted"
