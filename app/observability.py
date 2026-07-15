from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.database import connect

LOGGER = logging.getLogger("wine_api")

HTTP_REQUESTS = Counter(
    "wine_api_http_requests_total",
    "Requêtes HTTP",
    ["method", "path", "status"],
)
HTTP_ERRORS = Counter(
    "wine_api_http_errors_total",
    "Erreurs HTTP 5xx",
    ["method", "path"],
)
HTTP_LATENCY = Histogram(
    "wine_api_http_request_duration_seconds",
    "Latence HTTP",
    ["method", "path"],
)
PREDICTIONS = Counter(
    "wine_model_predictions_total",
    "Prédictions",
    ["model_name", "predicted_class"],
)
MODEL_LOADED = Gauge(
    "wine_model_loaded",
    "1 si le modèle est chargé",
    ["model_name"],
)
DRIFT_RATIO = Gauge(
    "wine_data_drift_ratio",
    "Proportion de variables en drift",
)
DRIFT_DETECTED = Gauge(
    "wine_data_drift_detected",
    "1 si drift global",
)
ROLLING_ACCURACY = Gauge(
    "wine_model_rolling_accuracy",
    "Accuracy glissante des feedbacks",
)


def refresh_quality_metrics() -> None:
    try:
        with connect() as database:
            with database.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT drift_ratio, drift_detected
                    FROM monitoring_runs
                    WHERE drift_ratio IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                drift = cursor.fetchone()

                cursor.execute(
                    """
                    SELECT rolling_accuracy
                    FROM monitoring_runs
                    WHERE rolling_accuracy IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                )
                performance = cursor.fetchone()

        if drift:
            DRIFT_RATIO.set(float(drift[0] or 0.0))
            DRIFT_DETECTED.set(1 if drift[1] else 0)

        if performance:
            ROLLING_ACCURACY.set(float(performance[0]))
    except Exception:
        LOGGER.exception(
            "monitoring_metrics_refresh_failed",
            extra={"request_id": "-"},
        )


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)

        # Rend l'identifiant accessible dans les routes.
        request.state.request_id = request_id
        started = time.perf_counter()
        status = 500

        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            duration = time.perf_counter() - started

            HTTP_REQUESTS.labels(
                request.method,
                path,
                str(status),
            ).inc()
            HTTP_LATENCY.labels(request.method, path).observe(duration)
            if status >= 500:
                HTTP_ERRORS.labels(request.method, path).inc()

            LOGGER.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": path,
                    "status_code": status,
                    "duration_ms": round(duration * 1000, 3),
                },
            )


def install_observability(app: FastAPI) -> None:
    app.add_middleware(MetricsMiddleware)

    @app.get(
        "/metrics/prometheus",
        include_in_schema=False,
        response_class=Response,
    )
    def metrics() -> Response:
        refresh_quality_metrics()
        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
