from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from src.monitoring.database import connect


def save_prediction(
    *,
    request_id: str | None,
    model_name: str,
    model_version: str | None,
    model_alias: str | None,
    features: dict[str, Any],
    predicted_class: str | int,
    probabilities: dict[str, float] | list[float] | None,
    latency_ms: float | None,
) -> str:
    query = """
        INSERT INTO prediction_events (
            request_id,
            model_name,
            model_version,
            model_alias,
            features,
            predicted_class,
            probabilities,
            latency_ms
        )
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s)
        RETURNING prediction_id::text
    """

    with connect() as database:
        with database.cursor() as cursor:
            cursor.execute(
                query,
                (
                    request_id,
                    model_name,
                    model_version,
                    model_alias,
                    json.dumps(features),
                    str(predicted_class),
                    json.dumps(probabilities)
                    if probabilities is not None
                    else None,
                    latency_ms,
                ),
            )
            return cursor.fetchone()[0]


def attach_feedback(
    *,
    prediction_id: UUID,
    actual_class: str | int,
    comment: str | None,
) -> bool:
    query = """
        UPDATE prediction_events
        SET
            actual_class = %s,
            feedback_at = now(),
            feedback_comment = %s,
            is_correct = (predicted_class = %s)
        WHERE prediction_id = %s
    """

    with connect() as database:
        with database.cursor() as cursor:
            cursor.execute(
                query,
                (
                    str(actual_class),
                    comment,
                    str(actual_class),
                    str(prediction_id),
                ),
            )
            return cursor.rowcount == 1
