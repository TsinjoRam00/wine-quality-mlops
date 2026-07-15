from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.monitoring.prediction_store import attach_feedback


router = APIRouter(tags=["feedback"])


class FeedbackPayload(BaseModel):
    prediction_id: UUID

    # Même espace de classes que predicted_class.
    actual_class: int = Field(ge=0)

    comment: str | None = Field(
        default=None,
        max_length=500,
    )


@router.post("/feedback")
def submit_feedback(payload: FeedbackPayload) -> dict[str, str]:
    updated = attach_feedback(
        prediction_id=payload.prediction_id,
        actual_class=payload.actual_class,
        comment=payload.comment,
    )

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="prediction_id introuvable",
        )

    return {
        "status": "accepted",
        "prediction_id": str(payload.prediction_id),
    }
