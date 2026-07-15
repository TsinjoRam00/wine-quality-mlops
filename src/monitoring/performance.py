from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from src.monitoring.database import connect


def load_feedback(limit: int) -> tuple[list[str], list[str]]:
    query = """
        SELECT actual_class, predicted_class
        FROM prediction_events
        WHERE actual_class IS NOT NULL
        ORDER BY feedback_at DESC
        LIMIT %s
    """

    with connect() as database:
        with database.cursor() as cursor:
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()

    actual = [row[0] for row in rows]
    predicted = [row[1] for row in rows]
    return actual, predicted


def evaluate(
    *,
    limit: int = 500,
    minimum_feedbacks: int = 50,
    minimum_accuracy: float = 0.55,
) -> dict:
    actual, predicted = load_feedback(limit)

    if len(actual) < minimum_feedbacks:
        return {
            "labeled_sample_size": len(actual),
            "enough_feedback": False,
            "performance_degraded": False,
        }

    accuracy = float(accuracy_score(actual, predicted))
    weighted_f1 = float(
        f1_score(
            actual,
            predicted,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "labeled_sample_size": len(actual),
        "enough_feedback": True,
        "rolling_accuracy": accuracy,
        "rolling_f1_weighted": weighted_f1,
        "performance_degraded": accuracy < minimum_accuracy,
    }


def save_report(report: dict) -> None:
    query = """
        INSERT INTO monitoring_runs (
            labeled_sample_size,
            rolling_accuracy,
            rolling_f1_weighted,
            retraining_required,
            reason,
            report
        )
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
    """
    degraded = bool(report.get("performance_degraded", False))
    accuracy = report.get("rolling_accuracy")
    reason = (
        f"rolling_accuracy={accuracy:.3f}"
        if degraded and accuracy is not None
        else "none"
    )
    with connect() as database:
        with database.cursor() as cursor:
            cursor.execute(
                query,
                (
                    report.get("labeled_sample_size", 0),
                    accuracy,
                    report.get("rolling_f1_weighted"),
                    degraded,
                    reason,
                    json.dumps(report),
                ),
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--minimum-feedbacks", type=int, default=50)
    parser.add_argument("--minimum-accuracy", type=float, default=0.55)
    parser.add_argument(
        "--output",
        default="artifacts/monitoring/performance_report.json",
    )
    args = parser.parse_args()

    report = evaluate(
        limit=args.limit,
        minimum_feedbacks=args.minimum_feedbacks,
        minimum_accuracy=args.minimum_accuracy,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    save_report(report)
    print(json.dumps(report, indent=2))
    return 2 if report["performance_degraded"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
