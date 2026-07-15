from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.monitoring.database import connect

TARGETS = {
    "quality",
    "quality_class",
    "quality_category",
    "quality_label",
    "target",
    "label",
}


def load_production_features(limit: int) -> pd.DataFrame:
    query = """
        SELECT features
        FROM prediction_events
        WHERE created_at >= now() - interval '30 days'
        ORDER BY created_at DESC
        LIMIT %s
    """
    with connect() as database:
        rows = pd.read_sql_query(query, database, params=(limit,))
    if rows.empty:
        return pd.DataFrame()
    return rows["features"].apply(pd.Series)


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = 0.05,
    minimum_statistic: float = 0.10,
) -> dict:
    reference_numeric = set(
        reference.select_dtypes(include=[np.number]).columns
    ) - TARGETS
    current_numeric = set(
        current.select_dtypes(include=[np.number]).columns
    ) - TARGETS
    features = sorted(reference_numeric & current_numeric)

    if not features:
        raise ValueError("Aucune variable numérique commune")

    corrected_alpha = alpha / len(features)
    details = {}

    for feature in features:
        ref = reference[feature].dropna().to_numpy()
        cur = current[feature].dropna().to_numpy()
        test = ks_2samp(ref, cur, alternative="two-sided", method="auto")
        drifted = bool(
            test.pvalue < corrected_alpha
            and test.statistic >= minimum_statistic
        )
        details[feature] = {
            "ks_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "drifted": drifted,
        }

    count = sum(int(item["drifted"]) for item in details.values())
    ratio = count / len(features)

    return {
        "reference_samples": len(reference),
        "current_samples": len(current),
        "features_tested": len(features),
        "features_drifted": count,
        "drift_ratio": ratio,
        "drift_detected": ratio >= 0.30,
        "features": details,
    }


def save_report(report: dict) -> None:
    query = """
        INSERT INTO monitoring_runs (
            sample_size,
            drift_ratio,
            drift_detected,
            retraining_required,
            reason,
            report
        )
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
    """
    retraining = bool(report["drift_detected"])
    reason = (
        f"drift_ratio={report['drift_ratio']:.3f}"
        if retraining
        else "none"
    )
    with connect() as database:
        with database.cursor() as cursor:
            cursor.execute(
                query,
                (
                    report["current_samples"],
                    report["drift_ratio"],
                    report["drift_detected"],
                    retraining,
                    reason,
                    json.dumps(report),
                ),
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        default="data/processed/train.csv",
    )
    parser.add_argument(
        "--current",
        default=None,
        help="CSV courant. Sans ce paramètre, lit prediction_events.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/monitoring/drift_report.json",
    )
    parser.add_argument("--minimum-samples", type=int, default=100)
    parser.add_argument("--maximum-samples", type=int, default=2000)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    current = (
        pd.read_csv(args.current)
        if args.current
        else load_production_features(args.maximum_samples)
    )

    if len(current) < args.minimum_samples:
        report = {
            "drift_detected": False,
            "reason": "insufficient_samples",
            "current_samples": len(current),
        }
        print(json.dumps(report, indent=2))
        return 0

    report = compute_drift_report(reference, current)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    save_report(report)
    print(json.dumps(report, indent=2))
    return 2 if report["drift_detected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
