from __future__ import annotations

import argparse
import json
from pathlib import Path


def decide(
    drift_report: dict,
    performance_report: dict,
) -> dict:
    reasons: list[str] = []

    if (
        drift_report.get("current_samples", 0) >= 100
        and drift_report.get("drift_ratio", 0.0) >= 0.30
    ):
        reasons.append(
            f"drift_ratio={drift_report['drift_ratio']:.3f}"
        )

    if (
        performance_report.get("labeled_sample_size", 0) >= 50
        and performance_report.get("rolling_accuracy", 1.0) < 0.55
    ):
        reasons.append(
            "rolling_accuracy="
            f"{performance_report['rolling_accuracy']:.3f}"
        )

    return {
        "retraining_required": bool(reasons),
        "reason": ",".join(reasons) if reasons else "none",
    }


def read_json(path: str) -> dict:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drift-report",
        default="artifacts/monitoring/drift_report.json",
    )
    parser.add_argument(
        "--performance-report",
        default="artifacts/monitoring/performance_report.json",
    )
    args = parser.parse_args()

    decision = decide(
        read_json(args.drift_report),
        read_json(args.performance_report),
    )
    print(json.dumps(decision, indent=2))
    return 2 if decision["retraining_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
