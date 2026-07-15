from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import compute_drift_report


@pytest.mark.unit
def test_no_drift_for_similar_distributions():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "alcohol": rng.normal(10.5, 1.0, 600),
            "density": rng.normal(0.996, 0.002, 600),
        }
    )
    current = pd.DataFrame(
        {
            "alcohol": rng.normal(10.5, 1.0, 600),
            "density": rng.normal(0.996, 0.002, 600),
        }
    )
    report = compute_drift_report(reference, current)
    assert report["drift_detected"] is False


@pytest.mark.unit
def test_drift_for_shifted_distributions():
    rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        {
            "alcohol": rng.normal(10.0, 0.5, 600),
            "density": rng.normal(0.996, 0.001, 600),
            "pH": rng.normal(3.3, 0.1, 600),
        }
    )
    current = pd.DataFrame(
        {
            "alcohol": rng.normal(13.0, 0.5, 600),
            "density": rng.normal(1.005, 0.001, 600),
            "pH": rng.normal(4.0, 0.1, 600),
        }
    )
    report = compute_drift_report(reference, current)
    assert report["drift_detected"] is True
    assert report["drift_ratio"] >= 0.30
