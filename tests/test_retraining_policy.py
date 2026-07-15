from __future__ import annotations

import pytest

from src.monitoring.retraining_policy import decide


@pytest.mark.unit
def test_no_retraining_without_signal():
    decision = decide(
        {"current_samples": 500, "drift_ratio": 0.10},
        {"labeled_sample_size": 100, "rolling_accuracy": 0.70},
    )
    assert decision["retraining_required"] is False


@pytest.mark.unit
def test_retraining_on_data_drift():
    decision = decide(
        {"current_samples": 500, "drift_ratio": 0.45},
        {"labeled_sample_size": 10},
    )
    assert decision["retraining_required"] is True
    assert "drift_ratio" in decision["reason"]


@pytest.mark.unit
def test_retraining_on_performance_drop():
    decision = decide(
        {"current_samples": 50, "drift_ratio": 0.0},
        {"labeled_sample_size": 100, "rolling_accuracy": 0.40},
    )
    assert decision["retraining_required"] is True
    assert "rolling_accuracy" in decision["reason"]
