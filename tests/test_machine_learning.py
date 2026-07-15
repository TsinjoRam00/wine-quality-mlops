from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from tests.conftest import load_split, target_column


@pytest.fixture
def sample_xy(processed_dir):
    train = load_split(processed_dir, "train")
    target = target_column(train)
    sample = train.sample(n=min(400, len(train)), random_state=42)
    X = sample.drop(columns=[target]).select_dtypes(include=[np.number])
    y = sample[target]
    return X, y


@pytest.mark.unit
def test_model_fits_and_predicts(sample_xy):
    X, y = sample_xy
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y)
    prediction = model.predict(X)
    assert len(prediction) == len(X)
    assert set(np.unique(prediction)).issubset(set(np.unique(y)))


@pytest.mark.unit
def test_metrics_are_bounded(sample_xy):
    X, y = sample_xy
    model = LogisticRegression(max_iter=500, random_state=42).fit(X, y)
    prediction = model.predict(X)
    assert 0 <= accuracy_score(y, prediction) <= 1
    assert 0 <= f1_score(y, prediction, average="weighted", zero_division=0) <= 1


@pytest.mark.unit
def test_model_serialization(tmp_path, sample_xy):
    X, y = sample_xy
    model = LogisticRegression(max_iter=500, random_state=42).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(
        model.predict(X.iloc[:20]),
        restored.predict(X.iloc[:20]),
    )


@pytest.mark.unit
def test_training_is_reproducible(sample_xy):
    X, y = sample_xy
    first = LogisticRegression(max_iter=500, random_state=42).fit(X, y)
    second = LogisticRegression(max_iter=500, random_state=42).fit(X, y)
    np.testing.assert_array_equal(
        first.predict(X.iloc[:20]),
        second.predict(X.iloc[:20]),
    )
