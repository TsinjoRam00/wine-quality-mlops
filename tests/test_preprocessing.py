from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from tests.conftest import load_split, target_column


@pytest.mark.unit
def test_processed_artifacts_exist(project_root, processed_dir):
    for split in ("train", "validation", "test"):
        assert (processed_dir / f"{split}.csv").is_file()

    assert (
        project_root
        / "artifacts"
        / "preprocessing"
        / "preprocessing_objects.joblib"
    ).is_file()


@pytest.mark.unit
def test_splits_have_same_schema(processed_dir):
    frames = [load_split(processed_dir, name) for name in ("train", "validation", "test")]
    expected = list(frames[0].columns)
    assert all(list(frame.columns) == expected for frame in frames)


@pytest.mark.unit
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_no_nan_or_infinite(processed_dir, split):
    frame = load_split(processed_dir, split)
    assert not frame.isna().any().any()
    numeric = frame.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


@pytest.mark.unit
def test_target_is_valid(processed_dir):
    train = load_split(processed_dir, "train")
    target = target_column(train)
    assert train[target].nunique() >= 2
    assert not train[target].isna().any()


@pytest.mark.unit
def test_no_source_leakage_between_splits():
    """Vérifie que les lignes sources sont réparties dans un seul split."""
    import pandas as pd

    from src.features.preprocessing import (
        split_train_validation_test,
    )

    row_count = 120
    source_indices = range(1000, 1000 + row_count)

    X = pd.DataFrame(
        {
            "feature_a": range(row_count),
            "feature_b": [value * 2 for value in range(row_count)],
        },
        index=source_indices,
    )

    # Trois classes équilibrées pour permettre la stratification.
    y = pd.Series(
        [index % 3 for index in range(row_count)],
        index=X.index,
        name="quality_class",
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_train_validation_test(
        X=X,
        y=y,
        test_size=0.15,
        validation_size=0.15,
        random_state=42,
    )

    train_indices = set(X_train.index)
    validation_indices = set(X_validation.index)
    test_indices = set(X_test.index)

    # Aucun échantillon source ne doit appartenir à deux ensembles.
    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)

    # Aucun échantillon ne doit être perdu.
    assert (
        train_indices
        | validation_indices
        | test_indices
    ) == set(X.index)

    # Les features et les cibles doivent rester alignées.
    assert set(X_train.index) == set(y_train.index)
    assert set(X_validation.index) == set(y_validation.index)
    assert set(X_test.index) == set(y_test.index)


@pytest.mark.unit
def test_preprocessing_objects_load(project_root):
    path = (
        project_root
        / "artifacts"
        / "preprocessing"
        / "preprocessing_objects.joblib"
    )
    assert joblib.load(path) is not None


@pytest.mark.unit
def test_preprocessing_report_is_json(project_root):
    path = (
        project_root
        / "artifacts"
        / "preprocessing"
        / "preprocessing_report.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(report, dict) and report
