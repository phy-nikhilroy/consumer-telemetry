"""
Unit tests for feature validation, quality audits, and feature selection.
"""

import json
from pathlib import Path
import duckdb
import polars as pl
import pytest

from src.features.feature_validation import (
    audit_feature_quality,
    analyze_correlations_and_redundancy,
    compute_feature_importances,
    select_features,
)


def test_audit_feature_quality_train_matrix():
    """Verifies that the generated training feature matrix is clean with no infinities."""
    train_path = Path("data/features/train_features.parquet")
    if not train_path.exists():
        pytest.skip("Train feature matrix not available")

    df = pl.read_parquet(train_path)
    audit = audit_feature_quality(df)

    # Row count tracks the simulated population size, so assert internal consistency.
    assert audit["total_rows"] == df.height
    assert audit["total_rows"] > 0
    assert audit["is_clean"] is True
    assert len(audit["infinite_columns"]) == 0


def test_analyze_correlations_and_collinearity():
    """Verifies correlation calculations and collinear feature detection."""
    train_path = Path("data/features/train_features.parquet")
    if not train_path.exists():
        pytest.skip("Train feature matrix not available")

    df = pl.read_parquet(train_path)
    target_corrs, collinear = analyze_correlations_and_redundancy(df, collinearity_threshold=0.90)

    # All candidate features should have a correlation computed
    assert len(target_corrs) >= 30
    assert "will_purchase_next_7d" not in target_corrs
    assert all(-1.0 <= r <= 1.0 for r in target_corrs.values())

    # Should detect high collinearity pairs like total_events_30d and total_events_lifetime
    assert len(collinear) > 0


def test_selected_features_json_schema():
    """Verifies that selected_features.json exists and conforms to required schema."""
    json_path = Path("data/features/selected_features.json")
    if not json_path.exists():
        pytest.skip("selected_features.json not generated yet")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "selected_features" in data
    assert "target_variables" in data
    assert "top_predictive_features" in data

    assert len(data["selected_features"]) >= 15
    assert "will_purchase_next_7d" in data["target_variables"]
    assert len(data["top_predictive_features"]) >= 3


def _v2_info(tmp_path, df):
    return select_features(df, output_path=tmp_path / "selected_features.json")


def test_zero_variance_feature_is_dropped(tmp_path):
    """A constant column is excluded from every target list; others keep matrix order."""
    df = pl.DataFrame({
        "user_id": ["a", "b", "c", "d"],
        "const": [1.0, 1.0, 1.0, 1.0],
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": [0, 1, 0, 1],
        "top_category_l1": ["x", "y", "x", "y"],
        "will_purchase_next_7d": [1, 0, 1, 1],
        "will_switch_app_next_7d": [0, 0, 1, 0],
    })
    info = _v2_info(tmp_path, df)
    assert info["schema_version"] == 2
    assert "const" in info["dropped_zero_variance"]
    for target, names in info["features_by_target"].items():
        assert "const" not in names
        assert names[:2] == ["f1", "f2"] or target == "switched_given_purchase"
    assert info["selected_features"] == ["f1", "f2"]
    assert "top_category_l1" not in info["candidate_features"]


def test_zero_variance_is_evaluated_per_target(tmp_path):
    """A feature constant only on purchasers is dropped for the conditional target alone."""
    df = pl.DataFrame({
        "g": [5.0, 5.0, 1.0, 2.0],
        "h": [1.0, 2.0, 3.0, 4.0],
        "will_purchase_next_7d": [1, 1, 0, 0],
        "will_switch_app_next_7d": [0, 1, 0, 0],
    })
    info = _v2_info(tmp_path, df)
    by = info["features_by_target"]
    assert by["switched_given_purchase"] == ["h"]
    assert by["will_purchase_next_7d"] == ["g", "h"]


def test_cross_app_features_are_selected(tmp_path):
    """
    Cross-app features survive selection for both binary targets.

    Selection runs on a train matrix built at the configured reference dates in a temp
    directory, so the assertion holds for the split the models actually train on rather
    than whatever matrix happens to be committed.
    """
    from src.features.feature_store import generate_train_test_datasets
    from src.features.feature_validation import select_features

    db_path = Path("data/processed/consumer.duckdb")
    if not db_path.exists():
        pytest.skip("Warehouse database not available")
    train_df, _ = generate_train_test_datasets(db_path=db_path, features_dir=tmp_path)
    data = select_features(train_df, output_path=tmp_path / "selected_features.json")
    assert data["schema_version"] == 2
    for target in ("will_purchase_next_7d", "will_switch_app_next_7d"):
        for name in ("cross_app_comparison_count", "app_entropy", "num_apps_used"):
            assert name in data["features_by_target"][target], (target, name)
