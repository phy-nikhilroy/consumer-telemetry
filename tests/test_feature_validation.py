"""
Unit tests for feature validation, quality audits, and feature selection.
"""

import json
from pathlib import Path
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

    assert audit["total_rows"] == 50
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
