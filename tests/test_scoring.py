"""
Tests for live scoring: training-serving parity, model loading, and each scorer.

The parity test is the important one. It builds the batch feature matrix the way the
feature store does and the serving features the way the API does, for the same users
and reference date, and asserts both the features and the resulting propensity scores
are identical.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from src.api.scoring import ModelRegistry, UserScorer, compute_user_features, format_reference, risk_segment
from src.api.warehouse import ServingWarehouse
from src.config import PROPENSITY_HIGH_THRESHOLD, PROPENSITY_LOW_THRESHOLD
from src.features.feature_store import build_feature_matrix

REFERENCE = datetime(2026, 10, 21, 0, 0, 0)


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return ModelRegistry()


def test_risk_segment_bands():
    assert risk_segment(PROPENSITY_HIGH_THRESHOLD) == "High"
    assert risk_segment(0.99) == "High"
    assert risk_segment(PROPENSITY_LOW_THRESHOLD) == "Medium"
    assert risk_segment(PROPENSITY_LOW_THRESHOLD - 0.01) == "Low"


def test_registry_loads_models_with_training_feature_order(registry: ModelRegistry):
    assert len(registry.propensity_features) == registry.propensity_model.n_features_in_
    assert len(registry.segmentation_features) == registry.segmentation_scaler.n_features_in_
    assert 0.0 < registry.switching_threshold < 1.0
    assert set(registry.segment_names) == set(range(registry.segmentation_model.n_clusters))
    assert "propensity" in registry.describe()


def test_serving_features_match_batch_feature_store(tiny_warehouse: Path, registry: ModelRegistry):
    con = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        batch = build_feature_matrix(reference_date=format_reference(REFERENCE), con=con)
        users = batch["user_id"].to_list()[:5]
        serving = compute_user_features(con, REFERENCE, user_ids=users)
    finally:
        con.close()

    expected = batch.filter(pl.col("user_id").is_in(users)).select(serving.columns).sort("user_id")
    assert_frame_equal(serving, expected)

    scorer = UserScorer(registry)
    np.testing.assert_allclose(scorer.purchase_probability(serving), scorer.purchase_probability(expected))


def test_scorer_outputs(tiny_warehouse: Path, registry: ModelRegistry):
    con = duckdb.connect(str(tiny_warehouse), read_only=True)
    try:
        features = compute_user_features(con, REFERENCE)
    finally:
        con.close()
    scorer = UserScorer(registry)

    purchase = scorer.purchase_probability(features)
    switching = scorer.switching_probability(features)
    assert purchase.shape == switching.shape == (features.height,)
    assert np.all((purchase >= 0) & (purchase <= 1))
    assert np.all((switching >= 0) & (switching <= 1))
    assert purchase.std() > 0, "scores should vary across users"

    segments = scorer.segments(features)
    assert len(segments) == features.height
    assert {s["segment_name"] for s in segments} <= set(registry.segment_names.values())


def test_scorer_rejects_missing_model_inputs(registry: ModelRegistry):
    with pytest.raises(ValueError, match="missing model inputs"):
        UserScorer(registry).purchase_probability(pl.DataFrame({"user_id": ["u1"]}))


def test_lifetime_value_for_buyer_and_non_buyer(serving_db: Path, registry: ModelRegistry):
    warehouse = ServingWarehouse(serving_db)
    try:
        buyer = warehouse.scalar(
            "SELECT user_id FROM fact_events WHERE action = 'buy' GROUP BY user_id ORDER BY COUNT(*) DESC LIMIT 1"
        )
        warehouse.record_event(
            "window_only", "flipkart",
            {"canonical_product_id": "p1", "canonical_name": "P", "action": "view", "price": 10.0, "currency": "INR"},
        )
        reference = warehouse.reference_date()
        scorer = UserScorer(registry)
        value = scorer.lifetime_value(warehouse.con, buyer, reference)
        assert value is not None
        assert value["predicted_clv_inr"] > 0
        assert 0.0 <= value["probability_alive"] <= 1.0
        assert value["order_value_source"] in {"gamma_gamma", "observed"}
        assert scorer.lifetime_value(warehouse.con, "window_only", reference) is None
    finally:
        warehouse.close()
