"""
Contract tests for the persisted model artifacts.

These load each committed artifact from disk exactly as a serving layer would and check
the guarantees a caller depends on: probabilities inside the unit interval, stable
feature ordering, and reproducible predictions. They deliberately do not retrain
anything, so a stale or corrupted artifact is caught rather than silently replaced.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
from lifetimes import BetaGeoFitter, GammaGammaFitter

from src.config import (
    CLV_BGNBD_PATH,
    CLV_GAMMAGAMMA_PATH,
    PRIMARY_TARGET,
    PROPENSITY_CALIBRATED_PATH,
    PROPENSITY_MODEL_PATH,
    SEGMENTATION_MODEL_PATH,
    SWITCHING_MODEL_PATH,
    USER_CLV_PATH,
    USER_SEGMENTS_PATH,
)
from src.features.feature_store import get_modeling_matrices
from src.models.propensity import load_model
from src.models.segmentation import SEGMENTATION_SCALER_PATH
from src.models.switching import SWITCHING_MULTICLASS_PATH, get_switching_matrices


def _require(path: Path):
    """Skips a test when the artifact has not been generated yet."""
    if not Path(path).exists():
        pytest.skip(f"Artifact not available at {path}")


@pytest.fixture(scope="module")
def propensity_matrices():
    """Loads the test split modeling matrix."""
    if not Path("data/features/test_features.parquet").exists():
        pytest.skip("Feature matrices not available")
    return get_modeling_matrices(split="test", target_col=PRIMARY_TARGET)


def test_propensity_model_loads_and_returns_probabilities(propensity_matrices):
    """Verifies the propensity artifact predicts calibrated probabilities in [0, 1]."""
    _require(PROPENSITY_MODEL_PATH)
    X_test, y_test, feature_names = propensity_matrices

    model = load_model(PROPENSITY_MODEL_PATH)
    proba = model.predict_proba(X_test)

    assert proba.shape == (X_test.shape[0], 2)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    # Each row is a distribution over the two classes.
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(X_test.shape[0]), rtol=1e-6)

    positive = proba[:, 1]
    assert positive.min() >= 0.0 and positive.max() <= 1.0
    # A loaded model whose scores collapse to a constant cannot rank anyone.
    assert positive.max() - positive.min() > 0.2

    # The artifact must expect exactly the features the feature store produces.
    assert model.n_features_in_ == len(feature_names)


def test_propensity_predictions_are_deterministic(propensity_matrices):
    """Verifies repeated prediction from the same artifact is reproducible."""
    _require(PROPENSITY_MODEL_PATH)
    X_test, _, _ = propensity_matrices

    model = load_model(PROPENSITY_MODEL_PATH)
    first = model.predict_proba(X_test)[:, 1]
    second = model.predict_proba(X_test)[:, 1]
    np.testing.assert_array_equal(first, second)

    reloaded = load_model(PROPENSITY_MODEL_PATH)
    np.testing.assert_allclose(first, reloaded.predict_proba(X_test)[:, 1])


def test_calibrated_propensity_model_returns_probabilities(propensity_matrices):
    """Verifies the calibrated variant also yields valid probabilities."""
    _require(PROPENSITY_CALIBRATED_PATH)
    X_test, _, _ = propensity_matrices

    model = joblib.load(PROPENSITY_CALIBRATED_PATH)
    proba = model.predict_proba(X_test)[:, 1]

    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    assert np.isfinite(proba).all()


def test_switching_binary_model_returns_probabilities():
    """Verifies the binary switching artifact predicts probabilities in [0, 1]."""
    _require(SWITCHING_MODEL_PATH)
    X_test, y_test, _, feature_names = get_switching_matrices("test")

    model = joblib.load(SWITCHING_MODEL_PATH)
    proba = model.predict_proba(X_test)[:, 1]

    assert len(proba) == X_test.shape[0]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    assert model.n_features_in_ == len(feature_names)


def test_switching_multiclass_model_returns_a_distribution():
    """Verifies the three-class artifact returns a proper probability distribution."""
    _require(SWITCHING_MULTICLASS_PATH)
    X_test, _, y_mc, _ = get_switching_matrices("test")

    model = joblib.load(SWITCHING_MULTICLASS_PATH)
    proba = model.predict_proba(X_test)

    assert proba.shape == (X_test.shape[0], 3)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(X_test.shape[0]), rtol=1e-6)

    predictions = np.argmax(proba, axis=1)
    assert set(predictions.tolist()).issubset({0, 1, 2})


def test_clv_models_load_from_disk_and_predict():
    """
    Verifies both CLV fitters reload and still produce usable predictions.

    They are persisted with the library's dill-based save_model rather than joblib,
    because the fitters attach a lambda during fit that pickle cannot serialize.
    """
    _require(CLV_BGNBD_PATH)
    _require(CLV_GAMMAGAMMA_PATH)

    bgf = BetaGeoFitter()
    bgf.load_model(str(CLV_BGNBD_PATH))
    ggf = GammaGammaFitter()
    ggf.load_model(str(CLV_GAMMAGAMMA_PATH))

    for name in ("r", "alpha", "a", "b"):
        assert np.isfinite(float(bgf.params_[name]))
    for name in ("p", "q", "v"):
        assert np.isfinite(float(ggf.params_[name]))

    frequency = np.array([0.0, 1.0, 5.0, 12.0])
    recency = np.array([0.0, 5.0, 20.0, 27.0])
    age = np.array([10.0, 15.0, 25.0, 29.0])

    alive = np.asarray(
        bgf.conditional_probability_alive(frequency, recency, age), dtype=float
    )
    # P(alive) is a probability and must respect its bounds.
    assert alive.min() >= 0.0
    assert alive.max() <= 1.0
    assert np.isfinite(alive).all()

    expected = np.asarray(
        bgf.conditional_expected_number_of_purchases_up_to_time(30, frequency, recency, age),
        dtype=float,
    )
    assert (expected[np.isfinite(expected)] >= 0.0).all()


def test_segmentation_model_and_scaler_round_trip():
    """Verifies the KMeans artifact and its scaler agree on feature count."""
    _require(SEGMENTATION_MODEL_PATH)
    _require(SEGMENTATION_SCALER_PATH)

    kmeans = joblib.load(SEGMENTATION_MODEL_PATH)
    scaler = joblib.load(SEGMENTATION_SCALER_PATH)

    assert kmeans.n_features_in_ == scaler.n_features_in_
    assert kmeans.cluster_centers_.shape[1] == scaler.n_features_in_
    assert kmeans.n_clusters >= 2

    # Scaling then predicting must assign a valid cluster to every row.
    raw = np.tile(scaler.mean_, (5, 1))
    labels = kmeans.predict(scaler.transform(raw))
    assert len(labels) == 5
    assert set(labels.tolist()).issubset(set(range(kmeans.n_clusters)))


def test_clv_predictions_file_is_well_formed():
    """Verifies the persisted CLV table is finite, non-negative, and consistent."""
    _require(USER_CLV_PATH)
    df = pl.read_parquet(USER_CLV_PATH)

    assert df.height > 0
    assert df["user_id"].n_unique() == df.height

    for column in ("predicted_clv", "predicted_purchases", "predicted_avg_order_value"):
        assert df[column].is_finite().all()
        assert df[column].min() >= 0.0

    assert df["probability_alive"].min() >= 0.0
    assert df["probability_alive"].max() <= 1.0

    product = df["predicted_purchases"] * df["predicted_avg_order_value"]
    np.testing.assert_allclose(
        df["predicted_clv"].to_numpy(), product.to_numpy(), rtol=1e-6
    )


def test_segment_assignments_file_is_well_formed():
    """Verifies every user carries a named segment and a valid cluster id."""
    _require(USER_SEGMENTS_PATH)
    df = pl.read_parquet(USER_SEGMENTS_PATH)

    assert df.height > 0
    assert df["user_id"].n_unique() == df.height
    assert df["kmeans_cluster"].null_count() == 0
    assert df["kmeans_cluster"].min() >= 0
    assert df["segment_name"].null_count() == 0

    # HDBSCAN may legitimately assign -1 for noise, but nothing below that.
    assert df["hdbscan_cluster"].min() >= -1

    # Each KMeans cluster maps to exactly one segment name.
    mapping = df.group_by("kmeans_cluster").agg(
        pl.col("segment_name").n_unique().alias("names")
    )
    assert mapping["names"].max() == 1


def test_metrics_summaries_are_present_and_sane():
    """Verifies each model wrote a summary with metrics in valid ranges."""
    from src.config import ARTIFACTS_DIR

    expectations = {
        "propensity_metrics.json": ("test", "roc_auc"),
        "switching_metrics.json": ("binary_test", "roc_auc"),
    }
    for filename, (block, metric) in expectations.items():
        path = ARTIFACTS_DIR / filename
        _require(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        value = payload[block][metric]
        assert 0.0 <= value <= 1.0
        # A persisted model that cannot beat chance should never have been committed.
        assert value > 0.5
