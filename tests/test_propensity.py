"""
Unit tests for the LightGBM purchase propensity model.
"""

from pathlib import Path

import numpy as np
import pytest

from src.config import PRIMARY_TARGET
from src.features.feature_store import get_modeling_matrices
from src.models.propensity import (
    compute_scale_pos_weight,
    evaluate_propensity_model,
    load_model,
    plot_feature_importance,
    save_model,
    train_propensity_model,
)

TRAIN_PATH = Path("data/features/train_features.parquet")
TEST_PATH = Path("data/features/test_features.parquet")


def _require_matrices():
    """Skips a test when the feature store has not been generated."""
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        pytest.skip("Feature matrices not available")


def test_compute_scale_pos_weight():
    """Verifies the negative-to-positive ratio used for class reweighting."""
    assert compute_scale_pos_weight(np.array([0, 0, 1, 1])) == pytest.approx(1.0)
    assert compute_scale_pos_weight(np.array([0, 0, 0, 1])) == pytest.approx(3.0)
    assert compute_scale_pos_weight(np.array([0, 1, 1, 1, 1])) == pytest.approx(0.25)
    # A target with no positives must not divide by zero.
    assert compute_scale_pos_weight(np.array([0, 0, 0])) == pytest.approx(1.0)


def test_train_propensity_model_beats_random_on_held_out_split():
    """Trains on the train reference date and verifies real signal on the later test split."""
    _require_matrices()

    X_train, y_train, feature_names = get_modeling_matrices(split="train", target_col=PRIMARY_TARGET)
    X_test, y_test, test_names = get_modeling_matrices(split="test", target_col=PRIMARY_TARGET)

    assert feature_names == test_names
    assert X_train.shape[1] == len(feature_names)

    model = train_propensity_model(X_train, y_train)
    metrics = evaluate_propensity_model(model, X_test, y_test)

    # The simulated population carries latent purchase propensity, so the model must
    # rank users meaningfully better than chance on an unseen target window.
    assert metrics["roc_auc"] > 0.70
    assert metrics["pr_auc"] > 0.70
    assert 0.0 <= metrics["f1"] <= 1.0
    # Log loss must at least beat predicting the base rate for every user.
    base_rate = float(np.mean(y_test))
    naive_log_loss = -(
        base_rate * np.log(base_rate) + (1.0 - base_rate) * np.log(1.0 - base_rate)
    )
    assert metrics["log_loss"] < naive_log_loss

    for key in ("roc_auc", "pr_auc", "f1", "log_loss", "brier_score", "confusion_matrix"):
        assert key in metrics


def test_predicted_probabilities_are_valid_and_spread():
    """Verifies predictions are probabilities that actually span a usable range."""
    _require_matrices()

    X_train, y_train, _ = get_modeling_matrices(split="train", target_col=PRIMARY_TARGET)
    X_test, _, _ = get_modeling_matrices(split="test", target_col=PRIMARY_TARGET)

    model = train_propensity_model(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    # A model whose probabilities collapse onto the base rate is useless for ranking
    # and for the decile lift analysis built on top of it.
    assert proba.max() - proba.min() > 0.30


def test_save_and_load_model_roundtrip(tmp_path):
    """Verifies joblib serialization preserves predictions exactly."""
    _require_matrices()

    X_train, y_train, _ = get_modeling_matrices(split="train", target_col=PRIMARY_TARGET)
    model = train_propensity_model(X_train, y_train)

    out = tmp_path / "propensity_test.joblib"
    saved = save_model(model, out)
    assert saved.exists()

    reloaded = load_model(out)
    np.testing.assert_allclose(
        model.predict_proba(X_train)[:, 1],
        reloaded.predict_proba(X_train)[:, 1],
    )


def test_load_model_missing_path_raises(tmp_path):
    """Verifies a clear error when the artifact is absent."""
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.joblib")


def test_plot_feature_importance_writes_image(tmp_path):
    """Verifies the top-N feature importance chart is written to disk."""
    _require_matrices()

    X_train, y_train, feature_names = get_modeling_matrices(split="train", target_col=PRIMARY_TARGET)
    model = train_propensity_model(X_train, y_train)

    out = tmp_path / "importance.png"
    saved = plot_feature_importance(model, feature_names, top_n=15, output_path=out)

    assert saved.exists()
    assert saved.stat().st_size > 1000


def _grouped_data(n_users: int = 60, rows_per_user: int = 3):
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(n_users), rows_per_user)
    user_y = rng.integers(0, 2, n_users)
    y = np.repeat(user_y, rows_per_user)
    return y, groups


def test_grouped_validation_split_keeps_users_together():
    from src.models.propensity import grouped_validation_split

    y, groups = _grouped_data()
    fit_idx, val_idx = grouped_validation_split(y, groups)

    assert set(groups[fit_idx]).isdisjoint(set(groups[val_idx]))
    assert sorted(np.concatenate([fit_idx, val_idx]).tolist()) == list(range(len(y)))
    assert len(val_idx) > 0 and len(fit_idx) > len(val_idx)


def test_grouped_validation_split_is_deterministic():
    from src.models.propensity import grouped_validation_split

    y, groups = _grouped_data()
    a = grouped_validation_split(y, groups, random_state=7)
    b = grouped_validation_split(y, groups, random_state=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_train_propensity_model_accepts_groups():
    from src.models.propensity import train_propensity_model

    y, groups = _grouped_data(n_users=120)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(len(y), 4)) + y[:, None] * 0.8
    model = train_propensity_model(X, y, groups=groups)
    assert model.predict_proba(X).shape == (len(y), 2)
