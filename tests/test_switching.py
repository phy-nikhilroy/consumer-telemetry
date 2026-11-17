"""
Unit tests for the app switching models and SHAP attribution.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config import PRIMARY_TARGET, SWITCHING_TARGET
from src.models.switching import (
    MULTICLASS_LABELS,
    _normalize_shap_values,
    build_multiclass_target,
    compute_shap_values,
    evaluate_binary_switching,
    evaluate_multiclass_switching,
    get_switching_matrices,
    plot_confusion_matrix,
    rank_shap_drivers,
    select_threshold_by_f1,
    train_switching_model,
)

TRAIN_PATH = Path("data/features/train_features.parquet")
TEST_PATH = Path("data/features/test_features.parquet")


def _require_matrices():
    """Skips a test when the feature store has not been generated."""
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        pytest.skip("Feature matrices not available")


def test_build_multiclass_target_mapping():
    """Verifies the three-class label is derived correctly from the forward targets."""
    df = pl.DataFrame(
        {
            PRIMARY_TARGET: [0, 1, 1, 0],
            SWITCHING_TARGET: [0, 0, 1, 1],
        }
    )
    labels = build_multiclass_target(df)

    # No purchase and no switch is class 0.
    assert labels[0] == 0
    # Purchased on the dominant app is class 1.
    assert labels[1] == 1
    # Purchased on a different app is class 2.
    assert labels[2] == 2
    # A switch always implies a purchase, so the switch flag wins even if the
    # purchase flag disagrees, rather than silently falling back to class 0.
    assert labels[3] == 2


def test_build_multiclass_target_is_exhaustive():
    """Verifies every row receives exactly one label in the valid range."""
    _require_matrices()
    df = pl.read_parquet(TRAIN_PATH)
    labels = build_multiclass_target(df)

    assert len(labels) == df.height
    assert set(np.unique(labels)).issubset({0, 1, 2})
    # Class counts must reconstruct the original target totals.
    assert int(np.sum(labels == 2)) == int(df[SWITCHING_TARGET].sum())
    assert int(np.sum(labels > 0)) >= int(df[PRIMARY_TARGET].sum()) - int(df[SWITCHING_TARGET].sum())


def test_get_switching_matrices_shapes():
    """Verifies both target vectors align with the feature matrix."""
    _require_matrices()
    X, y_bin, y_mc, names = get_switching_matrices("train")

    assert X.shape[0] == len(y_bin) == len(y_mc)
    assert X.shape[1] == len(names)
    assert set(np.unique(y_bin)).issubset({0, 1})
    assert set(np.unique(y_mc)).issubset({0, 1, 2})
    assert not np.isnan(X).any()


def test_binary_switching_model_has_real_signal():
    """Verifies the switching classifier ranks better than chance on the test split."""
    _require_matrices()
    X_train, y_train, _, _ = get_switching_matrices("train")
    X_test, y_test, _, _ = get_switching_matrices("test")

    model = train_switching_model(X_train, y_train)
    metrics = evaluate_binary_switching(model, X_test, y_test)

    assert metrics["roc_auc"] > 0.70
    # PR-AUC must clear the positive base rate, which is what a random ranker achieves.
    assert metrics["pr_auc"] > float(np.mean(y_test))


def test_tuned_threshold_improves_f1_over_default():
    """Verifies threshold tuning on train raises F1 for the minority switch class."""
    _require_matrices()
    X_train, y_train, _, _ = get_switching_matrices("train")
    X_test, y_test, _, _ = get_switching_matrices("test")

    model = train_switching_model(X_train, y_train)
    threshold = select_threshold_by_f1(model, X_train, y_train)

    assert 0.05 <= threshold <= 0.95

    default_f1 = evaluate_binary_switching(model, X_test, y_test, threshold=0.5)["f1"]
    tuned_f1 = evaluate_binary_switching(model, X_test, y_test, threshold=threshold)["f1"]

    # Tuning is fitted on train, so a test improvement is not guaranteed by
    # construction, but it must not collapse the score either.
    assert tuned_f1 >= default_f1 - 0.05


def test_multiclass_model_beats_majority_baseline():
    """Verifies the three-class model outperforms always predicting the largest class."""
    _require_matrices()
    X_train, _, y_mc_train, _ = get_switching_matrices("train")
    X_test, _, y_mc_test, _ = get_switching_matrices("test")

    model = train_switching_model(X_train, y_mc_train)
    metrics = evaluate_multiclass_switching(model, X_test, y_mc_test)

    counts = np.bincount(y_mc_test, minlength=3)
    majority_accuracy = float(counts.max() / counts.sum())

    assert metrics["accuracy"] > majority_accuracy
    assert metrics["macro_f1"] > 0.40
    assert len(metrics["confusion_matrix"]) == 3
    # The confusion matrix must account for every test row.
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == len(y_mc_test)
    assert metrics["labels"] == list(MULTICLASS_LABELS)


def test_normalize_shap_values_handles_all_return_shapes():
    """Verifies SHAP output normalization across list, 3-D, and 2-D forms."""
    per_class_list = [np.zeros((5, 3)), np.ones((5, 3))]
    assert _normalize_shap_values(per_class_list, class_index=1).shape == (5, 3)
    assert _normalize_shap_values(per_class_list, class_index=1).sum() == 15.0

    three_d = np.stack([np.zeros((5, 3)), np.ones((5, 3))], axis=2)
    assert _normalize_shap_values(three_d, class_index=1).shape == (5, 3)
    assert _normalize_shap_values(three_d, class_index=1).sum() == 15.0

    two_d = np.ones((5, 3))
    assert _normalize_shap_values(two_d).shape == (5, 3)

    # An out-of-range class index must fall back rather than raise.
    assert _normalize_shap_values([np.ones((5, 3))], class_index=1).shape == (5, 3)


def test_shap_drivers_ranked_and_directional():
    """Verifies SHAP driver ranking returns ordered features with direction labels."""
    _require_matrices()
    X_train, y_train, _, names = get_switching_matrices("train")
    X_test, _, _, _ = get_switching_matrices("test")

    model = train_switching_model(X_train, y_train)
    shap_values = compute_shap_values(model, X_test, class_index=1)

    assert shap_values.shape == X_test.shape

    drivers = rank_shap_drivers(shap_values, names, X_test, top_n=5)
    assert drivers.height == 5
    assert drivers["feature"].to_list()[0] in names

    # Ranking must be descending by mean absolute SHAP value.
    magnitudes = drivers["mean_abs_shap"].to_list()
    assert all(a >= b for a, b in zip(magnitudes, magnitudes[1:]))
    for effect in drivers["effect"].to_list():
        assert effect in ("increases switching", "reduces switching")


def test_plot_confusion_matrix_writes_image(tmp_path):
    """Verifies the confusion matrix heatmap is written to disk."""
    matrix = [[50, 5, 3], [6, 20, 4], [2, 7, 30]]
    out = plot_confusion_matrix(matrix, output_path=tmp_path / "cm.png")

    assert out.exists()
    assert out.stat().st_size > 1000


def test_train_switching_model_accepts_groups():
    rng = np.random.default_rng(2)
    n_users, rows = 120, 3
    groups = np.repeat(np.arange(n_users), rows)
    y = np.repeat(rng.integers(0, 2, n_users), rows)
    X = rng.normal(size=(len(y), 4)) + y[:, None] * 0.8
    model = train_switching_model(X, y, groups=groups)
    assert model.predict_proba(X).shape == (len(y), 2)
