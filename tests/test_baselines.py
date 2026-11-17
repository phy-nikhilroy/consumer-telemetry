"""Tests for src/models/baselines.py: baselines and bootstrap intervals."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.models.baselines import (
    baseline_report,
    bootstrap_metric_ci,
    fit_logistic_baseline,
    paired_bootstrap_delta,
    single_feature_scores,
)


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = 0.4 * y + rng.random(300)
    return y, p


def test_perfect_scores_point_is_one():
    y = np.array([0, 0, 1, 1] * 10)
    ci = bootstrap_metric_ci(y, y.astype(float), roc_auc_score, n_boot=100)
    assert ci["point"] == 1.0


def test_ci_brackets_point(data):
    y, p = data
    ci = bootstrap_metric_ci(y, p, roc_auc_score, n_boot=200)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_identical_scores_zero_delta(data):
    y, p = data
    d = paired_bootstrap_delta(y, p, p.copy(), roc_auc_score, n_boot=100)
    assert d["delta"] == 0.0 and d["lo"] == 0.0 and d["hi"] == 0.0


def test_same_seed_identical_output(data):
    y, p = data
    assert bootstrap_metric_ci(y, p, roc_auc_score, n_boot=100, seed=3) == bootstrap_metric_ci(
        y, p, roc_auc_score, n_boot=100, seed=3
    )
    rng = np.random.default_rng(1)
    q = rng.random(len(y))
    assert paired_bootstrap_delta(y, p, q, roc_auc_score, n_boot=100, seed=3) == paired_bootstrap_delta(
        y, p, q, roc_auc_score, n_boot=100, seed=3
    )


def test_better_p_a_gives_positive_lower_bound():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    p_a = y + 0.3 * rng.random(500)
    p_b = rng.random(500)
    d = paired_bootstrap_delta(y, p_a, p_b, roc_auc_score, n_boot=200)
    assert d["lo"] > 0 and d["delta"] > 0


def test_single_class_is_nan():
    ci = bootstrap_metric_ci(np.ones(10), np.arange(10.0), roc_auc_score, n_boot=5)
    assert np.isnan(ci["lo"]) and np.isnan(ci["hi"])


def test_single_feature_scores_selects_column_and_rejects_unknown():
    X = np.array([[1.0, 2.0], [3.0, np.nan]])
    out = single_feature_scores(X, ["a", "total_purchases_30d"])
    assert out.tolist() == [2.0, 0.0]
    with pytest.raises(ValueError):
        single_feature_scores(X, ["a", "b"])


def test_logistic_baseline_clips_recency():
    rng = np.random.default_rng(4)
    n = 200
    y = rng.integers(0, 2, n)
    rec = np.where(rng.random(n) < 0.3, 9999.0, rng.random(n) * 30)
    X = np.column_stack([y + rng.normal(size=n), rec])
    names = ["x", "days_since_last_event"]
    model = fit_logistic_baseline(X, y, names)
    X2 = X.copy()
    X2[:, 1] = np.where(X2[:, 1] > 60, 60.0, X2[:, 1])
    assert np.allclose(model.predict_proba(X)[:, 1], model.predict_proba(X2)[:, 1])
    assert roc_auc_score(y, model.predict_proba(X)[:, 1]) > 0.7


def test_baseline_report_keys(data):
    y, p = data
    rng = np.random.default_rng(5)
    X = rng.random((len(y), 2))
    rep = baseline_report(y, p, X, y, X, ["total_purchases_30d", "z"], n_boot=50)
    assert set(rep["ci"]) == {"roc_auc", "pr_auc"}
    assert set(rep["baselines"]) == {"single_feature", "logistic"}
    assert {"delta", "lo", "hi"} <= set(rep["delta_vs_logistic"])
    rep2 = baseline_report(y, p, X, y, X, ["q", "z"], n_boot=50)
    assert rep2["baselines"]["single_feature"] is None
