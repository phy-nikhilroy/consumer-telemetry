"""Tests for the feature-block ablation (S6): block partition, model sets, evaluation, aggregation."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from src.config import (FEATURE_BLOCKS, PRIMARY_TARGET, SELECTED_FEATURES_PATH,
                        SWITCHING_CONDITIONAL_TARGET, SWITCHING_TARGET)
from src.experiments import ablation

BLOCK_NAMES = {"rfm", "funnel", "cross_app", "price", "category"}


def _candidates() -> list:
    return json.loads(SELECTED_FEATURES_PATH.read_text())["candidate_features"]


def test_every_candidate_in_exactly_one_block():
    flat = [f for fs in FEATURE_BLOCKS.values() for f in fs]
    assert len(flat) == len(set(flat))
    assert sorted(flat) == sorted(_candidates())
    for c in _candidates():
        assert sum(c in fs for fs in FEATURE_BLOCKS.values()) == 1


def test_block_names_and_cross_app_contents():
    assert set(FEATURE_BLOCKS) == BLOCK_NAMES
    assert set(FEATURE_BLOCKS["cross_app"]) == {
        "num_apps_used", "app_entropy", "dominant_app_share", "cross_app_comparison_count",
        "informed_purchase_count", "chose_cheaper_app_rate", "avg_abs_price_gap_pct"}


def test_model_feature_sets_nested():
    sets = ablation.model_feature_sets()
    assert set(sets["A"]) < set(sets["B"]) and set(sets["A"]) < set(sets["C"])
    assert set(sets["B"]).isdisjoint(FEATURE_BLOCKS["cross_app"])
    assert set(sets["C"]).isdisjoint(FEATURE_BLOCKS["funnel"])
    assert set(sets["D"]) == set(_candidates())


def test_d_without_cross_app_drops_block():
    d = ablation.model_feature_sets(include_cross_app_in_d=False)["D"]
    assert set(d).isdisjoint(FEATURE_BLOCKS["cross_app"])
    assert set(FEATURE_BLOCKS["price"]) <= set(d)


def _frame(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {f: rng.normal(size=n) for fs in FEATURE_BLOCKS.values() for f in fs}
    signal = cols["total_events_7d"] + 0.5 * rng.normal(size=n)
    buy = (signal > 0).astype(int)
    cols["user_id"] = [f"u{i % (n // 2)}" for i in range(n)]
    cols[PRIMARY_TARGET] = buy
    cols[SWITCHING_TARGET] = (cols["app_entropy"] + rng.normal(size=n) > 0).astype(int)
    return pl.DataFrame(cols)


@pytest.mark.parametrize("target", [PRIMARY_TARGET, SWITCHING_CONDITIONAL_TARGET])
def test_evaluate_target_structure(target):
    res = ablation.evaluate_target(_frame(600, 1), _frame(300, 2), target, ablation.model_feature_sets(), n_boot=30)
    assert set(res["models"]) == {"A", "B", "C", "D"}
    assert set(res["deltas"]) == {"B-A", "C-A", "D-A"}
    for m in res["models"].values():
        assert 0.0 <= m["roc_auc"] <= 1.0 and m["ci_lo"] <= m["ci_hi"]
    d = res["deltas"]["B-A"]
    assert d["lo"] <= d["hi"]


def test_conditional_target_uses_buyers_only():
    res = ablation.evaluate_target(_frame(600, 1), _frame(300, 2), SWITCHING_CONDITIONAL_TARGET,
                                   ablation.model_feature_sets(), n_boot=20)
    assert res["n_test"] < 300


def test_single_class_raises():
    tr, te = _frame(200, 1), _frame(100, 2).with_columns(pl.lit(0).alias(PRIMARY_TARGET))
    with pytest.raises(ValueError):
        ablation.evaluate_target(tr, te, PRIMARY_TARGET, ablation.model_feature_sets(), n_boot=10)


def test_aggregate_seeds():
    def one(a, d):
        r = {"n_train": 1, "n_test": 1, "test_positive_rate": 0.5,
             "models": {"A": {"roc_auc": a}, "B": {"roc_auc": a + d}},
             "deltas": {"B-A": {"delta": d, "lo": d - 0.01, "hi": d + 0.01}}}
        return {PRIMARY_TARGET: r, SWITCHING_CONDITIONAL_TARGET: r}
    agg = ablation.aggregate_seeds({"1": one(0.7, 0.05), "2": one(0.72, 0.03)})
    t = agg[PRIMARY_TARGET]
    assert t["roc_auc"]["A"]["n"] == 2 and t["roc_auc"]["A"]["mean"] == pytest.approx(0.71)
    assert t["deltas"]["B-A"]["mean"] == pytest.approx(0.04)
    assert t["deltas"]["B-A"]["seeds_ci_excludes_zero"] == 2


def test_caveat_mentions_simulated_mechanism():
    assert "simulated" in ablation.CAVEAT and "mechanism" in ablation.CAVEAT
