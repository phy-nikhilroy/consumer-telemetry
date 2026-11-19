"""Tests for the learning curve experiment: point summaries, curve assembly and defaults."""

from __future__ import annotations

import math
from pathlib import Path

from src.experiments import learning_curve as lc


def test_defaults_match_spec() -> None:
    assert lc.USER_COUNTS == (250, 500, 1000, 2000)
    assert lc.CURVE_DAYS == 60
    assert lc.RANDOM_SEED == 42
    assert "data/experiments" in str(lc.DEFAULT_OUT)


def test_summarize_point_reads_ci_forms() -> None:
    p = lc.summarize_point(250, {"roc_auc": 0.7, "n_samples": 10, "ci": {"roc_auc": [0.6, 0.8]}},
                           {"roc_auc": 0.6, "n_samples": 5, "ci": [0.5, 0.7]})
    assert p["propensity"]["ci"] == [0.6, 0.8]
    assert p["conditional_switching"]["ci"] == [0.5, 0.7]
    assert p["users"] == 250


def test_summarize_point_missing_ci_is_nan() -> None:
    p = lc.summarize_point(250, {"roc_auc": 0.7}, {"roc_auc": 0.6})
    assert all(math.isnan(v) for v in p["propensity"]["ci"])


def test_run_learning_curve_assembles_points(monkeypatch, tmp_path: Path) -> None:
    def fake(n, out_dir, days, seed):
        assert out_dir.parent == tmp_path and days == 60 and seed == 42
        return lc.summarize_point(n, {"roc_auc": 0.5 + n / 10000, "ci": {"roc_auc": [0, 1]}},
                                  {"roc_auc": 0.5, "ci": [0, 1]})
    monkeypatch.setattr(lc, "run_point", fake)
    r = lc.run_learning_curve(tmp_path)
    assert [p["users"] for p in r["points"]] == [250, 500, 1000, 2000]
    assert abs(r["auc_gain_first_to_last"]["propensity"] - 0.175) < 1e-9
    assert r["auc_gain_first_to_last"]["conditional_switching"] == 0.0
