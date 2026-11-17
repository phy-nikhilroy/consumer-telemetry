"""
Baselines and bootstrap intervals for the binary models.

A model score means little without a floor to beat and an interval around it. This module
supplies a single-feature baseline, a clipped logistic baseline, a percentile bootstrap
confidence interval for any metric, and a paired bootstrap for the difference between two
score vectors on the same rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Sequence

# Ensure the project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import RANDOM_SEED, RECENCY_FEATURES

RECENCY_CLIP_DAYS = 60.0
DEFAULT_SINGLE_FEATURE = "total_purchases_30d"
Metric = Callable[[np.ndarray, np.ndarray], float]


def single_feature_scores(
    X: np.ndarray,
    feature_names: Sequence[str],
    feature: str = DEFAULT_SINGLE_FEATURE,
) -> np.ndarray:
    """Returns one feature column as a score vector (nulls become 0); raises if absent."""
    names = list(feature_names)
    if feature not in names:
        raise ValueError(f"Feature '{feature}' is not among the model features.")
    col = np.asarray(X, dtype=float)[:, names.index(feature)]
    return np.nan_to_num(col, nan=0.0)


def _clip_columns(X: np.ndarray, idx: List[int], upper: float) -> np.ndarray:
    """Returns a copy of X with the given columns clipped above at `upper`."""
    out = np.array(X, dtype=float, copy=True)
    if idx:
        out[:, idx] = np.clip(out[:, idx], None, upper)
    return out


def fit_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
) -> Pipeline:
    """Fits clip(recency at 60) -> StandardScaler -> LogisticRegression(max_iter=2000)."""
    names = list(feature_names)
    idx = [names.index(f) for f in RECENCY_FEATURES if f in names]
    pipe = Pipeline([
        ("clip", FunctionTransformer(
            _clip_columns, kw_args={"idx": idx, "upper": RECENCY_CLIP_DAYS}
        )),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def _safe_point(metric: Metric, y: np.ndarray, p: np.ndarray) -> float:
    """Metric on the full sample, nan when undefined (for example one class)."""
    try:
        return float(metric(y, p))
    except ValueError:
        return float("nan")


def bootstrap_metric_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: Metric = roc_auc_score,
    n_boot: int = 2000,
    seed: int = RANDOM_SEED,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile bootstrap CI for `metric(y, p)`; returns {"point", "lo", "hi"} (nan if undefined)."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    stats: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(float(metric(y[idx], p[idx])))
    point = _safe_point(metric, y, p)
    if not stats:
        return {"point": point, "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def paired_bootstrap_delta(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: Metric = roc_auc_score,
    n_boot: int = 2000,
    seed: int = RANDOM_SEED,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired bootstrap of metric(a) - metric(b) on shared resamples; {"delta", "lo", "hi"}."""
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    rng = np.random.default_rng(seed)
    stats: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(float(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx])))
    delta = _safe_point(metric, y, p_a) - _safe_point(metric, y, p_b)
    if not stats:
        return {"delta": delta, "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta": float(delta), "lo": float(lo), "hi": float(hi)}


def baseline_report(
    y_test: np.ndarray,
    model_proba: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Sequence[str],
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """
    Builds the `ci`, `baselines`, and delta blocks merged into the metrics JSON `test` entry.

    The single-feature baseline is null when its feature is not in the model features.
    """
    y_test = np.asarray(y_test)
    out: Dict[str, Any] = {
        "ci": {
            "roc_auc": bootstrap_metric_ci(y_test, model_proba, roc_auc_score, n_boot=n_boot),
            "pr_auc": bootstrap_metric_ci(y_test, model_proba, average_precision_score, n_boot=n_boot),
        },
        "baselines": {},
    }
    logit = fit_logistic_baseline(X_train, y_train, feature_names)
    logit_scores = logit.predict_proba(X_test)[:, 1]
    out["baselines"]["logistic"] = {
        "roc_auc": _safe_point(roc_auc_score, y_test, logit_scores),
        "pr_auc": _safe_point(average_precision_score, y_test, logit_scores),
    }
    out["delta_vs_logistic"] = paired_bootstrap_delta(
        y_test, model_proba, logit_scores, roc_auc_score, n_boot=n_boot
    )
    try:
        single = single_feature_scores(X_test, feature_names)
        out["baselines"]["single_feature"] = {
            "feature": DEFAULT_SINGLE_FEATURE,
            "roc_auc": _safe_point(roc_auc_score, y_test, single),
            "pr_auc": _safe_point(average_precision_score, y_test, single),
        }
        out["delta_vs_single_feature"] = paired_bootstrap_delta(
            y_test, model_proba, single, roc_auc_score, n_boot=n_boot
        )
    except ValueError:
        out["baselines"]["single_feature"] = None
        out["delta_vs_single_feature"] = None
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CI demo on synthetic scores.")
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args()
    rng = np.random.default_rng(RANDOM_SEED)
    y_demo = rng.integers(0, 2, 500)
    p_demo = y_demo * 0.5 + rng.random(500)
    print(bootstrap_metric_ci(y_demo, p_demo, roc_auc_score, n_boot=args.n_boot))
