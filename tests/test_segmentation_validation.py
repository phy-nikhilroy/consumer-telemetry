"""
Tests for E8: segmentation log transform, archetype validation, bootstrap stability, GMM BIC.

Everything runs on synthetic three-blob data written to a temporary features directory, so
no committed artifact or data/processed file is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.segmentation import (
    SEGMENTATION_FEATURES,
    SEGMENTATION_LOG_FEATURES,
    run_segmentation_pipeline,
    scale_features,
)

N_PER_BLOB = 40
LOG_SET = ("total_events_30d", "total_purchases_30d", "total_spend_30d",
           "avg_order_value", "cross_app_comparison_count")


def _blob_frame(seed: int = 0) -> tuple[pl.DataFrame, list[str]]:
    """Three well separated blobs in log space; returns the frame and the true blob names."""
    rng = np.random.default_rng(seed)
    centres = {
        "alpha": dict(zip(SEGMENTATION_FEATURES, [1.0, 0.5, 2.0, 2.0, 0.3, 0.9, 0.2, 0.2, 0.2, 5.0, 0.2, 5.0])),
        "beta": dict(zip(SEGMENTATION_FEATURES, [3.0, 2.0, 7.0, 6.0, 1.2, 0.5, 2.0, 0.6, 0.6, 15.0, 0.6, 15.0])),
        "gamma": dict(zip(SEGMENTATION_FEATURES, [5.0, 3.5, 11.0, 9.0, 0.6, 0.2, 4.0, 0.9, 0.9, 30.0, 0.9, 30.0])),
    }
    rows, truth = [], []
    for name, centre in centres.items():
        for i in range(N_PER_BLOB):
            row = {"user_id": f"{name}_{i}"}
            for feat in SEGMENTATION_FEATURES:
                value = centre[feat] + rng.normal(0, 0.05 * max(abs(centre[feat]), 1.0))
                row[feat] = float(np.expm1(value)) if feat in LOG_SET else float(value)
            rows.append(row)
            truth.append(name)
    return pl.DataFrame(rows), truth


def _write(features_dir: Path, with_archetype: bool, permute: bool = False) -> None:
    frame, truth = _blob_frame()
    features_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(features_dir / "train_features.parquet")
    if with_archetype:
        rename = {"alpha": "z1", "beta": "a2", "gamma": "m3"} if permute else {k: k for k in set(truth)}
        pl.DataFrame({
            "user_id": frame["user_id"],
            "predicted_clv": [1.0] * frame.height,
            "archetype": [rename[t] for t in truth],
        }).write_parquet(features_dir / "user_clv.parquet")


@pytest.fixture(scope="module")
def run_with_archetype(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("seg_arch")
    _write(root / "features", with_archetype=True, permute=True)
    result = run_segmentation_pipeline(
        features_dir=root / "features", artifacts_dir=root / "artifacts"
    )
    return json.loads(Path(result["paths"]["summary"]).read_text())


@pytest.fixture(scope="module")
def run_without_archetype(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("seg_noarch")
    _write(root / "features", with_archetype=False)
    result = run_segmentation_pipeline(
        features_dir=root / "features", artifacts_dir=root / "artifacts"
    )
    return json.loads(Path(result["paths"]["summary"]).read_text())


def test_segmentation_log_features_exact():
    assert tuple(SEGMENTATION_LOG_FEATURES) == LOG_SET


def test_log_features_are_clustering_features():
    assert set(SEGMENTATION_LOG_FEATURES) <= set(SEGMENTATION_FEATURES)


def test_scale_features_applies_log1p_before_standardizing():
    frame, _ = _blob_frame()
    X, names, scaler = scale_features(frame)
    for feat in SEGMENTATION_LOG_FEATURES:
        i = names.index(feat)
        expected = np.log1p(frame[feat].to_numpy())
        assert scaler.mean_[i] == pytest.approx(expected.mean(), rel=1e-9)
        assert scaler.scale_[i] == pytest.approx(expected.std(), rel=1e-9)
        assert X[:, i].mean() == pytest.approx(0.0, abs=1e-9)
    # A column outside the log set keeps its raw mean.
    j = names.index("app_entropy")
    assert scaler.mean_[j] == pytest.approx(frame["app_entropy"].mean(), rel=1e-9)


def test_scale_features_log_compresses_outlier():
    frame, _ = _blob_frame()
    spiked = frame.with_columns(
        pl.when(pl.col("user_id") == "alpha_0").then(1e7).otherwise(pl.col("total_spend_30d"))
        .alias("total_spend_30d")
    )
    X, names, _ = scale_features(spiked)
    col = X[:, names.index("total_spend_30d")]
    # A raw standardization of a 1e7 spike among 120 rows would sit near 10.9 sd.
    assert np.abs(col).max() < 8


def test_scale_features_zero_is_log_safe():
    frame, _ = _blob_frame()
    zeroed = frame.with_columns(
        pl.when(pl.col("user_id") == "alpha_0").then(0.0).otherwise(pl.col("cross_app_comparison_count"))
        .alias("cross_app_comparison_count")
    )
    X, _, _ = scale_features(zeroed)
    assert np.all(np.isfinite(X))


def test_summary_persists_log_feature_list(run_with_archetype):
    assert list(SEGMENTATION_LOG_FEATURES) in run_with_archetype.values()


def test_summary_has_required_keys(run_with_archetype):
    for key in ("ari_vs_archetype", "bootstrap_stability", "gmm_bic_by_k", "gmm_best_k"):
        assert key in run_with_archetype


def test_ari_vs_archetype_is_one_for_identical_partition(run_with_archetype):
    """Archetype names are permuted relative to blobs; ARI ignores names, so a perfect partition scores 1."""
    assert run_with_archetype["ari_vs_archetype"] == pytest.approx(1.0)


def test_ari_absent_without_archetypes(run_without_archetype):
    assert run_without_archetype.get("ari_vs_archetype") is None
    assert "bootstrap_stability" in run_without_archetype


def test_bootstrap_stability_in_unit_interval_and_high_on_blobs(run_with_archetype):
    stability = run_with_archetype["bootstrap_stability"]
    assert 0.0 <= stability <= 1.0
    assert stability > 0.9


def test_gmm_bic_covers_k_3_to_8(run_with_archetype):
    bic = run_with_archetype["gmm_bic_by_k"]
    assert sorted(int(k) for k in bic) == [3, 4, 5, 6, 7, 8]
    assert all(np.isfinite(v) for v in bic.values())


def test_gmm_best_k_is_argmin_bic_and_three_on_blobs(run_with_archetype):
    bic = {int(k): v for k, v in run_with_archetype["gmm_bic_by_k"].items()}
    best = run_with_archetype["gmm_best_k"]
    assert best == min(bic, key=bic.get)
    assert best == 3


def test_segmentation_summary_is_deterministic(tmp_path, run_with_archetype):
    _write(tmp_path / "features", with_archetype=True, permute=True)
    result = run_segmentation_pipeline(features_dir=tmp_path / "features", artifacts_dir=tmp_path / "artifacts")
    again = json.loads(Path(result["paths"]["summary"]).read_text())
    assert again["bootstrap_stability"] == run_with_archetype["bootstrap_stability"]
    assert again["gmm_bic_by_k"] == run_with_archetype["gmm_bic_by_k"]


def _blob_matrix() -> tuple[np.ndarray, np.ndarray]:
    from src.models.segmentation import fit_kmeans

    X, _, _ = scale_features(_blob_frame()[0])
    _, labels = fit_kmeans(X, 3)
    return X, labels


def test_bootstrap_stability_direct_high_on_blobs():
    from src.models.segmentation import bootstrap_stability

    X, labels = _blob_matrix()
    value = bootstrap_stability(X, labels, 3)
    assert 0.9 < value <= 1.0


def test_bootstrap_stability_low_for_structureless_data():
    """Uniform noise has no stable partition, so ARI against the full-data labels must fall well below 1."""
    from src.models.segmentation import bootstrap_stability, fit_kmeans

    X = np.random.default_rng(3).uniform(size=(150, 4))
    _, labels = fit_kmeans(X, 6)
    value = bootstrap_stability(X, labels, 6)
    assert 0.0 <= value < 0.9


def test_bootstrap_stability_compares_against_given_labels():
    """Scrambled reference labels share no structure with any bootstrap model, so stability is near 0."""
    from src.models.segmentation import bootstrap_stability

    X, labels = _blob_matrix()
    scrambled = np.random.default_rng(1).permutation(labels)
    assert bootstrap_stability(X, scrambled, 3) < 0.3
