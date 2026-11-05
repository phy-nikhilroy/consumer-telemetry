"""
Unit tests for KMeans and HDBSCAN user segmentation.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.segmentation import (
    DEFAULT_K_VALUES,
    MIN_SIGNATURE_SIMILARITY,
    SEGMENT_SIGNATURES,
    SEGMENTATION_FEATURES,
    compute_cluster_zscores,
    evaluate_k_range,
    fit_hdbscan,
    fit_kmeans,
    load_segmentation_data,
    name_clusters,
    plot_cluster_profiles,
    plot_elbow_and_silhouette,
    plot_pca_scatter,
    profile_clusters,
    scale_features,
    score_clustering,
    select_best_k,
)

TRAIN_PATH = Path("data/features/train_features.parquet")


def _require_matrices():
    """Skips a test when the feature store has not been generated."""
    if not TRAIN_PATH.exists():
        pytest.skip("Feature matrix not available")


@pytest.fixture(scope="module")
def prepared():
    """Loads and scales the segmentation features once."""
    _require_matrices()
    df = load_segmentation_data()
    X, names, scaler = scale_features(df)
    return df, X, names, scaler


def test_load_segmentation_data_excludes_forward_targets(prepared):
    """Verifies no forward target leaks into the clustering features."""
    df, _, _, _ = prepared

    for target in ("will_purchase_next_7d", "num_purchases_next_7d", "will_switch_app_next_7d"):
        assert target not in df.columns

    assert "user_id" in df.columns
    assert df.height > 0


def test_scale_features_standardizes(prepared):
    """Verifies features are centered and unit scaled before distance clustering."""
    _, X, names, _ = prepared

    assert X.shape[1] == len(names)
    assert set(names).issubset(set(SEGMENTATION_FEATURES))
    np.testing.assert_allclose(X.mean(axis=0), np.zeros(X.shape[1]), atol=1e-9)
    np.testing.assert_allclose(X.std(axis=0), np.ones(X.shape[1]), atol=1e-9)
    assert np.isfinite(X).all()


def test_evaluate_k_range_scores_every_candidate(prepared):
    """Verifies each candidate k is scored and inertia falls monotonically."""
    _, X, _, _ = prepared
    scores = evaluate_k_range(X, k_values=DEFAULT_K_VALUES)

    assert scores.height == len(DEFAULT_K_VALUES)
    for col in ("k", "inertia", "silhouette", "davies_bouldin", "calinski_harabasz"):
        assert col in scores.columns

    # Inertia must decrease as clusters are added, which is why it alone cannot pick k.
    inertia = scores.sort("k")["inertia"].to_list()
    assert all(a > b for a, b in zip(inertia, inertia[1:]))

    # Silhouette is bounded and must indicate better-than-random structure.
    for value in scores["silhouette"].to_list():
        assert -1.0 <= value <= 1.0
        assert value > 0.0


def test_select_best_k_picks_highest_silhouette():
    """Verifies k selection uses silhouette rather than inertia."""
    scores = pl.DataFrame(
        {
            "k": [3, 4, 5, 6],
            "inertia": [3000.0, 2500.0, 2200.0, 2000.0],
            "silhouette": [0.28, 0.26, 0.31, 0.29],
            "davies_bouldin": [1.4, 1.5, 1.3, 1.35],
            "calinski_harabasz": [140.0, 120.0, 118.0, 108.0],
        }
    )
    assert select_best_k(scores) == 5


def test_kmeans_assigns_every_user(prepared):
    """Verifies KMeans produces a complete, valid partition."""
    _, X, _, _ = prepared
    model, labels = fit_kmeans(X, n_clusters=5)

    assert len(labels) == X.shape[0]
    assert set(labels.tolist()) == set(range(5))
    # KMeans has no noise label; every user must belong to a cluster.
    assert (labels >= 0).all()
    assert model.cluster_centers_.shape == (5, X.shape[1])


def test_hdbscan_finds_clusters_in_reduced_space(prepared):
    """
    Verifies HDBSCAN recovers clusters once dimensionality is reduced.

    Run on the full twelve standardized dimensions this labelled every user as noise,
    because 400 points are too sparse there for any region to clear the density
    threshold.
    """
    _, X, _, _ = prepared

    _, reduced_labels = fit_hdbscan(X, min_cluster_size=15, n_components=5)
    quality = score_clustering(X, reduced_labels)

    assert quality["n_clusters"] >= 2
    assert quality["noise_fraction"] < 1.0
    assert quality["silhouette"] is not None

    # Noise points carry label -1 and must not be counted as a cluster.
    assert -1 not in [int(c) for c in quality["cluster_sizes"]]

    _, full_labels = fit_hdbscan(X, min_cluster_size=15, n_components=None)
    full_quality = score_clustering(X, full_labels)
    assert quality["n_clusters"] >= full_quality["n_clusters"]


def test_score_clustering_handles_degenerate_input():
    """Verifies scoring returns None rather than raising for a single cluster."""
    X = np.random.default_rng(0).normal(size=(30, 4))

    single = score_clustering(X, np.zeros(30, dtype=int))
    assert single["n_clusters"] == 1
    assert single["silhouette"] is None

    all_noise = score_clustering(X, np.full(30, -1, dtype=int))
    assert all_noise["n_clusters"] == 0
    assert all_noise["n_noise"] == 30
    assert all_noise["noise_fraction"] == pytest.approx(1.0)
    assert all_noise["silhouette"] is None


def test_profile_clusters_accounts_for_all_users(prepared):
    """Verifies cluster profiles cover the whole population exactly once."""
    df, X, names, _ = prepared
    _, labels = fit_kmeans(X, n_clusters=5)

    profile = profile_clusters(df, labels, names)
    assert profile.height == 5
    assert profile["users"].sum() == df.height
    for name in names:
        assert name in profile.columns


def test_cluster_zscores_are_centered(prepared):
    """Verifies z-scores express deviation from the cross-cluster mean."""
    df, X, names, _ = prepared
    _, labels = fit_kmeans(X, n_clusters=5)

    zscores = compute_cluster_zscores(profile_clusters(df, labels, names), names)
    assert zscores.height == 5
    for name in names:
        values = np.array(zscores[name].to_list(), dtype=float)
        assert abs(float(values.mean())) < 1e-9


def test_name_clusters_rejects_poor_matches():
    """
    Verifies a cluster matching no signature is left unclassified.

    Greedy assignment without this guard handed the final unclaimed name to whichever
    cluster was left over, which once labelled a low-entropy single-app cluster
    "Comparison Shoppers", the exact inverse of its behavior.
    """
    names = list(SEGMENTATION_FEATURES)
    # Cluster 0 is an unambiguous comparison shopper; cluster 1 is flat on every axis
    # and therefore resembles nothing.
    rows = []
    for cluster, entropy in ((0, 2.0), (1, 0.0)):
        row = {"cluster": cluster, "users": 50}
        for name in names:
            row[name] = 0.0
        row["app_entropy"] = entropy
        row["cross_app_comparison_count"] = entropy
        row["dominant_app_share"] = -entropy
        rows.append(row)

    assigned = name_clusters(pl.DataFrame(rows), names)

    assert assigned[0] == "Comparison Shoppers"
    assert assigned[1].startswith("Unclassified Segment")


def test_name_clusters_normalizes_signature_length():
    """
    Verifies signature matching is scale free across signatures of different size.

    A raw dot product lets a signature listing more features win on term count alone,
    regardless of how well it actually fits.
    """
    names = list(SEGMENTATION_FEATURES)
    row = {"cluster": 0, "users": 40}
    for name in names:
        row[name] = 0.0
    # A pure comparison-shopper profile.
    row["app_entropy"] = 1.5
    row["cross_app_comparison_count"] = 1.5
    row["dominant_app_share"] = -1.5

    assigned = name_clusters(pl.DataFrame([row]), names)
    assert assigned[0] == "Comparison Shoppers"

    # Every configured signature must be reachable in principle.
    assert "Comparison Shoppers" in SEGMENT_SIGNATURES
    assert 0.0 < MIN_SIGNATURE_SIMILARITY < 1.0


def test_segmentation_plots_are_written(prepared, tmp_path):
    """Verifies all three segmentation figures are saved."""
    df, X, names, _ = prepared
    _, labels = fit_kmeans(X, n_clusters=4)
    _, hdb_labels = fit_hdbscan(X, min_cluster_size=15, n_components=5)

    zscores = compute_cluster_zscores(profile_clusters(df, labels, names), names)
    cluster_names = name_clusters(zscores, names)

    elbow = plot_elbow_and_silhouette(
        evaluate_k_range(X, k_values=(3, 4)), 4, output_path=tmp_path / "elbow.png"
    )
    pca, explained = plot_pca_scatter(
        X, labels, hdb_labels, cluster_names, output_path=tmp_path / "pca.png"
    )
    heatmap = plot_cluster_profiles(
        zscores, names, cluster_names, output_path=tmp_path / "profiles.png"
    )

    for path in (elbow, pca, heatmap):
        assert path.exists()
        assert path.stat().st_size > 1000

    # Two components cannot retain more than all the variance.
    assert 0.0 < explained <= 1.0
