"""
Unit tests for KMeans and HDBSCAN user segmentation.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.segmentation import (
    DEFAULT_K_VALUES,
    MIN_SIGNATURE_COVERAGE,
    MIN_SIGNATURE_SIMILARITY,
    SEGMENT_SIGNATURES,
    SEGMENTATION_FEATURES,
    compute_cluster_zscores,
    eligible_signatures,
    evaluate_k_range,
    fit_hdbscan,
    fit_kmeans,
    k_at_range_boundary,
    load_segmentation_data,
    name_clusters,
    plot_cluster_profiles,
    plot_elbow_and_silhouette,
    plot_pca_scatter,
    profile_clusters,
    scale_features,
    score_clustering,
    select_best_k,
    signature_coverage,
    split_live_features,
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


def test_cluster_zscores_are_centered_on_the_user_population(prepared):
    """
    Verifies cluster z-scores are deviations from the user population, so their
    user-weighted mean is zero for every feature.
    """
    df, X, names, _ = prepared
    _, labels = fit_kmeans(X, n_clusters=5)

    zscores = compute_cluster_zscores(X, labels, names)
    assert zscores.height == 5
    assert zscores["users"].sum() == df.height
    weights = np.array(zscores["users"].to_list(), dtype=float)
    for name in names:
        values = np.array(zscores[name].to_list(), dtype=float)
        assert abs(float(np.average(values, weights=weights))) < 1e-9


def test_tiny_extreme_cluster_does_not_invert_its_neighbours():
    """
    Verifies a heavy-buyer cluster reads as above average even beside a tiny cluster of
    far more extreme buyers.

    Equal-weighted z-scores across cluster means let a 6-user cluster averaging 117
    purchases drag the reference point up, so a cluster of heavy buyers scored below
    average and was named "Window Shoppers" on real data.
    """
    rng = np.random.default_rng(0)
    names = ["total_events_30d", "total_purchases_30d", "total_spend_30d"]
    bulk = rng.normal([5, 0.4, 100], [2, 0.3, 50], size=(500, 3))
    heavy = rng.normal([460, 19, 4000], [20, 2, 300], size=(22, 3))
    extreme = rng.normal([1300, 117, 30000], [50, 5, 1000], size=(6, 3))
    raw = np.vstack([bulk, heavy, extreme])
    labels = np.array([0] * 500 + [1] * 22 + [2] * 6)

    X = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    zscores = compute_cluster_zscores(X, labels, names)
    heavy_row = zscores.filter(pl.col("cluster") == 1).row(0, named=True)

    assert heavy_row["total_purchases_30d"] > 0
    assert heavy_row["total_spend_30d"] > 0

    assigned = name_clusters(zscores, names)
    assert assigned[1] != "Window Shoppers"


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

    zscores = compute_cluster_zscores(X, labels, names)
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


def test_split_live_features_drops_constant_columns():
    """Verifies features with no variance are separated out before clustering."""
    df = pl.DataFrame(
        {
            "user_id": ["a", "b", "c", "d"],
            "total_events_30d": [1, 5, 9, 2],
            "app_entropy": [0.0, 0.0, 0.0, 0.0],
            "avg_discount_pct_on_purchases": [None, None, None, None],
        }
    )
    live, dead = split_live_features(
        df, ["total_events_30d", "app_entropy", "avg_discount_pct_on_purchases"]
    )
    assert live == ["total_events_30d"]
    assert set(dead) == {"app_entropy", "avg_discount_pct_on_purchases"}


def test_signature_needs_its_defining_features_to_be_named():
    """
    Verifies a signature whose defining features are dead cannot be assigned.

    With both discount features constant, "Bargain Hunters" reduced to its single
    leftover term, low average order value, and matched any cheap cluster perfectly.
    """
    live = [
        name for name in SEGMENTATION_FEATURES
        if name not in {"avg_discount_pct_on_purchases", "pct_purchases_with_discount",
                        "app_entropy", "dominant_app_share", "cross_app_comparison_count"}
    ]
    coverage = eligible_signatures(live)
    assert coverage["Bargain Hunters"] < MIN_SIGNATURE_COVERAGE
    assert coverage["Single-App Regulars"] < MIN_SIGNATURE_COVERAGE
    assert coverage["Comparison Shoppers"] < MIN_SIGNATURE_COVERAGE
    assert coverage["Window Shoppers"] == pytest.approx(1.0)

    row = {"cluster": 0, "users": 100}
    for name in live:
        row[name] = 0.0
    row["avg_order_value"] = -2.0
    assigned = name_clusters(pl.DataFrame([row]), live)
    assert assigned[0] != "Bargain Hunters"


def test_signature_coverage_bounds():
    """Verifies coverage is one with every feature present and zero with none."""
    weights = {"a": 1.0, "b": 1.0}
    assert signature_coverage(weights, ["a", "b"]) == pytest.approx(1.0)
    assert signature_coverage(weights, []) == 0.0
    assert signature_coverage(weights, ["a"]) == pytest.approx(1 / np.sqrt(2))


def test_k_at_range_boundary_flags_unresolved_selection():
    """Verifies a k chosen at the top of the search range is reported as such."""
    scores = pl.DataFrame({"k": [3, 4, 5, 6], "silhouette": [0.73, 0.74, 0.75, 0.76]})
    assert k_at_range_boundary(scores, select_best_k(scores)) is True
    peaked = pl.DataFrame({"k": [3, 4, 5, 6], "silhouette": [0.2, 0.3, 0.25, 0.22]})
    assert k_at_range_boundary(peaked, select_best_k(peaked)) is False


def test_never_purchasers_cluster_as_lapsed_not_as_recent():
    """
    Verifies a user who never purchased is scaled as the most lapsed user, not as one
    who purchased today, and without the tree-model sentinel's extreme scale.
    """
    df = pl.DataFrame(
        {
            "user_id": ["a", "b", "c", "d"],
            "days_since_last_purchase": [None, 0.5, 3.0, 9.0],
            "total_events_30d": [10, 20, 30, 40],
        }
    )
    X, names, _ = scale_features(df, ["days_since_last_purchase", "total_events_30d"])
    recency = X[:, names.index("days_since_last_purchase")]
    assert recency[0] == pytest.approx(recency[3])
    assert recency[0] > recency[1]
    assert float(np.abs(X).max()) < 3.0
