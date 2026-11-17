"""
Unsupervised user segmentation with KMeans and HDBSCAN.

Segmentation answers a different question from the supervised models: rather than
predicting a known outcome, it asks what naturally distinct kinds of shopper exist in
the population. The number of clusters is chosen with the elbow method and silhouette
score rather than assumed, and HDBSCAN is run alongside KMeans because it finds
density-based clusters without being told how many to look for, and can label genuine
outliers as noise instead of forcing them into a group.

Clusters are profiled on interpretable behavioral features and then named by scoring
each cluster against known shopper signatures, so the output is a readable segment
rather than an anonymous cluster index.

Naming is restricted to the feature space the data actually populates. Features with no
variance are dropped before clustering, and a signature is only eligible when most of
its weight sits on surviving features. On single-platform data, where every cross-app
feature is constant, the unrestricted signatures produced labels that were the exact
inverse of the clusters' behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from src.config import (
    ARTIFACTS_DIR,
    RANDOM_SEED,
    SEGMENTATION_MODEL_PATH,
    TRAIN_FEATURES_PATH,
    USER_CLV_PATH,
    USER_SEGMENTS_PATH,
    rebase_path,
)
from src.features.feature_store import censored_recency_fill, fill_missing_features

ELBOW_PLOT_PATH = ARTIFACTS_DIR / "segmentation_elbow_silhouette.png"
PCA_PLOT_PATH = ARTIFACTS_DIR / "segmentation_pca_scatter.png"
PROFILE_PLOT_PATH = ARTIFACTS_DIR / "segmentation_cluster_profiles.png"
SEGMENTATION_SUMMARY_PATH = ARTIFACTS_DIR / "segmentation_summary.json"
SEGMENTATION_SCALER_PATH = ARTIFACTS_DIR / "segmentation_scaler.joblib"

# Interpretable behavioral features used for clustering and profiling. These are chosen
# for readability of the resulting segments rather than for predictive power.
SEGMENTATION_FEATURES: Tuple[str, ...] = (
    "total_events_30d",
    "total_purchases_30d",
    "total_spend_30d",
    "avg_order_value",
    "app_entropy",
    "dominant_app_share",
    "cross_app_comparison_count",
    "view_to_cart_ratio",
    "cart_to_purchase_ratio",
    "avg_discount_pct_on_purchases",
    "pct_purchases_with_discount",
    "days_since_last_purchase",
)

DEFAULT_K_VALUES: Tuple[int, ...] = (3, 4, 5, 6)

# Segment signatures, expressed as the direction each feature should deviate from the
# population mean. Clusters are named by whichever signature their z-score profile
# matches most closely.
SEGMENT_SIGNATURES: Dict[str, Dict[str, float]] = {
    "VIP Loyalists": {
        "total_spend_30d": 1.0,
        "total_purchases_30d": 1.0,
        "avg_order_value": 0.6,
        "app_entropy": -0.8,
        "dominant_app_share": 1.0,
    },
    "Comparison Shoppers": {
        "app_entropy": 1.0,
        "cross_app_comparison_count": 1.0,
        "dominant_app_share": -1.0,
        "total_purchases_30d": -0.3,
    },
    "Bargain Hunters": {
        "avg_discount_pct_on_purchases": 1.0,
        "pct_purchases_with_discount": 1.0,
        "avg_order_value": -0.5,
    },
    "Window Shoppers": {
        "total_events_30d": 1.0,
        "total_purchases_30d": -1.0,
        "total_spend_30d": -1.0,
        "view_to_cart_ratio": -0.6,
        "days_since_last_purchase": 1.0,
    },
    "Single-App Regulars": {
        "dominant_app_share": 1.0,
        "app_entropy": -1.0,
        "cross_app_comparison_count": -0.7,
        "total_purchases_30d": 0.3,
    },
    "Occasional Buyers": {
        "total_events_30d": -0.8,
        "total_purchases_30d": -0.3,
        "total_spend_30d": -0.3,
    },
}

# A cluster must resemble a signature at least this strongly (cosine similarity) to
# inherit its name. Greedy assignment would otherwise hand the last unclaimed name to
# whichever cluster was left over, which once labelled a low-entropy single-app cluster
# "Comparison Shoppers". The original floor of 0.15 let a 6-user cluster of extreme
# buyers match "Single-App Regulars" on real data, so it was raised.
MIN_SIGNATURE_SIMILARITY = 0.5

# A signature is only eligible when the features it can still be evaluated on carry at
# least this share of its total weight (by vector norm). Without it, a signature whose
# defining features are all dead degenerates to one leftover term: "Bargain Hunters"
# with both discount features constant reduced to "low average order value", and then
# matched any cheap cluster with perfect cosine similarity.
MIN_SIGNATURE_COVERAGE = 0.75

# Features whose standard deviation across users is at or below this are treated as dead.
DEAD_FEATURE_STD = 1e-9


def load_segmentation_data(
    features_path: Optional[Union[str, Path]] = None,
    clv_path: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    """
    Loads the user feature matrix for clustering, joining CLV when it is available.

    Only historical behavior is used. The forward targets are deliberately excluded:
    segmentation describes who users are, and mixing in future outcomes would make the
    segments unusable for acting on a user before that outcome is known.
    """
    path = Path(features_path) if features_path else TRAIN_FEATURES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found at {path}")

    df = pl.read_parquet(path)
    # The train matrix stacks one block per train reference date, so a user appears once
    # per date. Segmentation needs one row per user describing who they are now, so it
    # keeps only the latest reference date: the most recent history available in the file.
    if "reference_date" in df.columns:
        latest = df["reference_date"].max()
        df = df.filter(pl.col("reference_date") == latest)
    available = [c for c in SEGMENTATION_FEATURES if c in df.columns]
    selected = df.select(["user_id"] + available)

    clv_file = Path(clv_path) if clv_path else USER_CLV_PATH
    if clv_file.exists():
        clv = pl.read_parquet(clv_file).select(
            [c for c in ("user_id", "predicted_clv", "archetype") if c in
             pl.read_parquet_schema(clv_file)]
        )
        selected = selected.join(clv, on="user_id", how="left")

    return selected


def scale_features(
    df: pl.DataFrame,
    feature_names: Optional[List[str]] = None,
    recency_fill: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """
    Standardizes the clustering features to zero mean and unit variance.

    Distance-based clustering is scale sensitive: total spend runs into the hundreds of
    thousands of rupees while app entropy is bounded by 2, so without standardization
    spend alone would dictate every cluster boundary.

    A missing recency (the user never purchased, or never added to cart) is censored at
    the longest recency observed, or at `recency_fill` when supplied. Zero filling put
    never-purchasers beside users who purchased today; the tree models' 9999 sentinel is
    right for trees but collapses the standardized column into a binary flag that cut
    the synthetic segmentation from five recovered archetypes to three.
    """
    names = feature_names or [c for c in SEGMENTATION_FEATURES if c in df.columns]
    fills = recency_fill if recency_fill is not None else censored_recency_fill(df, names)
    raw = fill_missing_features(df, names, recency_fill=fills).to_numpy().astype(np.float64)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    return scaled, names, scaler


def split_live_features(
    df: pl.DataFrame,
    feature_names: Optional[List[str]] = None,
    min_std: float = DEAD_FEATURE_STD,
) -> Tuple[List[str], List[str]]:
    """
    Separates features that vary across users from those that are constant.

    A constant column carries no information, makes `StandardScaler` divide by zero, and
    silently distorts signature matching, so it is excluded from clustering and naming.
    Returns `(live, dead)`.
    """
    names = feature_names or [c for c in SEGMENTATION_FEATURES if c in df.columns]
    filled = fill_missing_features(df, names, recency_fill=censored_recency_fill(df, names))
    live: List[str] = []
    dead: List[str] = []
    for name in names:
        std = filled[name].std()
        if std is None or not np.isfinite(std) or std <= min_std:
            dead.append(name)
        else:
            live.append(name)
    return live, dead


def evaluate_k_range(
    X: np.ndarray,
    k_values: Tuple[int, ...] = DEFAULT_K_VALUES,
    random_state: int = RANDOM_SEED,
) -> pl.DataFrame:
    """
    Fits KMeans across candidate k values and scores each one.

    Inertia always falls as k rises, so it is only useful for spotting an elbow.
    Silhouette (higher is better) and Davies-Bouldin (lower is better) actually
    penalize over-splitting, so they are what the choice of k is based on.
    """
    rows: List[Dict[str, Any]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X, labels)),
                "davies_bouldin": float(davies_bouldin_score(X, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            }
        )
    return pl.DataFrame(rows)


def select_best_k(scores: pl.DataFrame) -> int:
    """Returns the k with the highest silhouette score."""
    best_row = scores.sort("silhouette", descending=True).head(1)
    return int(best_row["k"][0])


def k_at_range_boundary(scores: pl.DataFrame, best_k: int) -> bool:
    """
    Returns whether the chosen k is the largest candidate tried.

    When silhouette is still rising at the top of the range, the range did not contain
    an optimum and the selection is an artifact of where the search stopped. On real data
    this happened while silhouette was rewarding the isolation of a few extreme users.
    """
    return int(best_k) == int(scores["k"].max())


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = RANDOM_SEED,
) -> Tuple[KMeans, np.ndarray]:
    """Fits the final KMeans model and returns it with its cluster assignments."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = None,
    n_components: Optional[int] = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[HDBSCAN, np.ndarray]:
    """
    Fits HDBSCAN, which infers the cluster count and can mark points as noise.

    Noise points receive label -1. Unlike KMeans, which must assign every user to some
    centroid, HDBSCAN can say that a user belongs to no dense group at all.

    The features are projected onto their leading principal components first. Density
    is the quantity HDBSCAN estimates, and in twelve standardized dimensions 400 points
    are so sparse that no region clears the density threshold: run directly on the full
    feature space this labelled all 400 users as noise and found zero clusters. Reducing
    the dimensionality concentrates the same points into a space where density is
    actually measurable.
    """
    if n_components and n_components < X.shape[1]:
        reducer = PCA(n_components=n_components, random_state=random_state)
        embedded = reducer.fit_transform(X)
    else:
        embedded = X

    # copy is set explicitly: scikit-learn 1.10 flips its default, and leaving it
    # unset would let HDBSCAN mutate the caller's feature array in place.
    model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True
    )
    labels = model.fit_predict(embedded)
    return model, labels


def score_clustering(X: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Scores a clustering, excluding HDBSCAN noise points from the quality metrics.

    Silhouette is undefined for a single cluster, so that case returns None rather than
    raising.
    """
    mask = labels != -1
    n_noise = int(np.sum(~mask))
    unique = sorted(set(labels[mask].tolist()))

    result: Dict[str, Any] = {
        "n_clusters": len(unique),
        "n_noise": n_noise,
        "noise_fraction": float(n_noise / len(labels)) if len(labels) else 0.0,
        "cluster_sizes": {str(c): int(np.sum(labels == c)) for c in unique},
    }

    if len(unique) >= 2 and int(np.sum(mask)) > len(unique):
        result["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
        result["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
    else:
        result["silhouette"] = None
        result["davies_bouldin"] = None

    return result


def profile_clusters(
    df: pl.DataFrame,
    labels: np.ndarray,
    feature_names: List[str],
) -> pl.DataFrame:
    """
    Returns per-cluster means for every profiling feature, plus the cluster size.

    Means are reported in original units so the profile is readable, with z-scores
    computed separately for naming.
    """
    labelled = df.with_columns(pl.Series("cluster", labels))
    aggregations = [pl.len().alias("users")]
    aggregations += [pl.col(name).mean().alias(name) for name in feature_names]
    if "predicted_clv" in labelled.columns:
        aggregations.append(pl.col("predicted_clv").mean().alias("predicted_clv"))

    return labelled.group_by("cluster").agg(aggregations).sort("cluster")


def compute_cluster_zscores(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
) -> pl.DataFrame:
    """
    Expresses each cluster's centroid as a deviation from the user population.

    `X` is the standardized feature matrix, so each column already has mean zero and
    unit variance across users, and a cluster's mean row is its z-profile directly.

    Z-scores were previously computed across cluster means with every cluster weighted
    equally. One tiny cluster of extreme users then dominated the reference point: on
    real data a 6-user cluster averaging 117 purchases dragged the mean high enough that
    a cluster of heavy buyers scored below average and was named "Window Shoppers".
    """
    rows: List[Dict[str, Any]] = []
    for cluster in sorted(set(int(c) for c in labels)):
        mask = labels == cluster
        centroid = X[mask].mean(axis=0)
        row: Dict[str, Any] = {"cluster": cluster, "users": int(mask.sum())}
        row.update({name: float(centroid[i]) for i, name in enumerate(feature_names)})
        rows.append(row)
    return pl.DataFrame(rows)


def signature_coverage(weights: Dict[str, float], feature_names: List[str]) -> float:
    """
    Returns the share of a signature's weight norm carried by available features.
    """
    total = float(np.sqrt(sum(w * w for w in weights.values())))
    if total == 0.0:
        return 0.0
    usable = float(np.sqrt(sum(w * w for f, w in weights.items() if f in feature_names)))
    return usable / total


def eligible_signatures(
    feature_names: List[str],
    signatures: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """Returns each signature's coverage, for reporting which names the data supports."""
    sigs = signatures or SEGMENT_SIGNATURES
    return {name: signature_coverage(weights, feature_names) for name, weights in sigs.items()}


def name_clusters(
    zscores: pl.DataFrame,
    feature_names: List[str],
    signatures: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[int, str]:
    """
    Assigns each cluster the segment name whose signature best matches its z-profile.

    Matching is the cosine similarity between the cluster's z-scores and the signature
    weights, over the features present in `feature_names`. A signature is skipped when
    those features carry less than `MIN_SIGNATURE_COVERAGE` of its weight, because what
    remains no longer describes the segment it is named after. Each name is used at
    most once, assigned greedily by strongest match, so two clusters never collapse to
    the same label.
    """
    sigs = signatures or SEGMENT_SIGNATURES

    candidates: List[Tuple[float, int, str]] = []
    for row in zscores.iter_rows(named=True):
        cluster = int(row["cluster"])
        for segment, weights in sigs.items():
            usable = {
                feature: weight
                for feature, weight in weights.items()
                if feature in feature_names and feature in row
            }
            if not usable:
                continue
            if signature_coverage(weights, feature_names) < MIN_SIGNATURE_COVERAGE:
                continue

            dot = sum(float(row[feature]) * weight for feature, weight in usable.items())
            # Normalize by both vector magnitudes, making this a cosine similarity rather
            # than a raw dot product. Without it a signature that simply lists more
            # features outscores a better-matching shorter one purely on term count,
            # which mislabeled the high-entropy cross-app cluster as window shoppers.
            weight_norm = float(np.sqrt(sum(w * w for w in usable.values())))
            profile_norm = float(
                np.sqrt(sum(float(row[feature]) ** 2 for feature in usable))
            )
            if weight_norm == 0.0 or profile_norm == 0.0:
                continue

            candidates.append((dot / (weight_norm * profile_norm), cluster, segment))

    candidates.sort(reverse=True)
    assigned: Dict[int, str] = {}
    used: set = set()
    for score, cluster, segment in candidates:
        if cluster in assigned or segment in used:
            continue
        if score < MIN_SIGNATURE_SIMILARITY:
            # Every remaining pairing is a poor match, so stop rather than forcing a
            # misleading label onto the leftover clusters.
            break
        assigned[cluster] = segment
        used.add(segment)

    # Clusters that matched nothing well keep a neutral label instead of a wrong one.
    for row in zscores.iter_rows(named=True):
        cluster = int(row["cluster"])
        assigned.setdefault(cluster, f"Unclassified Segment {cluster}")

    return assigned


def describe_cluster(
    row: Dict[str, Any],
    zrow: Dict[str, Any],
    feature_names: List[str],
    top_n: int = 3,
) -> str:
    """Builds a short readable sentence describing what distinguishes a cluster."""
    deviations = [
        (abs(float(zrow[name])), name, float(zrow[name]))
        for name in feature_names
        if name in zrow
    ]
    deviations.sort(reverse=True)

    parts: List[str] = []
    for _, name, value in deviations[:top_n]:
        direction = "high" if value > 0 else "low"
        parts.append(f"{direction} {name.replace('_', ' ')}")

    return f"{int(row['users'])} users with " + ", ".join(parts)


def plot_elbow_and_silhouette(
    scores: pl.DataFrame,
    best_k: int,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Saves the elbow curve alongside silhouette and Davies-Bouldin scores."""
    out = Path(output_path) if output_path else ELBOW_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    ks = scores["k"].to_list()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    axes[0].plot(ks, scores["inertia"].to_list(), marker="o", color="#2f6f9f")
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Within-cluster sum of squares")

    axes[1].plot(ks, scores["silhouette"].to_list(), marker="s", color="#7a9f3f")
    axes[1].set_title("Silhouette Score (higher is better)")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette")

    axes[2].plot(ks, scores["davies_bouldin"].to_list(), marker="^", color="#c0504d")
    axes[2].set_title("Davies-Bouldin Index (lower is better)")
    axes[2].set_xlabel("Number of clusters (k)")
    axes[2].set_ylabel("Davies-Bouldin")

    for ax in axes:
        ax.set_xticks(ks)
        ax.grid(alpha=0.3, linestyle=":")
        ax.axvline(best_k, color="#888888", linestyle="--", alpha=0.8)

    fig.suptitle(f"Cluster Count Selection (selected k = {best_k})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_pca_scatter(
    X: np.ndarray,
    kmeans_labels: np.ndarray,
    hdbscan_labels: np.ndarray,
    cluster_names: Dict[int, str],
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, float]:
    """
    Projects the scaled features to two dimensions and plots both clusterings.

    Returns the output path and the share of variance the two components retain, which
    is the honest caveat on any 2-D view of a twelve-dimensional clustering.
    """
    out = Path(output_path) if output_path else PCA_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X)
    explained = float(np.sum(pca.explained_variance_ratio_))

    fig, (ax_km, ax_hdb) = plt.subplots(1, 2, figsize=(15, 6.2))
    palette = plt.get_cmap("tab10")

    for cluster in sorted(set(kmeans_labels.tolist())):
        mask = kmeans_labels == cluster
        ax_km.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=28,
            alpha=0.75,
            color=palette(cluster % 10),
            label=f"{cluster}: {cluster_names.get(int(cluster), cluster)}",
        )
    ax_km.set_title("KMeans Segments")
    ax_km.legend(fontsize=8, loc="best")

    for cluster in sorted(set(hdbscan_labels.tolist())):
        mask = hdbscan_labels == cluster
        is_noise = cluster == -1
        ax_hdb.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=28,
            alpha=0.45 if is_noise else 0.75,
            color="#bbbbbb" if is_noise else palette(cluster % 10),
            label="noise" if is_noise else f"cluster {cluster}",
        )
    ax_hdb.set_title("HDBSCAN Clusters (density based)")
    ax_hdb.legend(fontsize=8, loc="best")

    for ax in (ax_km, ax_hdb):
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.grid(alpha=0.3, linestyle=":")

    fig.suptitle(
        f"User Segments in PCA Space ({explained:.1%} of variance retained)", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, explained


def plot_cluster_profiles(
    zscores: pl.DataFrame,
    feature_names: List[str],
    cluster_names: Dict[int, str],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Saves a heatmap of per-cluster feature z-scores."""
    out = Path(output_path) if output_path else PROFILE_PLOT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.array(
        [[float(row[name]) for name in feature_names] for row in zscores.iter_rows(named=True)]
    )
    clusters = [int(row["cluster"]) for row in zscores.iter_rows(named=True)]
    limit = float(np.abs(matrix).max()) if matrix.size else 1.0

    fig, ax = plt.subplots(figsize=(13, 1.2 + 0.7 * len(clusters)))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([n.replace("_", " ") for n in feature_names], rotation=40, ha="right")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"{c}: {cluster_names.get(c, c)}" for c in clusters])
    ax.set_title("Segment Profiles (z-score against population mean)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7,
                color="white" if abs(matrix[i, j]) > limit * 0.6 else "black",
            )

    fig.colorbar(im, ax=ax, shrink=0.8, label="z-score")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def run_segmentation_pipeline(
    k_values: Tuple[int, ...] = DEFAULT_K_VALUES,
    min_cluster_size: int = 15,
    output_path: Optional[Union[str, Path]] = None,
    features_dir: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Runs cluster selection, fits both clusterers, profiles and names the segments.

    `features_dir` reads the train matrix and CLV table from another dataset's feature
    directory and writes the segment table there; `artifacts_dir` redirects the model,
    scaler, plots, and summary. Explicit paths win over them.
    """
    df = load_segmentation_data(
        features_path=rebase_path(TRAIN_FEATURES_PATH, features_dir),
        clv_path=rebase_path(USER_CLV_PATH, features_dir),
    )
    feature_names, dead_features = split_live_features(df)
    if dead_features:
        print(f"Dropping {len(dead_features)} constant features with no information: "
              f"{', '.join(dead_features)}")
    recency_fill = censored_recency_fill(df, feature_names)
    X, feature_names, scaler = scale_features(df, feature_names, recency_fill=recency_fill)
    print(f"Clustering {X.shape[0]} users on {X.shape[1]} standardized features")

    coverage = eligible_signatures(feature_names)
    unsupported = [name for name, value in coverage.items() if value < MIN_SIGNATURE_COVERAGE]
    if unsupported:
        print(f"  Segment names the populated features cannot support: {', '.join(unsupported)}")

    print("\n--- Cluster Count Selection ---")
    scores = evaluate_k_range(X, k_values=k_values)
    print(scores)

    best_k = select_best_k(scores)
    at_boundary = k_at_range_boundary(scores, best_k)
    print(f"\n  Selected k = {best_k} by highest silhouette score")
    if at_boundary:
        print("  Note: the selected k is the largest candidate tried, so silhouette had not")
        print("  peaked within the range. Treat the cluster count as unresolved.")

    kmeans_model, kmeans_labels = fit_kmeans(X, n_clusters=best_k)
    kmeans_quality = score_clustering(X, kmeans_labels)
    print(f"  KMeans: {kmeans_quality['n_clusters']} clusters, "
          f"silhouette {kmeans_quality['silhouette']:.4f}, "
          f"sizes {kmeans_quality['cluster_sizes']}")

    print(f"\n--- HDBSCAN (min_cluster_size = {min_cluster_size}) ---")
    hdbscan_model, hdbscan_labels = fit_hdbscan(X, min_cluster_size=min_cluster_size)
    hdbscan_quality = score_clustering(X, hdbscan_labels)
    silhouette_text = (
        f"{hdbscan_quality['silhouette']:.4f}"
        if hdbscan_quality["silhouette"] is not None
        else "undefined (fewer than 2 clusters)"
    )
    print(f"  HDBSCAN: {hdbscan_quality['n_clusters']} clusters, "
          f"{hdbscan_quality['n_noise']} noise points "
          f"({hdbscan_quality['noise_fraction']:.1%}), silhouette {silhouette_text}")
    print("  HDBSCAN infers the cluster count from density and labels points that belong")
    print("  to no dense region as noise, which KMeans cannot do.")

    profile = profile_clusters(df, kmeans_labels, feature_names)
    zscores = compute_cluster_zscores(X, kmeans_labels, feature_names)
    cluster_names = name_clusters(zscores, feature_names)

    print("\n--- Segment Profiles ---")
    descriptions: List[Dict[str, Any]] = []
    profile_rows = list(profile.iter_rows(named=True))
    zscore_rows = {int(r["cluster"]): r for r in zscores.iter_rows(named=True)}

    for row in profile_rows:
        cluster = int(row["cluster"])
        name = cluster_names[cluster]
        description = describe_cluster(row, zscore_rows[cluster], feature_names)
        print(f"\n  Cluster {cluster} \"{name}\": {description}")
        readout = [
            ("events 30d", "total_events_30d", "{:.0f}"),
            ("purchases 30d", "total_purchases_30d", "{:.2f}"),
            ("spend 30d", "total_spend_30d", "{:,.0f}"),
            ("app entropy", "app_entropy", "{:.3f}"),
            ("cross-app comparisons", "cross_app_comparison_count", "{:.1f}"),
            ("avg discount %", "avg_discount_pct_on_purchases", "{:.1f}"),
        ]
        parts = [
            f"{label} {fmt.format(float(row[key]))}"
            for label, key, fmt in readout
            if key in row and row[key] is not None
        ]
        print("    " + " | ".join(parts))
        if "predicted_clv" in row and row["predicted_clv"] is not None:
            print(f"    mean predicted CLV {float(row['predicted_clv']):,.0f} INR")

        descriptions.append(
            {
                "cluster": cluster,
                "name": name,
                "users": int(row["users"]),
                "description": description,
            }
        )

    segments = df.select("user_id").with_columns(
        [
            pl.Series("kmeans_cluster", kmeans_labels),
            pl.Series("hdbscan_cluster", hdbscan_labels),
            pl.Series("segment_name", [cluster_names[int(c)] for c in kmeans_labels]),
        ]
    )
    out_path = Path(output_path) if output_path else rebase_path(USER_SEGMENTS_PATH, features_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    segments.write_parquet(out_path)

    model_path = rebase_path(SEGMENTATION_MODEL_PATH, artifacts_dir)
    scaler_path = rebase_path(SEGMENTATION_SCALER_PATH, artifacts_dir)
    summary_path = rebase_path(SEGMENTATION_SUMMARY_PATH, artifacts_dir)

    elbow_plot = plot_elbow_and_silhouette(
        scores, best_k, output_path=rebase_path(ELBOW_PLOT_PATH, artifacts_dir)
    )
    pca_plot, explained_variance = plot_pca_scatter(
        X, kmeans_labels, hdbscan_labels, cluster_names,
        output_path=rebase_path(PCA_PLOT_PATH, artifacts_dir),
    )
    profile_plot = plot_cluster_profiles(
        zscores, feature_names, cluster_names,
        output_path=rebase_path(PROFILE_PLOT_PATH, artifacts_dir),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans_model, model_path)
    joblib.dump(scaler, scaler_path)

    if "archetype" in df.columns:
        crosstab = (
            df.with_columns(pl.Series("segment_name", [cluster_names[int(c)] for c in kmeans_labels]))
            .group_by(["segment_name", "archetype"])
            .agg(pl.len().alias("users"))
            .sort(["segment_name", "users"], descending=[False, True])
        )
        print("\n--- Discovered Segment against Simulated Archetype ---")
        print(crosstab)
        print("\n  The archetype is simulator ground truth and was never given to the")
        print("  clusterer. Agreement here means the unsupervised segments recovered the")
        print("  behavioral groups that actually generated the data.")

    payload = {
        "n_users": int(X.shape[0]),
        "feature_names": feature_names,
        "dropped_constant_features": dead_features,
        # Serving must fill missing recency with these exact values before applying the
        # saved scaler, or new users will be placed inconsistently with training.
        "recency_fill_values": recency_fill,
        "signature_coverage": coverage,
        "k_selection": scores.to_dicts(),
        "selected_k": best_k,
        "selected_k_at_range_boundary": at_boundary,
        "kmeans_quality": kmeans_quality,
        "hdbscan_quality": hdbscan_quality,
        "pca_explained_variance": explained_variance,
        "segments": descriptions,
        "cluster_profile": profile.to_dicts(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)

    print(f"\nSaved segment assignments to {out_path}")
    print(f"Saved elbow plot to         {elbow_plot}")
    print(f"Saved PCA scatter to        {pca_plot}")
    print(f"Saved profile heatmap to    {profile_plot}")
    print(f"Saved model to              {model_path}")
    print(f"Saved summary to            {summary_path}")

    return {
        "kmeans": kmeans_model,
        "hdbscan": hdbscan_model,
        "scaler": scaler,
        "scaled_features": X,
        "feature_names": feature_names,
        "dropped_features": dead_features,
        "k_scores": scores,
        "selected_k": best_k,
        "kmeans_labels": kmeans_labels,
        "hdbscan_labels": hdbscan_labels,
        "profile": profile,
        "zscores": zscores,
        "cluster_names": cluster_names,
        "segments": segments,
        "paths": {
            "segments": out_path,
            "elbow_plot": elbow_plot,
            "pca_plot": pca_plot,
            "profile_plot": profile_plot,
            "model": model_path,
            "scaler": scaler_path,
            "summary": summary_path,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KMeans and HDBSCAN user segmentation.")
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--features-dir", type=str, default=None, help="Feature directory to read and write")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory for model outputs")
    args = parser.parse_args()

    run_segmentation_pipeline(
        k_values=tuple(args.k_values),
        min_cluster_size=args.min_cluster_size,
        features_dir=args.features_dir,
        artifacts_dir=args.artifacts_dir,
    )
