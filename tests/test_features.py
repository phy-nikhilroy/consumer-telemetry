"""
Integrity tests for the assembled feature store.

These guard the properties every downstream model assumes: no missing or infinite
values in columns that feed an estimator, every feature inside its mathematically
possible range, and consistent schemas across the train and test splits.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.config import COUNT_TARGET, PRIMARY_TARGET, SWITCHING_TARGET, TARGET_COLUMNS

TRAIN_PATH = Path("data/features/train_features.parquet")
TEST_PATH = Path("data/features/test_features.parquet")
SELECTED_PATH = Path("data/features/selected_features.json")

# Columns that must never be null, because an estimator consumes them directly.
CRITICAL_COLUMNS = (
    "user_id",
    "total_events_30d",
    "total_purchases_30d",
    "total_spend_30d",
    "app_entropy",
    "dominant_app_share",
    "view_to_cart_ratio",
    "cart_to_purchase_ratio",
    PRIMARY_TARGET,
    COUNT_TARGET,
    SWITCHING_TARGET,
)

# Features whose valid range is fixed by their definition rather than by the data.
BOUNDED_FEATURES = {
    # Shannon entropy over 4 apps cannot exceed log2(4) = 2.
    "app_entropy": (0.0, 2.0),
    "dominant_app_share": (0.0, 1.0),
    "pct_purchases_with_discount": (0.0, 1.0),
    "weekend_event_share": (0.0, 1.0),
    "evening_event_share": (0.0, 1.0),
    "avg_discount_pct_on_purchases": (0.0, 100.0),
}

NON_NEGATIVE_FEATURES = (
    "total_events_7d",
    "total_events_30d",
    "total_events_lifetime",
    "total_purchases_7d",
    "total_purchases_30d",
    "total_sessions_7d",
    "total_sessions_30d",
    "avg_events_per_session",
    "total_spend_7d",
    "total_spend_30d",
    "total_spend_lifetime",
    "avg_order_value",
    "max_single_purchase",
    "num_apps_used",
    "cross_app_comparison_count",
    "avg_session_duration_minutes",
    "price_range_browsed",
    "category_diversity",
)


def _load(path: Path) -> pl.DataFrame:
    """Loads a feature matrix or skips the test when it has not been generated."""
    if not path.exists():
        pytest.skip(f"Feature matrix not available at {path}")
    return pl.read_parquet(path)


@pytest.fixture(scope="module")
def train() -> pl.DataFrame:
    return _load(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_matrix() -> pl.DataFrame:
    return _load(TEST_PATH)


@pytest.mark.parametrize("split", ["train", "test"])
def test_critical_columns_have_no_nulls(split, train, test_matrix):
    """Verifies no estimator-facing column contains a null."""
    df = train if split == "train" else test_matrix

    for column in CRITICAL_COLUMNS:
        assert column in df.columns, f"Missing critical column: {column}"
        assert df[column].null_count() == 0, f"{split}.{column} contains nulls"


@pytest.mark.parametrize("split", ["train", "test"])
def test_no_infinite_values_in_numeric_features(split, train, test_matrix):
    """Verifies no numeric feature holds an infinity, which would break scaling."""
    df = train if split == "train" else test_matrix

    for column, dtype in zip(df.columns, df.dtypes):
        if not dtype.is_numeric():
            continue
        values = df[column].cast(pl.Float64).fill_null(0.0).to_numpy()
        assert np.isfinite(values).all(), f"{split}.{column} contains non-finite values"


@pytest.mark.parametrize("split", ["train", "test"])
def test_bounded_features_stay_within_definition(split, train, test_matrix):
    """Verifies features with a definitional range never exceed it."""
    df = train if split == "train" else test_matrix

    for column, (low, high) in BOUNDED_FEATURES.items():
        if column not in df.columns:
            continue
        series = df[column].cast(pl.Float64).drop_nulls()
        if series.len() == 0:
            continue
        assert series.min() >= low - 1e-9, f"{split}.{column} below {low}"
        assert series.max() <= high + 1e-9, f"{split}.{column} above {high}"


@pytest.mark.parametrize("split", ["train", "test"])
def test_count_and_money_features_are_non_negative(split, train, test_matrix):
    """Verifies counts, spends, and durations are never negative."""
    df = train if split == "train" else test_matrix

    for column in NON_NEGATIVE_FEATURES:
        if column not in df.columns:
            continue
        series = df[column].cast(pl.Float64).drop_nulls()
        if series.len() == 0:
            continue
        assert series.min() >= -1e-9, f"{split}.{column} is negative"


@pytest.mark.parametrize("split", ["train", "test"])
def test_binary_targets_are_binary(split, train, test_matrix):
    """Verifies the binary targets hold only 0 and 1, and are not degenerate."""
    df = train if split == "train" else test_matrix

    for target in (PRIMARY_TARGET, SWITCHING_TARGET):
        values = set(df[target].unique().to_list())
        assert values.issubset({0, 1}), f"{split}.{target} holds {values}"
        # A target that is constant carries no information and silently makes every
        # classification metric meaningless.
        assert len(values) == 2, f"{split}.{target} is constant"

        rate = float(df[target].mean())
        assert 0.02 < rate < 0.98, f"{split}.{target} is degenerate at rate {rate:.3f}"


@pytest.mark.parametrize("split", ["train", "test"])
def test_count_target_is_consistent_with_binary_target(split, train, test_matrix):
    """Verifies the purchase count and purchase flag agree row by row."""
    df = train if split == "train" else test_matrix

    mismatched = df.filter(
        ((pl.col(COUNT_TARGET) > 0) & (pl.col(PRIMARY_TARGET) == 0))
        | ((pl.col(COUNT_TARGET) == 0) & (pl.col(PRIMARY_TARGET) == 1))
    )
    assert mismatched.height == 0, f"{split} has {mismatched.height} inconsistent target rows"


@pytest.mark.parametrize("split", ["train", "test"])
def test_switching_implies_purchasing(split, train, test_matrix):
    """Verifies a user cannot switch apps without making a purchase."""
    df = train if split == "train" else test_matrix

    impossible = df.filter((pl.col(SWITCHING_TARGET) == 1) & (pl.col(PRIMARY_TARGET) == 0))
    assert impossible.height == 0, (
        f"{split} has {impossible.height} users flagged as switching without purchasing"
    )


def test_user_ids_are_unique_within_each_split(train, test_matrix):
    """Verifies one row per user, since every feature is a per-user aggregate."""
    for name, df in (("train", train), ("test", test_matrix)):
        assert df["user_id"].n_unique() == df.height, f"{name} has duplicate user rows"
        assert df["user_id"].null_count() == 0


def test_train_and_test_schemas_match(train, test_matrix):
    """Verifies both splits share an identical schema and population."""
    assert train.columns == test_matrix.columns
    assert train.schema == test_matrix.schema
    # Both splits are snapshots of the same users at different reference dates.
    assert set(train["user_id"].to_list()) == set(test_matrix["user_id"].to_list())


def test_selected_features_are_present_and_pruned(train):
    """Verifies the selected feature list is usable and excludes targets."""
    import json

    if not SELECTED_PATH.exists():
        pytest.skip("Selected feature list not available")

    with open(SELECTED_PATH, "r", encoding="utf-8") as f:
        info = json.load(f)

    selected = info["selected_features"]
    assert len(selected) == info["selected_feature_count"]
    assert len(selected) == len(set(selected)), "Duplicate entries in selected features"

    for name in selected:
        assert name in train.columns, f"Selected feature {name} missing from matrix"
        assert name not in TARGET_COLUMNS, f"Target {name} leaked into the feature list"

    # Pruned features must not also appear as selected.
    overlap = set(selected) & set(info.get("pruned_collinear_features", []))
    assert not overlap, f"Features both selected and pruned: {overlap}"
