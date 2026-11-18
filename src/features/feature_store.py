"""
Unified Feature Store Assembly and Target Variable Engineering.
Merges RFM and behavioral feature matrices, creates leak-free forward prediction targets,
and generates temporal train and test splits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import polars as pl

from src.config import (
    DEFAULT_DB_PATH,
    MISSING_AS_SENTINEL_FEATURES,
    PREDICTION_WINDOW_DAYS,
    RECENCY_FEATURES,
    RECENCY_NULL_SENTINEL_DAYS,
    TRAIN_FEATURES_PATH,
    TEST_FEATURES_PATH,
    TEST_REFERENCE_DATE,
    TRAIN_REFERENCE_DATES,
    SELECTED_FEATURES_PATH,
    rebase_path,
)
from src.features.database import DatabaseManager
from src.features.rfm_features import compute_rfm_features
from src.features.behavioral_features import compute_behavioral_features

DEFAULT_TRAIN_PATH = TRAIN_FEATURES_PATH
DEFAULT_TEST_PATH = TEST_FEATURES_PATH

# Columns that identify a row or hold a forward target; never model inputs.
NON_FEATURE_COLUMNS = frozenset({
    "user_id",
    "reference_date",
    "will_purchase_next_7d",
    "num_purchases_next_7d",
    "will_switch_app_next_7d",
    "top_category_l1",
})


def _parse_ref(value: Union[str, datetime]) -> datetime:
    """Parses a reference date string (or passes a datetime through)."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).strip())


def validate_split(
    train_ref_dates: Sequence[str],
    test_ref_date: str,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
    max_event_timestamp: Optional[datetime] = None,
) -> None:
    """
    Raises ValueError unless the temporal split is leak free.

    Requires (a) at least one train reference date, (b) every train target window
    `(r, r + window]` to end at or before the test reference date, and (c) when
    `max_event_timestamp` is given, the test target window to end at or before the last
    observed event, so test labels are not silently truncated by the end of the data.
    """
    refs = list(train_ref_dates)
    if not refs:
        raise ValueError("train_ref_dates must contain at least one reference date")
    window = timedelta(days=prediction_window_days)
    test_ref = _parse_ref(test_ref_date)
    for raw in refs:
        ref = _parse_ref(raw)
        if ref + window > test_ref:
            raise ValueError(
                f"train target window ({ref} -> {ref + window}) overlaps the test "
                f"reference date {test_ref}"
            )
    if max_event_timestamp is not None:
        data_end = _parse_ref(max_event_timestamp)
        if test_ref + window > data_end:
            raise ValueError(
                f"test target window ends at {test_ref + window}, after the last event "
                f"{data_end}"
            )


REFERENCE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def split_for_window(
    max_event_timestamp: datetime,
    min_event_timestamp: Optional[datetime] = None,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
    train_ref_dates: Sequence[str] = TRAIN_REFERENCE_DATES,
    test_ref_date: str = TEST_REFERENCE_DATE,
) -> Tuple[List[str], str]:
    """
    The configured split when its test window fits the data; otherwise the latest split on
    the same weekly grid that does.

    The grid is the configured test reference date moved back in whole weeks. The test date is
    the latest grid date whose target window ends by `max_event_timestamp`; the train dates are
    the up to `len(train_ref_dates)` grid dates before it whose windows end by the test date and,
    when `min_event_timestamp` is given, that have at least one window of history counted from
    the midnight of the first event's day. Shorter
    experiment runs (a 30 or 45 day simulation) use this instead of shifting dates by hand; the
    result always passes `validate_split`. Raises ValueError when no such split exists.
    """
    window = timedelta(days=prediction_window_days)
    data_end = _parse_ref(max_event_timestamp)
    test = _parse_ref(test_ref_date)
    step = timedelta(days=7)
    while test + window > data_end:
        test -= step
    n_train = max(1, len(list(train_ref_dates)))
    earliest = None
    if min_event_timestamp is not None:
        first = _parse_ref(min_event_timestamp)
        # History is counted in whole days from the midnight of the first event's day.
        earliest = datetime.combine(first.date(), datetime.min.time()) + window
    trains: List[datetime] = []
    ref = test - step
    while len(trains) < n_train and (earliest is None or ref >= earliest):
        if ref + window <= test:
            trains.append(ref)
        ref -= step
    if not trains:
        raise ValueError(f"no weekly split fits data ending {data_end}")
    train_out = [r.strftime(REFERENCE_DATE_FORMAT) for r in sorted(trains)]
    test_out = test.strftime(REFERENCE_DATE_FORMAT)
    validate_split(train_out, test_out, prediction_window_days, max_event_timestamp=data_end)
    return train_out, test_out


def split_for_warehouse(
    db_path: Optional[Union[str, Path]] = None,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
) -> Tuple[List[str], str]:
    """`split_for_window` over the first and last `fact_events` timestamps of a warehouse."""
    con = duckdb.connect(str(db_path or DEFAULT_DB_PATH), read_only=True)
    try:
        first, last = con.execute("SELECT MIN(timestamp), MAX(timestamp) FROM fact_events").fetchone()
    finally:
        con.close()
    if last is None:
        raise ValueError("fact_events is empty; cannot choose a split")
    return split_for_window(last, first, prediction_window_days)


def compute_target_variables(
    reference_date: Union[str, datetime],
    prediction_window_days: int = 7,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    """
    Computes 3 forward target variables for each user strictly within the forward
    window: timestamp > reference_date AND timestamp <= reference_date + prediction_window_days.
    Zero past information is leaked into targets, and zero target information leaks into features.

    Target 1: will_purchase_next_7d (binary: 0 or 1)
    Target 2: num_purchases_next_7d (integer count)
    Target 3: will_switch_app_next_7d (binary: did user buy on an app different from their dominant app?)
    """
    if isinstance(reference_date, datetime):
        ref_str = reference_date.strftime("%Y-%m-%d %H:%M:%S")
    else:
        ref_str = str(reference_date).strip()

    should_close = False
    if con is None:
        db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
        con = db_mgr.get_connection()
        should_close = True

    try:
        query = f"""
            WITH dominant_apps AS (
                -- Historical dominant app strictly before or at reference date
                SELECT user_id, MODE(app_name) AS dominant_app
                FROM fact_events
                WHERE timestamp <= CAST('{ref_str}' AS TIMESTAMP)
                GROUP BY user_id
            ),
            forward_events AS (
                -- Strictly future events within the forward prediction window
                SELECT
                    user_id,
                    app_name,
                    action
                FROM fact_events
                WHERE timestamp > CAST('{ref_str}' AS TIMESTAMP)
                  AND timestamp <= CAST('{ref_str}' AS TIMESTAMP) + INTERVAL '{prediction_window_days}' DAY
            ),
            targets AS (
                SELECT
                    da.user_id,
                    CASE
                        WHEN COUNT(*) FILTER (WHERE fe.action = 'buy') > 0 THEN 1
                        ELSE 0
                    END AS will_purchase_next_7d,
                    COALESCE(COUNT(*) FILTER (WHERE fe.action = 'buy'), 0) AS num_purchases_next_7d,
                    CASE
                        WHEN COUNT(*) FILTER (WHERE fe.action = 'buy' AND fe.app_name != da.dominant_app) > 0 THEN 1
                        ELSE 0
                    END AS will_switch_app_next_7d
                FROM dominant_apps da
                LEFT JOIN forward_events fe ON da.user_id = fe.user_id
                GROUP BY da.user_id, da.dominant_app
            )
            SELECT * FROM targets ORDER BY user_id
        """
        arrow_table = con.execute(query).to_arrow_table()
        return pl.from_arrow(arrow_table)
    finally:
        if should_close:
            con.close()


def build_feature_matrix(
    reference_date: str,
    prediction_window_days: int = 7,
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> pl.DataFrame:
    """
    Assembles a unified wide feature matrix:
    - Features: computed strictly from events <= reference_date.
    - Targets: computed strictly from events within (reference_date, reference_date + prediction_window_days].
    """
    should_close = False
    if con is None:
        db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
        con = db_mgr.get_connection()
        should_close = True

    try:
        rfm_df = compute_rfm_features(reference_date=reference_date, con=con)
        behavioral_df = compute_behavioral_features(reference_date=reference_date, con=con)
        targets_df = compute_target_variables(
            reference_date=reference_date,
            prediction_window_days=prediction_window_days,
            con=con,
        )

        # Merge on user_id
        matrix = rfm_df.join(behavioral_df, on="user_id", how="inner")
        matrix = matrix.join(targets_df, on="user_id", how="inner")

        return matrix
    finally:
        if should_close:
            con.close()


def generate_train_test_datasets(
    train_ref_dates: Sequence[str] = TRAIN_REFERENCE_DATES,
    test_ref_date: str = TEST_REFERENCE_DATE,
    prediction_window_days: int = PREDICTION_WINDOW_DAYS,
    train_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    features_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Constructs temporal train and test splits to strictly prevent data leakage.

    The train matrix stacks one feature matrix per train reference date, so a user appears
    once per reference date; the `reference_date` column tells the rows apart, and grouped
    validation keys on `user_id` (see `get_modeling_groups`). The split is
    validated against the warehouse's last event before anything is built or written.
    `features_dir` redirects both matrices to another directory; explicit `train_path` and
    `test_path` win over it.
    """
    if isinstance(train_ref_dates, str):
        train_ref_dates = [train_ref_dates]
    train_refs = [str(r).strip() for r in train_ref_dates]
    test_ref = str(test_ref_date).strip()

    db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    con = db_mgr.get_connection()

    try:
        max_event = con.execute("SELECT MAX(timestamp) FROM fact_events").fetchone()[0]
        if max_event is None:
            raise ValueError("fact_events is empty; cannot build a temporal split")
        validate_split(
            train_refs,
            test_ref,
            prediction_window_days=prediction_window_days,
            max_event_timestamp=max_event,
        )

        blocks = []
        for ref in train_refs:
            print(f"Building TRAIN feature matrix (reference_date: {ref})...")
            block = build_feature_matrix(
                reference_date=ref,
                prediction_window_days=prediction_window_days,
                con=con,
            )
            blocks.append(block.with_columns(pl.lit(ref, dtype=pl.Utf8).alias("reference_date")))
        train_df = pl.concat(blocks, how="vertical")

        print(f"Building TEST feature matrix (reference_date: {test_ref})...")
        test_df = build_feature_matrix(
            reference_date=test_ref,
            prediction_window_days=prediction_window_days,
            con=con,
        ).with_columns(pl.lit(test_ref, dtype=pl.Utf8).alias("reference_date"))

        t_out = Path(train_path) if train_path else rebase_path(DEFAULT_TRAIN_PATH, features_dir)
        te_out = Path(test_path) if test_path else rebase_path(DEFAULT_TEST_PATH, features_dir)

        t_out.parent.mkdir(parents=True, exist_ok=True)
        te_out.parent.mkdir(parents=True, exist_ok=True)

        train_df.write_parquet(t_out)
        test_df.write_parquet(te_out)

        print(f"Saved Train matrix to {t_out} (shape: {train_df.shape})")
        print(f"Saved Test matrix to {te_out} (shape: {test_df.shape})")

        print_target_distributions(
            train_df,
            test_df,
            train_ref_date=", ".join(train_refs) if len(train_refs) > 1 else train_refs[0],
            test_ref_date=test_ref,
            prediction_window_days=prediction_window_days,
        )

        return train_df, test_df
    finally:
        con.close()


def _window_label(prefix: str, ref_date: Optional[str], window_days: int) -> str:
    """
    Formats a split header from its real reference dates and target window.

    `ref_date` may list several comma-separated dates (a stacked train matrix); the label
    then spans the first window's start to the last window's end.
    """
    if not ref_date:
        return prefix
    refs = sorted(_parse_ref(part) for part in str(ref_date).split(",") if part.strip())
    start = refs[0]
    end = refs[-1] + timedelta(days=window_days)
    return f"{prefix} ({start:%Y-%m-%d} -> {end:%Y-%m-%d})"


def print_target_distributions(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_ref_date: Optional[str] = None,
    test_ref_date: Optional[str] = None,
    prediction_window_days: int = 7,
) -> None:
    """
    Prints comparative class distributions between Train and Test sets.

    Column headers are derived from the reference dates actually used. They were once
    hardcoded as simulation day numbers, which printed 2026 day labels for a 2015 run.
    """
    train_label = _window_label("Train", train_ref_date, prediction_window_days)
    test_label = _window_label("Test", test_ref_date, prediction_window_days)
    train_buyers = train_df.filter(pl.col("will_purchase_next_7d") == 1).height
    train_total = train_df.height
    train_rate = (train_buyers / train_total) * 100.0 if train_total > 0 else 0.0

    test_buyers = test_df.filter(pl.col("will_purchase_next_7d") == 1).height
    test_total = test_df.height
    test_rate = (test_buyers / test_total) * 100.0 if test_total > 0 else 0.0

    train_switchers = train_df.filter(pl.col("will_switch_app_next_7d") == 1).height
    test_switchers = test_df.filter(pl.col("will_switch_app_next_7d") == 1).height

    print("\n--- Target Variable Distributions (Train vs Test) ---")
    print(f"{'Metric':<32} | {train_label:<32} | {test_label:<32}")
    print("-" * 100)
    train_buyer_text = f"{train_buyers} ({train_rate:.1f}%)"
    test_buyer_text = f"{test_buyers} ({test_rate:.1f}%)"
    train_mean = f"{train_df['num_purchases_next_7d'].mean():.2f}"
    test_mean = f"{test_df['num_purchases_next_7d'].mean():.2f}"
    print(f"{'Total Users':<32} | {train_total:<32} | {test_total:<32}")
    print(f"{'Buyers (will_purchase = 1)':<32} | {train_buyer_text:<32} | {test_buyer_text:<32}")
    print(f"{'App Switchers (will_switch = 1)':<32} | {train_switchers:<32} | {test_switchers:<32}")
    print(f"{'Avg Purchases (num_purchases)':<32} | {train_mean:<32} | {test_mean:<32}")


def fill_missing_features(
    df: pl.DataFrame,
    feature_names: Optional[List[str]] = None,
    recency_sentinel: float = RECENCY_NULL_SENTINEL_DAYS,
    recency_fill: Optional[Dict[str, float]] = None,
) -> pl.DataFrame:
    """
    Fills nulls in estimator-facing feature columns with semantically correct values.

    Recency columns (`RECENCY_FEATURES`) take `recency_sentinel`, because a null there
    means the event never happened and zero would claim it happened today. A per-column
    `recency_fill` overrides the sentinel; distance-based methods use it to censor at the
    longest observed recency instead (see `censored_recency_fill`). Columns in
    `MISSING_AS_SENTINEL_FEATURES` take their out-of-range sentinel, because a null there
    means the quantity could not be measured. Every other feature is a count, ratio, or
    amount whose null genuinely means zero.

    Returns only the requested columns, cast to Float64.
    """
    names = feature_names if feature_names is not None else df.columns
    overrides = recency_fill or {}
    expressions = []
    for name in names:
        column = pl.col(name).cast(pl.Float64)
        if name in RECENCY_FEATURES:
            fill = float(overrides.get(name, recency_sentinel))
        elif name in MISSING_AS_SENTINEL_FEATURES:
            fill = float(MISSING_AS_SENTINEL_FEATURES[name])
        else:
            fill = 0.0
        expressions.append(column.fill_null(fill).fill_nan(fill).alias(name))
    return df.select(expressions)


def censored_recency_fill(
    df: pl.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Returns, per recency column, the longest recency observed in `df`.

    Filling a never-happened recency with this value treats the user as at least as
    lapsed as anyone observed, which is the correct direction and, unlike the tree-model
    sentinel, keeps the column on its natural scale. That matters for standardized
    distance-based clustering, where a 9999 sentinel turns the column into a near-binary
    flag that dominates every distance. The values must be persisted with any fitted
    scaler so serving can reproduce them.
    """
    names = feature_names if feature_names is not None else df.columns
    fills: Dict[str, float] = {}
    for name in names:
        if name in RECENCY_FEATURES and name in df.columns:
            observed = df[name].cast(pl.Float64).drop_nulls().drop_nans()
            fills[name] = float(observed.max()) if observed.len() else 0.0
    return fills


def get_modeling_matrices(
    split: str = "train",
    target_col: str = "will_purchase_next_7d",
    features_path: Optional[Union[str, Path]] = None,
    selected_features_path: Optional[Union[str, Path]] = None,
    features_dir: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads train or test feature matrix, filters to selected non-collinear features,
    and returns (X, y, feature_names) ready for ML estimators.

    `features_dir` reads the split and its `selected_features.json` from another
    dataset's feature directory. Explicit paths win over it.
    """
    if features_path is None:
        default = TRAIN_FEATURES_PATH if split == "train" else TEST_FEATURES_PATH
        features_path = rebase_path(default, features_dir)

    f_file = Path(features_path)
    if not f_file.exists():
        raise FileNotFoundError(f"Feature matrix not found at {f_file}")

    df = pl.read_parquet(f_file)

    s_file = (
        Path(selected_features_path) if selected_features_path
        else rebase_path(SELECTED_FEATURES_PATH, features_dir)
    )
    if s_file.exists():
        with open(s_file, "r", encoding="utf-8") as f:
            selected_info = json.load(f)
        if selected_info.get("schema_version") == 2:
            by_target = selected_info.get("features_by_target", {})
            names = by_target.get(target_col, selected_info["selected_features"])
        else:
            names = selected_info["selected_features"]
        feature_names = [f for f in names if f in df.columns]
    else:
        feature_names = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    X = fill_missing_features(df, feature_names).to_numpy().astype(np.float32)
    y = df[target_col].to_numpy()

    return X, y, feature_names


def get_modeling_groups(
    split: str = "train",
    features_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Returns the `user_id` column of a split in row order.

    The train matrix holds one row per user per reference date, so grouped cross-validation
    must keep all of a user's rows in the same fold; these ids are the groups.
    """
    default = TRAIN_FEATURES_PATH if split == "train" else TEST_FEATURES_PATH
    path = rebase_path(default, features_dir)
    if not Path(path).exists():
        raise FileNotFoundError(f"Feature matrix not found at {path}")
    return pl.read_parquet(path, columns=["user_id"])["user_id"].to_numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble feature store and temporal train/test datasets.")
    parser.add_argument("--train-ref", type=str, nargs="+", default=list(TRAIN_REFERENCE_DATES),
                        help="One or more train reference dates")
    parser.add_argument("--test-ref", type=str, default=TEST_REFERENCE_DATE,
                        help="Test reference date")
    parser.add_argument("--db-path", type=str, default=None, help="Warehouse to read (default: synthetic)")
    parser.add_argument("--features-dir", type=str, default=None,
                        help="Directory for the train and test matrices (default: data/features)")
    args = parser.parse_args()

    generate_train_test_datasets(
        train_ref_dates=args.train_ref,
        test_ref_date=args.test_ref,
        db_path=args.db_path,
        features_dir=args.features_dir,
    )
