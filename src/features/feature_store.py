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
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import numpy as np
import polars as pl

from src.config import (
    DEFAULT_DB_PATH,
    TRAIN_FEATURES_PATH,
    TEST_FEATURES_PATH,
    SELECTED_FEATURES_PATH,
)
from src.features.database import DatabaseManager
from src.features.rfm_features import compute_rfm_features
from src.features.behavioral_features import compute_behavioral_features

DEFAULT_TRAIN_PATH = TRAIN_FEATURES_PATH
DEFAULT_TEST_PATH = TEST_FEATURES_PATH


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
    train_ref_date: str = "2026-10-18 00:00:00",
    test_ref_date: str = "2026-10-21 00:00:00",
    prediction_window_days: int = 7,
    train_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Constructs temporal train and test splits to strictly prevent data leakage:
    - Train: reference_date = Day 20, target window = Days 21-27
    - Test:  reference_date = Day 23, target window = Days 24-30
    """
    db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    con = db_mgr.get_connection()

    try:
        print(f"Building TRAIN feature matrix (reference_date: {train_ref_date})...")
        train_df = build_feature_matrix(
            reference_date=train_ref_date,
            prediction_window_days=prediction_window_days,
            con=con,
        )

        print(f"Building TEST feature matrix (reference_date: {test_ref_date})...")
        test_df = build_feature_matrix(
            reference_date=test_ref_date,
            prediction_window_days=prediction_window_days,
            con=con,
        )

        t_out = Path(train_path) if train_path else DEFAULT_TRAIN_PATH
        te_out = Path(test_path) if test_path else DEFAULT_TEST_PATH

        t_out.parent.mkdir(parents=True, exist_ok=True)
        te_out.parent.mkdir(parents=True, exist_ok=True)

        train_df.write_parquet(t_out)
        test_df.write_parquet(te_out)

        print(f"Saved Train matrix to {t_out} (shape: {train_df.shape})")
        print(f"Saved Test matrix to {te_out} (shape: {test_df.shape})")

        print_target_distributions(train_df, test_df)

        return train_df, test_df
    finally:
        con.close()


def print_target_distributions(train_df: pl.DataFrame, test_df: pl.DataFrame) -> None:
    """Prints comparative class distributions between Train and Test sets."""
    train_buyers = train_df.filter(pl.col("will_purchase_next_7d") == 1).height
    train_total = train_df.height
    train_rate = (train_buyers / train_total) * 100.0 if train_total > 0 else 0.0

    test_buyers = test_df.filter(pl.col("will_purchase_next_7d") == 1).height
    test_total = test_df.height
    test_rate = (test_buyers / test_total) * 100.0 if test_total > 0 else 0.0

    train_switchers = train_df.filter(pl.col("will_switch_app_next_7d") == 1).height
    test_switchers = test_df.filter(pl.col("will_switch_app_next_7d") == 1).height

    print("\n--- Target Variable Distributions (Train vs Test) ---")
    print(f"{'Metric':<32} | {'Train (Day 20 -> 27)':<22} | {'Test (Day 23 -> 30)':<22}")
    print("-" * 80)
    print(f"{'Total Users':<32} | {train_total:<22} | {test_total:<22}")
    print(f"{'Buyers (will_purchase = 1)':<32} | {train_buyers} ({train_rate:.1f}%)" + " " * 12 + f"| {test_buyers} ({test_rate:.1f}%)")
    print(f"{'App Switchers (will_switch = 1)':<32} | {train_switchers:<22} | {test_switchers:<22}")
    print(f"{'Avg Purchases (num_purchases)':<32} | {train_df['num_purchases_next_7d'].mean():.2f}" + " " * 17 + f"| {test_df['num_purchases_next_7d'].mean():.2f}")


def get_modeling_matrices(
    split: str = "train",
    target_col: str = "will_purchase_next_7d",
    features_path: Optional[Union[str, Path]] = None,
    selected_features_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads train or test feature matrix, filters to selected non-collinear features,
    and returns (X, y, feature_names) ready for ML estimators.
    """
    if features_path is None:
        features_path = TRAIN_FEATURES_PATH if split == "train" else TEST_FEATURES_PATH

    f_file = Path(features_path)
    if not f_file.exists():
        raise FileNotFoundError(f"Feature matrix not found at {f_file}")

    df = pl.read_parquet(f_file)

    s_file = Path(selected_features_path) if selected_features_path else SELECTED_FEATURES_PATH
    if s_file.exists():
        with open(s_file, "r", encoding="utf-8") as f:
            selected_info = json.load(f)
        feature_names = [f for f in selected_info["selected_features"] if f in df.columns]
    else:
        ignore = {"user_id", "will_purchase_next_7d", "num_purchases_next_7d", "will_switch_app_next_7d", "top_category_l1"}
        feature_names = [c for c in df.columns if c not in ignore]

    X = df.select(feature_names).to_pandas().fillna(0.0).to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy()

    return X, y, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble feature store and temporal train/test datasets.")
    parser.add_argument("--train-ref", type=str, default="2026-10-18 00:00:00", help="Train reference date")
    parser.add_argument("--test-ref", type=str, default="2026-10-21 00:00:00", help="Test reference date")
    args = parser.parse_args()

    generate_train_test_datasets(train_ref_date=args.train_ref, test_ref_date=args.test_ref)
