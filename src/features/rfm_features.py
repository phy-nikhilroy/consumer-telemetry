"""
RFM (Recency, Frequency, Monetary) Feature Store Engineering.
Computes leak-free behavioral feature aggregations relative to an explicit reference date.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Union

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import polars as pl
from src.config import DEFAULT_DB_PATH, RFM_FEATURES_PATH, USD_TO_INR_RATE
from src.features.database import DatabaseManager

DEFAULT_FEATURES_PATH = RFM_FEATURES_PATH


def compute_rfm_features(
    reference_date: Union[str, datetime],
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[Union[str, Path]] = None,
    usd_to_inr_rate: float = USD_TO_INR_RATE,
    user_ids: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    Computes 15+ Recency, Frequency, and Monetary features per user.
    All aggregations are strictly bounded by reference_date (timestamp <= reference_date)
    to prevent target data leakage in predictive modeling.

    `user_ids` restricts the computation to those users. The serving API uses it to score
    one user with exactly the SQL the models were trained on, rather than a reimplementation
    that could drift from it. Every feature is a per-user aggregate, so the restricted
    result equals the matching rows of the unrestricted one.
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

    user_filter = "AND user_id IN (SELECT UNNEST($user_ids))" if user_ids is not None else ""
    params = {"user_ids": [str(u) for u in user_ids]} if user_ids is not None else None

    try:
        query = f"""
            WITH eligible_events AS (
                SELECT
                    event_id,
                    user_id,
                    timestamp,
                    session_id,
                    action,
                    price,
                    currency,
                    CASE
                        WHEN currency = 'USD' THEN price * {usd_to_inr_rate}
                        ELSE price
                    END AS norm_price
                FROM fact_events
                WHERE timestamp <= CAST('{ref_str}' AS TIMESTAMP)
                {user_filter}
            ),
            user_aggs AS (
                SELECT
                    user_id,
                    -- RECENCY FEATURES
                    ROUND(EPOCH(CAST('{ref_str}' AS TIMESTAMP) - MAX(timestamp)) / 86400.0, 2) AS days_since_last_event,
                    ROUND(EPOCH(CAST('{ref_str}' AS TIMESTAMP) - MAX(timestamp) FILTER (WHERE action = 'buy')) / 86400.0, 2) AS days_since_last_purchase,
                    ROUND(EPOCH(CAST('{ref_str}' AS TIMESTAMP) - MAX(timestamp) FILTER (WHERE action = 'cart')) / 86400.0, 2) AS days_since_last_cart_add,

                    -- FREQUENCY FEATURES
                    COUNT(*) FILTER (WHERE timestamp >= CAST('{ref_str}' AS TIMESTAMP) - INTERVAL 7 DAY) AS total_events_7d,
                    COUNT(*) FILTER (WHERE timestamp >= CAST('{ref_str}' AS TIMESTAMP) - INTERVAL 30 DAY) AS total_events_30d,
                    COUNT(*) AS total_events_lifetime,

                    COUNT(*) FILTER (WHERE action = 'buy' AND timestamp >= CAST('{ref_str}' AS TIMESTAMP) - INTERVAL 7 DAY) AS total_purchases_7d,
                    COUNT(*) FILTER (WHERE action = 'buy' AND timestamp >= CAST('{ref_str}' AS TIMESTAMP) - INTERVAL 30 DAY) AS total_purchases_30d,

                    COUNT(DISTINCT session_id) FILTER (WHERE timestamp >= CAST('{ref_str}' AS TIMESTAMP) - INTERVAL 7 DAY) AS total_sessions_7d,
                    COUNT(DISTINCT session_id) FILTER (WHERE timestamp >= CAST('{ref_str}' AS TIMESTAMP) - INTERVAL 30 DAY) AS total_sessions_30d,

                    ROUND(CAST(COUNT(*) AS FLOAT) / NULLIF(COUNT(DISTINCT session_id), 0), 2) AS avg_events_per_session,

                    -- MONETARY FEATURES (Normalized in INR)
                    COALESCE(ROUND(SUM(norm_price) FILTER (WHERE action = 'buy' AND timestamp >= CAST('{ref_str}' AS TIMESTAMP) - INTERVAL 7 DAY), 2), 0.0) AS total_spend_7d,
                    COALESCE(ROUND(SUM(norm_price) FILTER (WHERE action = 'buy' AND timestamp >= CAST('{ref_str}' AS TIMESTAMP) - INTERVAL 30 DAY), 2), 0.0) AS total_spend_30d,
                    COALESCE(ROUND(SUM(norm_price) FILTER (WHERE action = 'buy'), 2), 0.0) AS total_spend_lifetime,

                    COALESCE(ROUND(AVG(norm_price) FILTER (WHERE action = 'buy'), 2), 0.0) AS avg_order_value,
                    COALESCE(ROUND(MAX(norm_price) FILTER (WHERE action = 'buy'), 2), 0.0) AS max_single_purchase
                FROM eligible_events
                GROUP BY user_id
            )
            SELECT * FROM user_aggs ORDER BY user_id
        """
        result = con.execute(query, params) if params is not None else con.execute(query)
        arrow_table = result.to_arrow_table()
        df = pl.from_arrow(arrow_table)
        return df
    finally:
        if should_close:
            con.close()


def save_rfm_features(df: pl.DataFrame, output_path: Optional[Union[str, Path]] = None) -> Path:
    """Saves computed feature table to Parquet."""
    out_file = Path(output_path) if output_path else DEFAULT_FEATURES_PATH
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_file)
    return out_file


def spot_check_users(df: pl.DataFrame, con: duckdb.DuckDBPyConnection, sample_users: Optional[List[str]] = None) -> None:
    """Prints manual sanity check for 3 users comparing features against raw events."""
    if sample_users is None:
        sample_users = ["user_001", "user_002", "user_005"]

    print("\n--- Spot-Check: Validating 3 Users Against Raw Event History ---")
    for uid in sample_users:
        user_row = df.filter(pl.col("user_id") == uid).to_dicts()
        if not user_row:
            continue
        feat = user_row[0]
        raw_events_count = con.execute(f"SELECT COUNT(*) FROM fact_events WHERE user_id = '{uid}'").fetchone()[0]
        raw_buys_count = con.execute(f"SELECT COUNT(*) FROM fact_events WHERE user_id = '{uid}' AND action = 'buy'").fetchone()[0]

        print(f"\nUser: {uid}")
        print(f"  Feature total_events_lifetime: {feat['total_events_lifetime']} | Raw DB events: {raw_events_count}")
        print(f"  Feature total_purchases_30d:   {feat['total_purchases_30d']} | Raw DB buys:   {raw_buys_count}")
        print(f"  Days since last purchase:      {feat['days_since_last_purchase']}")
        print(f"  Total Spend 30d:               ₹{feat['total_spend_30d']:,.2f}")
        print(f"  Avg Order Value:               ₹{feat['avg_order_value']:,.2f}")


def run_rfm_pipeline(
    reference_date: Optional[str] = None,
    output_path: Optional[str] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Executes the complete RFM feature calculation pipeline."""
    db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    con = db_mgr.get_connection()

    try:
        if reference_date is None:
            # Default to the end of the observed event timeline
            max_ts = con.execute("SELECT MAX(timestamp) FROM fact_events").fetchone()[0]
            if max_ts is None:
                raise ValueError("No events found in fact_events to determine reference date.")
            reference_date = str(max_ts)

        print(f"Computing RFM features relative to reference_date: {reference_date}")
        df = compute_rfm_features(reference_date=reference_date, con=con)
        out_path = save_rfm_features(df, output_path=output_path)
        print(f"Saved {df.height} user records ({df.width} columns) to {out_path}")

        spot_check_users(df, con)

        return {
            "num_users": df.height,
            "num_features": df.width,
            "parquet_path": str(out_path),
            "reference_date": reference_date,
        }
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute leak-free RFM features relative to reference date.")
    parser.add_argument("--ref-date", type=str, default=None, help="Reference date (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    run_rfm_pipeline(reference_date=args.ref_date, output_path=args.output)
