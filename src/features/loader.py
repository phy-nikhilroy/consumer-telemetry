"""
Loads extracted product telemetry into DuckDB star schema tables.
"""

import sys
from pathlib import Path
import random
from typing import Optional, Union

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import polars as pl

from src.config import DEFAULT_DB_PATH, PROCESSED_DATA_DIR, USD_TO_INR_RATE
from src.features.database import DatabaseManager

DEFAULT_PARQUET_PATH = PROCESSED_DATA_DIR / "extracted_products.parquet"


def load_telemetry_to_warehouse(
    parquet_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Union[str, Path]] = None,
    num_users: int = 50,
    seed: int = 42
) -> dict:
    """
    Reads extracted product events from Parquet, assigns synthetic user IDs,
    and populates fact_events, dim_products, and dim_users in DuckDB.
    """
    p_path = Path(parquet_path) if parquet_path else DEFAULT_PARQUET_PATH
    if not p_path.exists():
        raise FileNotFoundError(f"Parquet source file not found at {p_path}")

    db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
    db_mgr.init_schema()

    df = pl.read_parquet(p_path)
    total_events = df.height

    # Assign synthetic user IDs (50 users with varying activity levels)
    random.seed(seed)
    user_pool = [f"user_{i:03d}" for i in range(1, num_users + 1)]
    # Use Zipfian-like distribution for realistic activity skew
    weights = [1.0 / (i + 1) ** 0.8 for i in range(num_users)]
    assigned_users = random.choices(user_pool, weights=weights, k=total_events)

    event_ids = [f"evt_{i + 1:06d}" for i in range(total_events)]

    # Prepare Polars DataFrame with assigned user and event IDs
    events_df = df.with_columns([
        pl.Series("event_id", event_ids),
        pl.Series("user_id", assigned_users),
        pl.col("app_source").alias("app_name"),
        pl.lit(None, dtype=pl.String).alias("session_id")
    ])

    con = db_mgr.get_connection()
    try:
        # Register events_df into DuckDB session
        con.register("raw_events_view", events_df.to_arrow())

        # 1. Populate fact_events
        con.execute("""
            INSERT OR REPLACE INTO fact_events (
                event_id,
                user_id,
                timestamp,
                app_name,
                canonical_product_id,
                action,
                price,
                currency,
                session_id
            )
            SELECT
                event_id,
                user_id,
                CAST(timestamp AS TIMESTAMP) AS timestamp,
                app_name,
                canonical_product_id,
                action,
                CAST(price AS DECIMAL(10,2)) AS price,
                currency,
                session_id
            FROM raw_events_view
        """)

        # 2. Populate dim_products
        con.execute(f"""
            INSERT OR REPLACE INTO dim_products (
                canonical_product_id,
                brand,
                product_title,
                category_l1,
                category_l2,
                avg_price
            )
            SELECT
                canonical_product_id,
                FIRST(brand) FILTER (WHERE brand IS NOT NULL) AS brand,
                FIRST(canonical_name) FILTER (WHERE canonical_name IS NOT NULL) AS product_title,
                FIRST(category_l1) FILTER (WHERE category_l1 IS NOT NULL) AS category_l1,
                FIRST(category_l2) FILTER (WHERE category_l2 IS NOT NULL) AS category_l2,
                -- Normalize currency before averaging. Averaging raw prices across a
                -- mix of INR and USD listings understates the true average by a factor
                -- of roughly two, and because the behavioral feature layer compares a
                -- currency-normalized price against this column, an unnormalized value
                -- made almost every purchase look like it was above average price,
                -- silently zeroing out the discount features for 96 percent of users.
                CAST(AVG(
                    CASE WHEN currency = 'USD' THEN price * {USD_TO_INR_RATE} ELSE price END
                ) AS DECIMAL(12,2)) AS avg_price
            FROM raw_events_view
            WHERE canonical_product_id IS NOT NULL
            GROUP BY canonical_product_id
        """)

        # 3. Populate dim_users
        con.execute("""
            INSERT OR REPLACE INTO dim_users (
                user_id,
                first_seen,
                primary_app,
                total_events
            )
            SELECT
                user_id,
                MIN(CAST(timestamp AS TIMESTAMP)) AS first_seen,
                MODE(app_name) AS primary_app,
                COUNT(event_id) AS total_events
            FROM raw_events_view
            GROUP BY user_id
        """)

        # Compute summary counts
        fact_count = con.execute("SELECT COUNT(*) FROM fact_events").fetchone()[0]
        prod_count = con.execute("SELECT COUNT(*) FROM dim_products").fetchone()[0]
        user_count = con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]

        return {
            "fact_events": fact_count,
            "dim_products": prod_count,
            "dim_users": user_count
        }
    finally:
        con.close()


if __name__ == "__main__":
    result = load_telemetry_to_warehouse()
    print("Warehouse loaded successfully:")
    for table, count in result.items():
        print(f"  {table}: {count} records")
