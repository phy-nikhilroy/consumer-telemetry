"""
Re-extraction job: rebuilds every extracted table of a warehouse from `raw_events`.

When the extractor, taxonomy model, or entity matcher changes, the stored captures stay the
same but what was derived from them is stale. `reextract` keeps each raw event (id, user,
timestamp, app, text), clears `fact_events`, `fact_price_observations`, `dim_products`,
`product_aliases` and `dim_users`, and reloads the events in their stored order through the
shared write path (`load_raw_events`) starting from an empty `ProductRegistry`. It then
re-runs sessionization and cross-app comparison detection. The whole rebuild is one
transaction, so a failure leaves the warehouse as it was.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from src.features.event_loader import load_raw_events
from src.features.sessionizer import detect_cross_app_comparisons, sessionize_events
from src.ingestion.entity_matcher import ProductRegistry

DERIVED_TABLES = (
    "fact_events", "fact_price_observations", "product_aliases", "dim_products", "dim_users",
)


def reextract(con: duckdb.DuckDBPyConnection, workers: int = 1) -> Dict[str, int]:
    """
    Rebuilds the extracted tables from `raw_events` and returns the row counts.

    The registry starts empty, so product ids are resolved afresh in stored order. Keys are
    `raw_events`, `fact_events`, `fact_price_observations`, `dim_products`, `dim_users`,
    `sessions`, and `comparisons`.
    """
    raw = con.execute(
        "SELECT event_id, user_id, timestamp, app_name, raw_text FROM raw_events ORDER BY rowid"
    ).pl()
    con.execute("BEGIN TRANSACTION")
    try:
        for table in DERIVED_TABLES:
            con.execute(f"DELETE FROM {table}")
        con.execute("DELETE FROM raw_events")
        counts = load_raw_events(con, raw, workers=workers, registry=ProductRegistry())
        sessionize_events(con=con)
        comparisons = detect_cross_app_comparisons(con=con)
        sessions = con.execute(
            "SELECT COUNT(DISTINCT session_id) FROM fact_events WHERE session_id IS NOT NULL"
        ).fetchone()[0]
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return {
        "raw_events": int(counts["raw_events"]),
        "fact_events": int(counts["fact_events"]),
        "fact_price_observations": int(counts["fact_price_observations"]),
        "dim_products": int(counts["dim_products"]),
        "dim_users": int(counts["dim_users"]),
        "sessions": int(sessions),
        "comparisons": int(comparisons["total_comparisons"]),
    }


if __name__ == "__main__":
    import argparse

    from src.config import DEFAULT_DB_PATH
    from src.features.migrations import migrate

    parser = argparse.ArgumentParser(description="Rebuild extracted warehouse tables from raw_events.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    connection = duckdb.connect(args.db_path)
    try:
        migrate(connection)
        print(reextract(connection, workers=args.workers))
    finally:
        connection.close()
