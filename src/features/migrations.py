"""
Brings a DuckDB warehouse up to the current schema version.

`migrate(con)` is the single upgrade path for every warehouse: a fresh one, a batch
warehouse built under schema version 1, a serving warehouse cloned from one, and the
RetailRocket warehouse. It is idempotent, so callers run it unconditionally on open.

Steps, each safe to repeat:

1. Run `schema.sql`, which creates any missing table at the current version. Tables that
   already exist are left as they are.
2. Add the columns that schema version 1 tables lack, with
   `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`. Every added column is nullable.
3. Backfill `fact_events.price_inr` from `price` and `currency` where it is null.
4. Record the version in `schema_info`.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb

from src.config import SCHEMA_SQL_PATH, SCHEMA_VERSION, USD_TO_INR_RATE

# Columns added to schema version 1 tables, in the order they are added. schema.sql
# declares the same columns for a fresh warehouse; tests/test_migrations.py checks that a
# migrated v1 warehouse and a fresh one end with the same columns.
ADDED_COLUMNS: Dict[str, List[Tuple[str, str]]] = {
    "fact_events": [
        ("mrp", "DECIMAL(18,2)"),
        ("discount_pct", "INTEGER"),
        ("rating", "DOUBLE"),
        ("extraction_confidence", "DOUBLE"),
        ("price_inr", "DECIMAL(18,2)"),
    ],
    "fact_cross_app_comparisons": [
        ("price_first_inr", "DECIMAL(18,2)"),
        ("price_last_inr", "DECIMAL(18,2)"),
        ("cheaper_app", "VARCHAR"),
    ],
    "dim_products": [
        ("first_seen", "TIMESTAMP"),
    ],
}


def price_inr_sql(price_col: str = "price", currency_col: str = "currency") -> str:
    """SQL expression converting a price to INR at `USD_TO_INR_RATE`, as DECIMAL(18,2)."""
    return (
        f"CAST(CASE WHEN {currency_col} = 'USD' THEN {price_col} * {USD_TO_INR_RATE} "
        f"ELSE {price_col} END AS DECIMAL(18,2))"
    )


def schema_version(con: duckdb.DuckDBPyConnection) -> str | None:
    """Returns the recorded schema version, or None when the warehouse has none."""
    has_table = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'schema_info'"
    ).fetchone()[0]
    if not has_table:
        return None
    row = con.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
    return row[0] if row else None


def migrate(con: duckdb.DuckDBPyConnection) -> str:
    """
    Upgrades the warehouse on `con` to `SCHEMA_VERSION` and returns that version.

    Idempotent: a second call finds every table and column present and no null
    `price_inr` left to backfill, so it changes nothing.
    """
    con.execute(SCHEMA_SQL_PATH.read_text(encoding="utf-8"))

    for table, columns in ADDED_COLUMNS.items():
        for name, sql_type in columns:
            con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {sql_type}")

    con.execute(
        f"""
        UPDATE fact_events
        SET price_inr = {price_inr_sql()}
        WHERE price_inr IS NULL AND price IS NOT NULL
        """
    )

    con.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('schema_version', ?)",
        [SCHEMA_VERSION],
    )
    return SCHEMA_VERSION


if __name__ == "__main__":
    import argparse

    from src.config import DEFAULT_DB_PATH

    parser = argparse.ArgumentParser(description="Upgrade a warehouse to the current schema version.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()
    connection = duckdb.connect(args.db_path)
    try:
        print(f"{args.db_path}: schema version {migrate(connection)}")
    finally:
        connection.close()
