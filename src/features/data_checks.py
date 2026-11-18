"""Warehouse integrity checks: keys, domains, referential links, and comparison sanity.

`run_checks(con)` returns one `CheckResult` per check. The CLI exits 1 when any check fails.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DEFAULT_CROSS_APP_WINDOW_MINUTES, DEFAULT_DB_PATH, WAREHOUSE_ACTIONS

PRIMARY_KEYS = {
    "raw_events": "event_id",
    "fact_events": "event_id",
    "fact_price_observations": "observation_id",
    "dim_products": "canonical_product_id",
    "dim_users": "user_id",
    "fact_cross_app_comparisons": "comparison_id",
}


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check: its name, whether it passed, and the violating row count."""

    name: str
    passed: bool
    failing_rows: int


def _tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0])


def run_checks(con: duckdb.DuckDBPyConnection) -> List[CheckResult]:
    """Run every warehouse check and return the results; missing tables are skipped."""
    tables = _tables(con)
    results: List[CheckResult] = []

    def add(name: str, sql: str) -> None:
        n = _count(con, sql)
        results.append(CheckResult(name, n == 0, n))

    for table, key in PRIMARY_KEYS.items():
        if table in tables:
            add(
                f"pk_unique:{table}",
                f"SELECT COALESCE(SUM(c), 0) FROM (SELECT COUNT(*) - 1 AS c FROM {table} "
                f"GROUP BY {key} HAVING COUNT(*) > 1) t",
            )
    if "fact_events" in tables:
        actions = ", ".join(f"'{a}'" for a in WAREHOUSE_ACTIONS)
        add("action_in_warehouse_actions",
            f"SELECT COUNT(*) FROM fact_events WHERE action IS NULL OR action NOT IN ({actions})")
        if "raw_events" in tables and _count(con, "SELECT COUNT(*) FROM raw_events") > 0:
            add("event_id_in_raw_events",
                "SELECT COUNT(*) FROM fact_events WHERE event_id NOT IN (SELECT event_id FROM raw_events)")
        if "dim_products" in tables:
            add("product_id_in_dim_products",
                "SELECT COUNT(*) FROM fact_events WHERE canonical_product_id IS NOT NULL "
                "AND canonical_product_id NOT IN (SELECT canonical_product_id FROM dim_products)")
        if "dim_users" in tables:
            add("user_id_in_dim_users",
                "SELECT COUNT(*) FROM fact_events WHERE user_id NOT IN (SELECT user_id FROM dim_users)")
        add("price_inr_positive",
            "SELECT COUNT(*) FROM fact_events WHERE price_inr IS NOT NULL AND price_inr <= 0")
        add("discount_pct_in_range",
            "SELECT COUNT(*) FROM fact_events WHERE discount_pct IS NOT NULL "
            "AND (discount_pct < 0 OR discount_pct > 99)")
        add("session_id_not_null", "SELECT COUNT(*) FROM fact_events WHERE session_id IS NULL")
    if "fact_cross_app_comparisons" in tables:
        add("comparison_order",
            "SELECT COUNT(*) FROM fact_cross_app_comparisons WHERE first_viewed > last_viewed")
        add("comparison_duration_in_window",
            "SELECT COUNT(*) FROM fact_cross_app_comparisons WHERE duration_minutes < 0 "
            f"OR duration_minutes > {float(DEFAULT_CROSS_APP_WINDOW_MINUTES)}")
    return results


def main() -> int:
    """CLI entry point; returns 1 when any check fails."""
    parser = argparse.ArgumentParser(description="Run warehouse data checks.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()
    con = duckdb.connect(str(args.db_path), read_only=True)
    try:
        results = run_checks(con)
    finally:
        con.close()
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name}  failing_rows={r.failing_rows}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
