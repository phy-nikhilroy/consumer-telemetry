"""
Sessionization and Cross-App Journey Detection for Consumer Telemetry Warehouse.
Assigns session IDs based on 30-minute inactivity gaps and identifies multi-app product comparisons.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

# Ensure consumer-telemetry root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
from src.config import (
    DEFAULT_DB_PATH,
    DEFAULT_SESSION_INACTIVITY_MINUTES,
    DEFAULT_CROSS_APP_WINDOW_MINUTES,
)
from src.features.database import DatabaseManager


def sessionize_events(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[Union[str, Path]] = None,
    inactivity_minutes: float = DEFAULT_SESSION_INACTIVITY_MINUTES
) -> int:
    """
    Computes session IDs using DuckDB SQL window functions with an inactivity cutoff.
    Updates fact_events.session_id in-place.
    """
    should_close = False
    if con is None:
        db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
        con = db_mgr.get_connection()
        should_close = True

    try:
        # SQL window query to calculate gaps and assign session_id
        update_query = f"""
            UPDATE fact_events
            SET session_id = sub.session_id
            FROM (
                WITH event_gaps AS (
                    SELECT
                        event_id,
                        user_id,
                        timestamp,
                        EPOCH(timestamp - LAG(timestamp) OVER (
                            PARTITION BY user_id ORDER BY timestamp, event_id
                        )) / 60.0 AS minutes_since_last
                    FROM fact_events
                ),
                session_starts AS (
                    SELECT
                        event_id,
                        user_id,
                        timestamp,
                        CASE
                            WHEN minutes_since_last > {inactivity_minutes} OR minutes_since_last IS NULL THEN 1
                            ELSE 0
                        END AS is_new_session
                    FROM event_gaps
                ),
                computed_sessions AS (
                    SELECT
                        event_id,
                        CONCAT(user_id, '_', SUM(is_new_session) OVER (
                            PARTITION BY user_id ORDER BY timestamp, event_id
                        )) AS session_id
                    FROM session_starts
                )
                SELECT event_id, session_id FROM computed_sessions
            ) sub
            WHERE fact_events.event_id = sub.event_id
        """
        con.execute(update_query)

        updated_count = con.execute("SELECT COUNT(*) FROM fact_events WHERE session_id IS NOT NULL").fetchone()[0]
        return updated_count
    finally:
        if should_close:
            con.close()


COMPARISON_COLUMNS = (
    "comparison_id, user_id, session_id, canonical_product_id, primary_app, compared_app, "
    "first_viewed, last_viewed, duration_minutes, price_first_inr, price_last_inr, cheaper_app"
)


def comparison_insert_sql(
    max_window_minutes: float,
    restrict_to_user_product: bool = False,
    id_prefix_sql: str = "CONCAT('cmp_', user_id, '_')",
) -> str:
    """
    Returns the single INSERT ... SELECT that detects cross-app comparisons, shared by the
    batch sessionizer and the serving write path. Events of one (user, product) are ordered
    by (timestamp, event_id); each event with a predecessor on a different app within the
    window yields one row (primary_app is the predecessor's app). With
    restrict_to_user_product the query takes two positional parameters, user_id and
    canonical_product_id, and only reads that user's events for that product.
    """
    where = "WHERE canonical_product_id IS NOT NULL"
    if restrict_to_user_product:
        where += " AND user_id = ? AND canonical_product_id = ?"
    return f"""
        INSERT INTO fact_cross_app_comparisons ({COMPARISON_COLUMNS})
        WITH ordered AS (
            SELECT
                user_id, canonical_product_id, event_id, app_name, timestamp, session_id, price_inr,
                LAG(app_name) OVER w AS prev_app,
                LAG(timestamp) OVER w AS prev_ts,
                LAG(session_id) OVER w AS prev_session,
                LAG(price_inr) OVER w AS prev_price
            FROM fact_events
            {where}
            WINDOW w AS (PARTITION BY user_id, canonical_product_id ORDER BY timestamp, event_id)
        ),
        pairs AS (
            SELECT *, EPOCH(timestamp - prev_ts) / 60.0 AS minutes
            FROM ordered
            WHERE prev_app IS NOT NULL AND prev_app <> app_name
        )
        SELECT
            CONCAT({id_prefix_sql}, ROW_NUMBER() OVER (ORDER BY user_id, canonical_product_id, timestamp, event_id)) AS comparison_id,
            user_id,
            prev_session AS session_id,
            canonical_product_id,
            prev_app AS primary_app,
            app_name AS compared_app,
            prev_ts AS first_viewed,
            timestamp AS last_viewed,
            CAST(ROUND(minutes, 2) AS DECIMAL(10,2)) AS duration_minutes,
            prev_price AS price_first_inr,
            price_inr AS price_last_inr,
            CASE
                WHEN prev_price IS NULL OR price_inr IS NULL OR prev_price = price_inr THEN NULL
                WHEN price_inr < prev_price THEN app_name
                ELSE prev_app
            END AS cheaper_app
        FROM pairs
        WHERE minutes >= 0 AND minutes <= {max_window_minutes}
    """


def detect_cross_app_comparisons(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[Union[str, Path]] = None,
    max_window_minutes: float = DEFAULT_CROSS_APP_WINDOW_MINUTES
) -> Dict[str, Any]:
    """
    Identifies sessions where a user browses the same canonical_product_id across
    distinct shopping apps within max_window_minutes (default 60 minutes).
    Populates fact_cross_app_comparisons.

    The query is built by comparison_insert_sql, shared with the serving path.
    """
    should_close = False
    if con is None:
        db_mgr = DatabaseManager(db_path=db_path or DEFAULT_DB_PATH)
        con = db_mgr.get_connection()
        should_close = True

    try:
        # Ensure table exists
        con.execute("""
            CREATE TABLE IF NOT EXISTS fact_cross_app_comparisons (
                comparison_id VARCHAR PRIMARY KEY,
                user_id VARCHAR NOT NULL,
                session_id VARCHAR,
                canonical_product_id VARCHAR NOT NULL,
                primary_app VARCHAR NOT NULL,
                compared_app VARCHAR NOT NULL,
                first_viewed TIMESTAMP NOT NULL,
                last_viewed TIMESTAMP NOT NULL,
                duration_minutes DECIMAL(10,2) NOT NULL,
                price_first_inr DECIMAL(18,2),
                price_last_inr DECIMAL(18,2),
                cheaper_app VARCHAR
            )
        """)

        # Clean existing comparisons before populating
        con.execute("DELETE FROM fact_cross_app_comparisons")

        insert_query = comparison_insert_sql(max_window_minutes)
        con.execute(insert_query)

        # Compute summary metrics
        total_sessions = con.execute("SELECT COUNT(DISTINCT session_id) FROM fact_events WHERE session_id IS NOT NULL").fetchone()[0]
        cross_app_sessions = con.execute("SELECT COUNT(DISTINCT session_id) FROM fact_cross_app_comparisons WHERE session_id IS NOT NULL").fetchone()[0]
        cross_app_users = con.execute("SELECT COUNT(DISTINCT user_id) FROM fact_cross_app_comparisons").fetchone()[0]
        total_comparisons = con.execute("SELECT COUNT(*) FROM fact_cross_app_comparisons").fetchone()[0]

        summary_message = f"{cross_app_users} users had {cross_app_sessions} cross-app comparison sessions out of {total_sessions} total sessions"
        print(summary_message)

        return {
            "total_sessions": total_sessions,
            "cross_app_sessions": cross_app_sessions,
            "cross_app_users": cross_app_users,
            "total_comparisons": total_comparisons,
            "summary_message": summary_message,
        }
    finally:
        if should_close:
            con.close()


def run_sessionization_pipeline(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Runs end-to-end sessionization and cross-app detection."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    print(f"Running sessionization on {target_path}...")

    db_mgr = DatabaseManager(db_path=target_path)
    con = db_mgr.get_connection()
    try:
        sessionized_events = sessionize_events(con=con)
        print(f"Sessionized {sessionized_events} events.")

        results = detect_cross_app_comparisons(con=con)
        results["sessionized_events"] = sessionized_events
        return results
    finally:
        con.close()


if __name__ == "__main__":
    run_sessionization_pipeline()
