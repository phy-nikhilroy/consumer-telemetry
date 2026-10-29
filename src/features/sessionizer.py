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
from src.features.database import DatabaseManager, DEFAULT_DB_PATH


def sessionize_events(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[Union[str, Path]] = None,
    inactivity_minutes: float = 30.0
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


def detect_cross_app_comparisons(
    con: Optional[duckdb.DuckDBPyConnection] = None,
    db_path: Optional[Union[str, Path]] = None,
    max_window_minutes: float = 60.0
) -> Dict[str, Any]:
    """
    Identifies sessions where a user browses the same canonical_product_id across
    distinct shopping apps within max_window_minutes (default 60 minutes).
    Populates fact_cross_app_comparisons.
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
                duration_minutes DECIMAL(10,2) NOT NULL
            )
        """)

        # Clean existing comparisons before populating
        con.execute("DELETE FROM fact_cross_app_comparisons")

        insert_query = f"""
            INSERT INTO fact_cross_app_comparisons (
                comparison_id,
                user_id,
                session_id,
                canonical_product_id,
                primary_app,
                compared_app,
                first_viewed,
                last_viewed,
                duration_minutes
            )
            WITH pairs AS (
                SELECT
                    e1.user_id,
                    e1.session_id,
                    e1.canonical_product_id,
                    e1.app_name AS primary_app,
                    e2.app_name AS compared_app,
                    e1.timestamp AS first_viewed,
                    e2.timestamp AS last_viewed,
                    ROUND(EPOCH(e2.timestamp - e1.timestamp) / 60.0, 2) AS duration_minutes,
                    ROW_NUMBER() OVER (
                        PARTITION BY e1.user_id, e1.canonical_product_id, e1.timestamp
                        ORDER BY e2.timestamp
                    ) AS rn
                FROM fact_events e1
                JOIN fact_events e2
                  ON e1.user_id = e2.user_id
                 AND e1.canonical_product_id = e2.canonical_product_id
                 AND e1.app_name < e2.app_name
                 AND e2.timestamp >= e1.timestamp
                 AND EPOCH(e2.timestamp - e1.timestamp) / 60.0 <= {max_window_minutes}
            )
            SELECT
                CONCAT('cmp_', user_id, '_', ROW_NUMBER() OVER ()) AS comparison_id,
                user_id,
                session_id,
                canonical_product_id,
                primary_app,
                compared_app,
                first_viewed,
                last_viewed,
                CAST(duration_minutes AS DECIMAL(10,2)) AS duration_minutes
            FROM pairs
            WHERE rn = 1
        """
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
