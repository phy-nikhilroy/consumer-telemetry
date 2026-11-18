"""
Serving-side warehouse access for the API.

Three responsibilities live here:

1. Bootstrap. The API never writes to the batch warehouse the models are trained from.
   It works on its own copy, `SERVING_DB_PATH`, cloned from the batch warehouse on first
   start or simulated from scratch when no batch warehouse exists (as in a fresh
   container). Live demo traffic therefore cannot leak into training data.
2. The live write path. One ingested screen is extracted and written by
   `event_loader.ingest_event`, the same write function the batch loader uses, with a
   session assigned by the batch sessionizer's 30-minute rule; then `dim_users` is updated
   and the cross-app comparisons the event completes are recorded.
3. Read queries for the dashboard, the monitor, and user profiles.

DuckDB allows one read-write process per database file, so every reader (dashboard,
monitor) goes through the API rather than opening the file. Inside the API a single
connection is shared and guarded by a lock, which is ample at demo traffic levels.

The serving clock: the simulated data ends on a fixed date in the past, so scoring
against the wall clock would make every simulated user look weeks dormant. In the
default "simulated" mode the clock starts at the latest event in the warehouse and
advances in real time from there; "wall" mode uses the real UTC time instead.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# Ensure consumer-telemetry root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import polars as pl

from src.features.event_loader import ingest_event, registry_from_warehouse, warehouse_action
from src.features.migrations import migrate
from src.features.sessionizer import comparison_insert_sql

from src.config import (
    DEFAULT_CROSS_APP_WINDOW_MINUTES,
    DEFAULT_DB_PATH,
    DEFAULT_SESSION_INACTIVITY_MINUTES,
    SERVING_DB_PATH,
    SERVING_SIMULATION_DAYS,
    SERVING_SIMULATION_USERS,
    USD_TO_INR_RATE,
)

CLOCK_SIMULATED = "simulated"
CLOCK_WALL = "wall"
LIVE_EVENT_PREFIX = "live_"


def to_naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Converts an aware datetime to naive UTC, the convention of the warehouse columns."""
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def normalized_price_inr(price: Optional[float], currency: Optional[str]) -> Optional[float]:
    """Returns a price in INR, converting USD at `USD_TO_INR_RATE`."""
    if price is None:
        return None
    return float(price) * USD_TO_INR_RATE if currency == "USD" else float(price)


def ensure_serving_warehouse(
    db_path: Optional[Union[str, Path]] = None,
    source_db_path: Optional[Union[str, Path]] = None,
    num_users: int = SERVING_SIMULATION_USERS,
    days: int = SERVING_SIMULATION_DAYS,
    reset: bool = False,
) -> str:
    """
    Makes sure a populated serving warehouse exists at `db_path`.

    Returns how it was obtained: "existing", "copied" (cloned from `source_db_path`), or
    "simulated" (generated with the behavioral simulator and sessionized). `reset`
    discards an existing serving warehouse first, which resets a demo to its initial
    state.
    """
    target = Path(db_path) if db_path else SERVING_DB_PATH
    source = Path(source_db_path) if source_db_path else DEFAULT_DB_PATH
    if target.resolve() == source.resolve():
        raise ValueError("The serving warehouse must not be the batch warehouse itself.")

    if reset:
        for leftover in (target, Path(f"{target}.wal")):
            if leftover.exists():
                leftover.unlink()

    if target.exists():
        con = duckdb.connect(str(target), read_only=True)
        try:
            populated = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'fact_events'"
            ).fetchone()[0] and con.execute("SELECT COUNT(*) FROM fact_events").fetchone()[0]
        finally:
            con.close()
        if populated:
            # An existing serving warehouse may predate the current schema version.
            con = duckdb.connect(str(target))
            try:
                migrate(con)
            finally:
                con.close()
            return "existing"
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)

    if source.exists():
        con = duckdb.connect(str(target))
        try:
            # ATTACH takes no bound parameters, so the path is embedded as an escaped literal.
            source_literal = str(source.resolve()).replace("'", "''")
            con.execute(f"ATTACH '{source_literal}' AS batch_source (READ_ONLY)")
            catalog = con.execute("SELECT current_database()").fetchone()[0]
            con.execute(f'COPY FROM DATABASE batch_source TO "{catalog}"')
            con.execute("DETACH batch_source")
            migrate(con)
        finally:
            con.close()
        return "copied"

    # No batch warehouse, as in a fresh clone or container: simulate one. The simulator
    # is seeded, so this reproduces the population the committed models were trained on.
    from src.features.sessionizer import run_sessionization_pipeline
    from src.ingestion.simulate_users import run_simulation

    run_simulation(
        num_users=num_users,
        days=days,
        db_path=str(target),
        output_parquet=str(target.with_name(f"{target.stem}_simulated_events.parquet")),
    )
    run_sessionization_pipeline(db_path=target)
    return "simulated"


class ServingClock:
    """
    The API's notion of "now", used to stamp events and as the scoring reference date.

    In simulated mode it starts at the latest event in the warehouse and advances in
    real time, so a demo continues the simulated timeline instead of jumping weeks
    ahead of it. In wall mode it is the real UTC time.
    """

    def __init__(self, anchor: Optional[datetime], mode: str = CLOCK_SIMULATED) -> None:
        if mode not in (CLOCK_SIMULATED, CLOCK_WALL):
            raise ValueError(f"Unknown clock mode {mode!r}")
        self.mode = mode
        self.anchor = to_naive_utc(anchor) or datetime.now(timezone.utc).replace(tzinfo=None)
        self._started = time.monotonic()

    def now(self) -> datetime:
        """Returns the current serving time as a naive UTC datetime."""
        if self.mode == CLOCK_WALL:
            return datetime.now(timezone.utc).replace(tzinfo=None)
        return self.anchor + timedelta(seconds=time.monotonic() - self._started)


class ServingWarehouse:
    """A single shared DuckDB connection to the serving warehouse, guarded by a lock."""

    def __init__(self, db_path: Union[str, Path], clock_mode: str = CLOCK_SIMULATED) -> None:
        self.db_path = Path(db_path)
        self.con = duckdb.connect(str(self.db_path))
        self.lock = threading.RLock()
        self._registry: Optional[Any] = None
        self._latest = self._query_latest_timestamp()
        self.clock = ServingClock(self._latest, mode=clock_mode)

    def close(self) -> None:
        """Closes the underlying connection."""
        with self.lock:
            self.con.close()

    # ------------------------------------------------------------------ helpers

    def frame(self, query: str, params: Optional[Any] = None) -> pl.DataFrame:
        """Runs a query under the lock and returns a Polars frame."""
        with self.lock:
            result = self.con.execute(query, params) if params is not None else self.con.execute(query)
            return pl.from_arrow(result.to_arrow_table())

    def scalar(self, query: str, params: Optional[Any] = None) -> Any:
        """Runs a query under the lock and returns the first column of the first row."""
        with self.lock:
            result = self.con.execute(query, params) if params is not None else self.con.execute(query)
            row = result.fetchone()
            return row[0] if row else None

    def _query_latest_timestamp(self) -> Optional[datetime]:
        row = self.con.execute("SELECT MAX(timestamp) FROM fact_events").fetchone()
        return row[0] if row else None

    def latest_event_timestamp(self) -> Optional[datetime]:
        """Returns the timestamp of the latest event in the warehouse."""
        return self._latest

    def reference_date(self) -> datetime:
        """
        Returns the scoring reference date: the serving clock, or the latest event if an
        explicitly timestamped event lies beyond it, so no stored event is ignored.
        """
        now = self.clock.now()
        return max(now, self._latest) if self._latest is not None else now

    # ------------------------------------------------------------------ writes

    def _assign_session(self, user_id: str, ts: datetime) -> str:
        """
        Applies the batch sessionizer's rule to one new event: it joins the session of
        the user's previous event when that event is at most the inactivity window
        earlier, and otherwise opens the user's next session number.
        """
        previous = self.con.execute(
            """
            SELECT session_id, timestamp FROM fact_events
            WHERE user_id = ? AND timestamp <= ?
            ORDER BY timestamp DESC, event_id DESC
            LIMIT 1
            """,
            [user_id, ts],
        ).fetchone()
        if previous is not None and previous[0] is not None:
            gap_minutes = (ts - previous[1]).total_seconds() / 60.0
            if gap_minutes <= DEFAULT_SESSION_INACTIVITY_MINUTES:
                return str(previous[0])

        prefix = f"{user_id}_"
        highest = self.con.execute(
            """
            SELECT MAX(TRY_CAST(SUBSTRING(session_id, ?) AS INTEGER))
            FROM fact_events
            WHERE user_id = ? AND starts_with(session_id, ?)
            """,
            [len(prefix) + 1, user_id, prefix],
        ).fetchone()[0]
        return f"{prefix}{(highest or 0) + 1}"

    def _record_comparisons(
        self,
        event_id: str,
        user_id: str,
        product_id: str,
        app_name: str,
        ts: datetime,
    ) -> int:
        """
        Rebuilds this user's comparisons for the product with the batch detector's SQL
        restricted to that user and product (exact for out-of-order timestamps). Returns
        the number of comparisons the new event takes part in: rows where it is the later
        event (last_viewed = ts, compared_app = app_name) plus rows where it is the earlier
        event (first_viewed = ts, primary_app = app_name).
        """
        self.con.execute(
            "DELETE FROM fact_cross_app_comparisons WHERE user_id = ? AND canonical_product_id = ?",
            [user_id, product_id],
        )
        prefix = f"cmp_{user_id}_{LIVE_EVENT_PREFIX}{uuid.uuid4().hex[:12]}_"
        escaped = prefix.replace("'", "''")
        self.con.execute(
            comparison_insert_sql(
                DEFAULT_CROSS_APP_WINDOW_MINUTES,
                restrict_to_user_product=True,
                id_prefix_sql=f"'{escaped}'",
            ),
            [user_id, product_id],
        )
        return self.con.execute(
            """
            SELECT COUNT(*) FROM fact_cross_app_comparisons
            WHERE user_id = ? AND canonical_product_id = ?
              AND ((last_viewed = ? AND compared_app = ?) OR (first_viewed = ? AND primary_app = ?))
            """,
            [user_id, product_id, ts, app_name, ts, app_name],
        ).fetchone()[0]

    @property
    def registry(self) -> Any:
        """The product registry, rebuilt from `dim_products` on first use."""
        if self._registry is None:
            with self.lock:
                if self._registry is None:
                    self._registry = registry_from_warehouse(self.con)
        return self._registry

    def record_event(
        self,
        user_id: str,
        app_name: str,
        raw_text: str,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Extracts one raw screen and stores it; returns what was written.

        The screen goes through `event_loader.ingest_event`, the batch loader's write
        function, so `raw_events`, `fact_events`, `fact_price_observations`, and
        `dim_products` receive exactly the rows a batch load would write. The whole write is
        one transaction, so a failure leaves no partial event, session, or comparison
        behind. `product` in the result is the focal `ExtractedProduct` (None when the
        screen held no text) and `products` every listing on the screen.
        """
        ts = to_naive_utc(timestamp) or self.clock.now()
        event_id = f"{LIVE_EVENT_PREFIX}{uuid.uuid4().hex[:16]}"
        registry = self.registry

        with self.lock:
            self.con.execute("BEGIN TRANSACTION")
            try:
                is_new_user = self.con.execute(
                    "SELECT COUNT(*) = 0 FROM dim_users WHERE user_id = ?", [user_id]
                ).fetchone()[0]
                session_id = self._assign_session(user_id, ts)

                products, _ = ingest_event(
                    self.con,
                    {
                        "event_id": event_id, "user_id": user_id, "timestamp": ts,
                        "app_name": app_name, "raw_text": raw_text, "session_id": session_id,
                    },
                    registry=registry,
                )
                focal = next((p for p in products if p.is_focal), products[0] if products else None)
                action = warehouse_action(str(focal.action)) if focal is not None else "view"

                if is_new_user:
                    self.con.execute(
                        "INSERT INTO dim_users (user_id, first_seen, primary_app, total_events) VALUES (?, ?, ?, 1)",
                        [user_id, ts, app_name],
                    )
                else:
                    self.con.execute(
                        """
                        UPDATE dim_users SET
                            total_events = total_events + 1,
                            first_seen = LEAST(first_seen, ?),
                            primary_app = (SELECT MODE(app_name) FROM fact_events WHERE user_id = ?)
                        WHERE user_id = ?
                        """,
                        [ts, user_id, user_id],
                    )

                comparisons = 0
                if focal is not None and focal.canonical_product_id:
                    comparisons = self._record_comparisons(
                        event_id, user_id, focal.canonical_product_id, app_name, ts
                    )
                self.con.execute("COMMIT")
            except Exception:
                self.con.execute("ROLLBACK")
                raise

            if self._latest is None or ts > self._latest:
                self._latest = ts

        return {
            "event_id": event_id,
            "user_id": user_id,
            "timestamp": ts,
            "app_name": app_name,
            "stored_action": action,
            "session_id": session_id,
            "is_new_user": bool(is_new_user),
            "cross_app_comparisons_detected": comparisons,
            "product": focal,
            "products": products,
        }

    # ------------------------------------------------------------------ reads

    def counts(self) -> Dict[str, int]:
        """Returns row counts for the main tables and the number of live events."""
        with self.lock:
            row = self.con.execute(
                f"""
                SELECT
                    (SELECT COUNT(*) FROM fact_events),
                    (SELECT COUNT(*) FROM dim_users),
                    (SELECT COUNT(*) FROM dim_products),
                    (SELECT COUNT(*) FROM fact_events WHERE starts_with(event_id, '{LIVE_EVENT_PREFIX}'))
                """
            ).fetchone()
        return {"events": row[0], "users": row[1], "products": row[2], "live_events": row[3]}

    def user_exists(self, user_id: str) -> bool:
        """Returns whether the user has at least one event."""
        return bool(self.scalar("SELECT COUNT(*) > 0 FROM fact_events WHERE user_id = ?", [user_id]))

    def list_users(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns users ordered by most recent activity."""
        return self.frame(
            """
            SELECT user_id, COUNT(*) AS events, MAX(timestamp) AS last_seen
            FROM fact_events
            GROUP BY user_id
            ORDER BY last_seen DESC, user_id
            LIMIT ?
            """,
            [limit],
        ).to_dicts()

    def recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the latest events with their product attributes, newest first."""
        return self.frame(
            """
            SELECT
                fe.event_id, fe.user_id, fe.timestamp, fe.app_name, fe.action,
                CAST(fe.price AS DOUBLE) AS price, fe.currency, fe.session_id,
                fe.canonical_product_id, dp.brand, dp.product_title, dp.category_l2
            FROM fact_events fe
            LEFT JOIN dim_products dp ON fe.canonical_product_id = dp.canonical_product_id
            ORDER BY fe.timestamp DESC, fe.event_id DESC
            LIMIT ?
            """,
            [limit],
        ).to_dicts()

    def activity_summary(self, window_hours: float = 24.0) -> Dict[str, Any]:
        """Returns event and user activity in the trailing window, and the last purchase."""
        now = self.reference_date()
        since = now - timedelta(hours=window_hours)
        with self.lock:
            window = self.con.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(*) FILTER (WHERE action = 'buy')
                FROM fact_events WHERE timestamp > ? AND timestamp <= ?
                """,
                [since, now],
            ).fetchone()
            last_purchase = self.con.execute(
                """
                SELECT fe.user_id, fe.timestamp, fe.app_name, dp.product_title,
                       CAST(fe.price AS DOUBLE), fe.currency
                FROM fact_events fe
                LEFT JOIN dim_products dp ON fe.canonical_product_id = dp.canonical_product_id
                WHERE fe.action = 'buy'
                ORDER BY fe.timestamp DESC, fe.event_id DESC
                LIMIT 1
                """
            ).fetchone()

        purchase = None
        if last_purchase is not None:
            purchase = {
                "user_id": last_purchase[0],
                "timestamp": last_purchase[1],
                "app_name": last_purchase[2],
                "product_title": last_purchase[3],
                "price": last_purchase[4],
                "currency": last_purchase[5],
            }
        return {
            "reference_date": now,
            "window_hours": window_hours,
            "events_in_window": window[0],
            "active_users_in_window": window[1],
            "purchases_in_window": window[2],
            "last_purchase": purchase,
            **self.counts(),
        }

    def user_activity(self, user_id: str, reference_date: datetime) -> Dict[str, Any]:
        """Returns a user's totals, app usage, and top categories up to the reference date."""
        params = [user_id, reference_date]
        with self.lock:
            totals = self.con.execute(
                f"""
                SELECT
                    COUNT(*),
                    COUNT(*) FILTER (WHERE action = 'buy'),
                    COALESCE(SUM(CASE WHEN currency = 'USD' THEN price * {USD_TO_INR_RATE} ELSE price END)
                             FILTER (WHERE action = 'buy'), 0),
                    MIN(timestamp),
                    MAX(timestamp),
                    COUNT(DISTINCT session_id)
                FROM fact_events
                WHERE user_id = ? AND timestamp <= ?
                """,
                params,
            ).fetchone()
            apps = self.con.execute(
                """
                SELECT app_name, COUNT(*) AS events, COUNT(*) FILTER (WHERE action = 'buy') AS purchases
                FROM fact_events
                WHERE user_id = ? AND timestamp <= ?
                GROUP BY app_name
                ORDER BY events DESC, app_name
                """,
                params,
            ).fetchall()
            categories = self.con.execute(
                """
                SELECT COALESCE(dp.category_l1, 'Unknown'), COALESCE(dp.category_l2, 'Unknown'), COUNT(*) AS events
                FROM fact_events fe
                LEFT JOIN dim_products dp ON fe.canonical_product_id = dp.canonical_product_id
                WHERE fe.user_id = ? AND fe.timestamp <= ?
                GROUP BY 1, 2
                ORDER BY events DESC, 2
                LIMIT 3
                """,
                params,
            ).fetchall()

        total_events = int(totals[0])
        return {
            "total_events": total_events,
            "total_purchases": int(totals[1]),
            "total_spend_inr": round(float(totals[2]), 2),
            "first_seen": totals[3],
            "last_seen": totals[4],
            "sessions": int(totals[5]),
            "app_usage": [
                {
                    "app_name": name,
                    "events": int(events),
                    "purchases": int(purchases),
                    "share": round(events / total_events, 4) if total_events else 0.0,
                }
                for name, events, purchases in apps
            ],
            "top_categories": [
                {"category_l1": l1, "category_l2": l2, "events": int(events)}
                for l1, l2, events in categories
            ],
        }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the API's serving warehouse.")
    parser.add_argument("--db-path", type=str, default=None, help="Serving warehouse (default: data/processed/serving.duckdb)")
    parser.add_argument("--source-db-path", type=str, default=None, help="Batch warehouse to clone (default: data/processed/consumer.duckdb)")
    parser.add_argument("--reset", action="store_true", help="Discard the serving warehouse and rebuild it")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    outcome = ensure_serving_warehouse(
        db_path=args.db_path, source_db_path=args.source_db_path, reset=args.reset
    )
    target = args.db_path or SERVING_DB_PATH
    con = duckdb.connect(str(target), read_only=True)
    try:
        events, users = con.execute("SELECT COUNT(*), COUNT(DISTINCT user_id) FROM fact_events").fetchone()
    finally:
        con.close()
    print(f"Serving warehouse {outcome}: {target} ({events} events, {users} users)")
