"""
Loads raw screen events into the warehouse with extraction in the loop.

The warehouse keeps what was captured (`raw_events`) and everything extraction produced from
it: one `fact_events` row per screen from its focal listing, one `fact_price_observations`
row per priced listing on the screen, and the `dim_products` rows those listings resolve to.

There is one write path. `write_events` stores already extracted events, and both the batch
loader (`load_raw_events`) and the serving API's `/ingest` (`ServingWarehouse.record_event`
through `ingest_event`) call it, so batch and live writes cannot diverge.

Extraction is split from resolution so that `workers > 1` gives the same tables as
`workers = 1`. Workers run `extract_screens` with no registry, which yields each listing's
exact v1.0 key. The parent then resolves every listing through the `ProductRegistry` in input
order, which is what `process(..., registry=...)` would have done in one process, and writes
in input order.
"""

from __future__ import annotations

import atexit
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import polars as pl

from src.config import (
    EVENT_LOADER_CHUNK_SIZE,
    EVENT_LOADER_MEMO_SIZE,
    EVENT_LOADER_TASK_SIZE,
    EXTRACTOR_VERSION,
    WAREHOUSE_ACTIONS,
)
from src.features.migrations import price_inr_sql

REQUIRED_COLUMNS = ("event_id", "user_id", "timestamp", "app_name", "raw_text")

# The fields of an extracted listing that the write path and the registry need. Workers
# return only these, which keeps the inter-process payload small.
PRODUCT_FIELDS = (
    "raw_text", "brand", "product_title", "specs", "canonical_name", "canonical_product_id",
    "category_l1", "category_l2", "price", "currency", "mrp", "discount_pct", "rating",
    "extraction_confidence", "action", "is_focal",
)

_RAW_SCHEMA = {
    "event_id": pl.Utf8, "user_id": pl.Utf8, "timestamp": pl.Datetime("us"),
    "app_name": pl.Utf8, "raw_text": pl.Utf8, "extractor_version": pl.Utf8,
}
_FACT_SCHEMA = {
    "event_id": pl.Utf8, "user_id": pl.Utf8, "timestamp": pl.Datetime("us"), "app_name": pl.Utf8,
    "canonical_product_id": pl.Utf8, "action": pl.Utf8, "price": pl.Float64, "currency": pl.Utf8,
    "session_id": pl.Utf8, "mrp": pl.Float64, "discount_pct": pl.Int32, "rating": pl.Float64,
    "extraction_confidence": pl.Float64,
}
_OBS_SCHEMA = {
    "observation_id": pl.Utf8, "source_event_id": pl.Utf8, "user_id": pl.Utf8,
    "timestamp": pl.Datetime("us"), "app_name": pl.Utf8, "canonical_product_id": pl.Utf8,
    "price": pl.Float64, "currency": pl.Utf8, "discount_pct": pl.Int32, "is_focal": pl.Boolean,
}
_ALIAS_SCHEMA = {
    "alias_key": pl.Utf8, "canonical_product_id": pl.Utf8, "source_observation_id": pl.Utf8,
}
_PRODUCT_SCHEMA = {
    "canonical_product_id": pl.Utf8, "brand": pl.Utf8, "product_title": pl.Utf8,
    "category_l1": pl.Utf8, "category_l2": pl.Utf8, "first_seen": pl.Datetime("us"),
}


def warehouse_action(extracted_action: str) -> str:
    """Maps an extracted intent onto the warehouse funnel of view, cart, and buy."""
    return extracted_action if extracted_action in WAREHOUSE_ACTIONS else "view"


def _naive(value: Any) -> datetime:
    """A timestamp as a naive UTC datetime, the convention of the warehouse columns."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is not None:
        from datetime import timezone

        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def observation_id(event_id: str, position: int) -> str:
    """The id of the price observation of the listing at `position` on screen `event_id`."""
    return f"{event_id}#{position}"


# ---------------------------------------------------------------------------------------
# Extraction (runs in workers) and resolution (runs in the parent, in input order)
# ---------------------------------------------------------------------------------------

def _slim(product: Any) -> Dict[str, Any]:
    """The `PRODUCT_FIELDS` of an `ExtractedProduct`."""
    data = product.model_dump() if hasattr(product, "model_dump") else dict(product)
    return {field: data.get(field) for field in PRODUCT_FIELDS}


def extract_screens(
    raw_texts: Sequence[str],
    app_names: Sequence[str],
    timestamps: Sequence[datetime],
) -> List[List[Any]]:
    """
    Extracts every screen, equal to `[process_screen(t, a, ts) for ...]` with no registry.

    Every segment of every screen goes through one `process_batch` call, which batches the
    spaCy and taxonomy work, and the focal listing is marked by the `process_screen` rule.
    Returns one list of `ExtractedProduct` per screen.
    """
    from src.ingestion.pipeline import FOCAL_ACTIONS, process_batch
    from src.ingestion.segmenter import segment_screen

    segments: List[str] = []
    seg_apps: List[str] = []
    seg_times: List[datetime] = []
    bounds: List[Tuple[int, int]] = []
    for text, app, ts in zip(raw_texts, app_names, timestamps):
        start = len(segments)
        for segment in segment_screen(text):
            segments.append(segment)
            seg_apps.append(app)
            seg_times.append(ts)
        bounds.append((start, len(segments)))

    products = process_batch(segments, app_sources=seg_apps, timestamps=seg_times) if segments else []
    screens: List[List[Any]] = []
    for start, end in bounds:
        screen = products[start:end]
        if screen:
            focal = next((i for i, p in enumerate(screen) if p.action in FOCAL_ACTIONS), 0)
            screen[focal].is_focal = True
        screens.append(screen)
    return screens


def _extract_task(task: Tuple[List[str], List[str], List[datetime]]) -> List[List[Dict[str, Any]]]:
    """Worker entry point: extracts one task of screens and returns slim product dicts."""
    texts, apps, times = task
    return [[_slim(p) for p in screen] for screen in extract_screens(texts, apps, times)]


def _init_worker(threads: int) -> None:
    """Limits each worker's math libraries so that workers do not oversubscribe the cores."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:  # pragma: no cover - torch is a hard dependency of the encoders
        pass


# Extracted screens by (raw text, app), most recent last. Extraction is deterministic and the
# memo holds listings before resolution and without timestamps, so a hit equals a fresh
# extraction. It saves re-extracting identical screens within one process, as when the same
# seed is simulated twice; it is bounded, so a large load only keeps its tail.
_SCREEN_MEMO: "OrderedDict[Tuple[str, str], List[Dict[str, Any]]]" = OrderedDict()


def _memo_get(text: str, app: str) -> Optional[List[Dict[str, Any]]]:
    hit = _SCREEN_MEMO.get((text, app))
    if hit is None:
        return None
    _SCREEN_MEMO.move_to_end((text, app))
    return [dict(p) for p in hit]


def _memo_put(text: str, app: str, products: List[Dict[str, Any]]) -> None:
    _SCREEN_MEMO[(text, app)] = [dict(p) for p in products]
    while len(_SCREEN_MEMO) > EVENT_LOADER_MEMO_SIZE:
        _SCREEN_MEMO.popitem(last=False)


# One pool per worker count, kept for the life of the process: each worker loads spaCy,
# the sentence encoder, and the taxonomy model once, and later loads reuse them.
_EXECUTORS: Dict[int, ProcessPoolExecutor] = {}


def _executor(workers: int) -> ProcessPoolExecutor:
    """The process pool of `workers` extraction processes, created on first use."""
    if workers not in _EXECUTORS:
        threads = max(1, (os.cpu_count() or 1) // workers)
        _EXECUTORS[workers] = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(threads,),
        )
    return _EXECUTORS[workers]


@atexit.register
def _shutdown_executors() -> None:
    for pool in _EXECUTORS.values():
        pool.shutdown(cancel_futures=True)
    _EXECUTORS.clear()


def _registry_record(product: Mapping[str, Any]) -> Dict[str, Any]:
    """The record `process` hands to `ProductRegistry.resolve` for one listing."""
    return {
        "raw_text": product.get("raw_text"),
        "brand": product.get("brand"),
        "product_title": product.get("product_title"),
        "specs": product.get("specs"),
        "price": product.get("price"),
        "currency": product.get("currency"),
        "canonical_product_id": product.get("canonical_product_id"),
    }


def resolve_products(products: List[Dict[str, Any]], registry: Any) -> List[Dict[str, Any]]:
    """
    Replaces each listing's exact key with its registry id, in screen order. The exact key
    is kept as `exact_key`, which `write_events` records in `product_aliases`.
    """
    for product in products:
        product["exact_key"] = product.get("canonical_product_id")
        if registry is not None:
            product["canonical_product_id"] = registry.resolve(_registry_record(product))
    return products


def _listing_text(raw_text: Optional[str], obs_id: Optional[str]) -> Optional[str]:
    """The segment of `raw_text` at the listing position named by an observation id."""
    if raw_text is None or obs_id is None or "#" not in obs_id:
        return None
    from src.ingestion.segmenter import segment_screen

    position = int(obs_id.rsplit("#", 1)[1])
    segments = segment_screen(raw_text)
    return segments[position] if 0 <= position < len(segments) else None


def registry_from_warehouse(con: duckdb.DuckDBPyConnection, **kwargs: Any) -> Any:
    """
    Rebuilds a `ProductRegistry` from the warehouse (`ProductRegistry.from_records`).

    Each `dim_products` row is registered with the listing that registered it during the
    load (its `product_aliases` registration row: the listing text, read back from
    `raw_events`, and its INR price), and every other alias key is restored, so the rebuilt
    registry resolves every listing it has seen exactly as the load did. A product without
    an alias row (written before the alias table existed) falls back to its earliest
    price observation.
    """
    from src.ingestion.entity_matcher import ProductRegistry

    rows = con.execute(
        """
        WITH first_obs AS (
            SELECT canonical_product_id, observation_id
            FROM fact_price_observations
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY canonical_product_id ORDER BY timestamp, observation_id
            ) = 1
        ),
        registration AS (
            SELECT canonical_product_id, source_observation_id
            FROM product_aliases WHERE alias_key = canonical_product_id
        ),
        source AS (
            SELECT d.canonical_product_id, d.brand, d.product_title, d.first_seen,
                   COALESCE(g.source_observation_id, f.observation_id) AS obs_id
            FROM dim_products d
            LEFT JOIN registration g USING (canonical_product_id)
            LEFT JOIN first_obs f USING (canonical_product_id)
        )
        SELECT s.canonical_product_id, s.brand, s.product_title, s.obs_id,
               CAST(o.price_inr AS DOUBLE), r.raw_text
        FROM source s
        LEFT JOIN fact_price_observations o ON o.observation_id = s.obs_id
        LEFT JOIN raw_events r ON r.event_id = split_part(s.obs_id, '#', 1)
        ORDER BY s.first_seen NULLS LAST, s.canonical_product_id
        """
    ).fetchall()
    aliases: Dict[str, List[str]] = {}
    for alias, product_id in con.execute(
        "SELECT alias_key, canonical_product_id FROM product_aliases "
        "WHERE alias_key <> canonical_product_id ORDER BY alias_key"
    ).fetchall():
        aliases.setdefault(product_id, []).append(alias)

    records = [
        {
            "canonical_product_id": product_id,
            "brand": brand,
            "product_title": title,
            "specs": [],
            "price": price_inr,
            "currency": "INR" if price_inr is not None else None,
            "raw_text": _listing_text(raw_text, obs_id),
            "aliases": aliases.get(product_id, []),
        }
        for product_id, brand, title, obs_id, price_inr, raw_text in rows
    ]
    return ProductRegistry.from_records(records, **kwargs)


# ---------------------------------------------------------------------------------------
# The one write path
# ---------------------------------------------------------------------------------------

def _event_rows(event: Mapping[str, Any]) -> Tuple[dict, dict, List[dict]]:
    """The raw, fact, and observation rows of one extracted event."""
    ts = _naive(event["timestamp"])
    products = event.get("products") or []
    raw = {
        "event_id": event["event_id"], "user_id": event["user_id"], "timestamp": ts,
        "app_name": event["app_name"], "raw_text": event["raw_text"],
        "extractor_version": EXTRACTOR_VERSION,
    }
    focal = next((p for p in products if p.get("is_focal")), products[0] if products else None)
    focal = focal or {}
    fact = {
        "event_id": event["event_id"], "user_id": event["user_id"], "timestamp": ts,
        "app_name": event["app_name"],
        "canonical_product_id": focal.get("canonical_product_id"),
        "action": warehouse_action(str(focal.get("action") or "view")),
        "price": focal.get("price"), "currency": focal.get("currency"),
        "session_id": event.get("session_id"),
        "mrp": focal.get("mrp"), "discount_pct": focal.get("discount_pct"),
        "rating": focal.get("rating"), "extraction_confidence": focal.get("extraction_confidence"),
    }
    observations = [
        {
            "observation_id": observation_id(event["event_id"], position),
            "source_event_id": event["event_id"], "user_id": event["user_id"], "timestamp": ts,
            "app_name": event["app_name"], "canonical_product_id": p["canonical_product_id"],
            "price": p["price"], "currency": p.get("currency"),
            "discount_pct": p.get("discount_pct"), "is_focal": bool(p.get("is_focal")),
        }
        for position, p in enumerate(products)
        if p.get("price") is not None and p.get("canonical_product_id")
    ]
    return raw, fact, observations


def write_events(con: duckdb.DuckDBPyConnection, events: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """
    Writes extracted events to `raw_events`, `fact_events`, `fact_price_observations`,
    `dim_products`, and `product_aliases`, in the order given. Shared by the batch loader and the serving API.

    Each event carries `event_id`, `user_id`, `timestamp`, `app_name`, `raw_text`,
    `products` (resolved listing dicts with `PRODUCT_FIELDS`), and optionally `session_id`.
    An event whose `event_id` is already in `raw_events` or `fact_events`, or repeats an
    earlier event of the same call, is skipped. `price_inr` is computed in SQL from the
    stored price and currency. A product new to `dim_products` is inserted with the first
    listing that names it; `first_seen` keeps the earliest timestamp, and `avg_price` is
    the mean INR price over every observation of the product. Returns rows written per
    table and the number of skipped events.
    """
    seen: set = set()
    fresh: List[Mapping[str, Any]] = []
    for event in events:
        if event["event_id"] not in seen:
            seen.add(event["event_id"])
            fresh.append(event)
    if fresh:
        con.register("_incoming_ids", pl.DataFrame({"event_id": [e["event_id"] for e in fresh]},
                                                   schema={"event_id": pl.Utf8}).to_arrow())
        existing = {r[0] for r in con.execute(
            """
            SELECT i.event_id FROM _incoming_ids i
            WHERE EXISTS (SELECT 1 FROM raw_events r WHERE r.event_id = i.event_id)
               OR EXISTS (SELECT 1 FROM fact_events f WHERE f.event_id = i.event_id)
            """
        ).fetchall()}
        con.unregister("_incoming_ids")
        fresh = [e for e in fresh if e["event_id"] not in existing]
    skipped = len(events) - len(fresh)
    if not fresh:
        return {"raw_events": 0, "fact_events": 0, "fact_price_observations": 0,
                "dim_products": 0, "skipped": skipped}

    raw_rows, fact_rows, obs_rows = [], [], []
    products: Dict[str, dict] = {}
    aliases: Dict[str, dict] = {}
    for event in fresh:
        raw, fact, observations = _event_rows(event)
        raw_rows.append(raw)
        fact_rows.append(fact)
        obs_rows.extend(observations)
        for position, p in enumerate(event.get("products") or []):
            pid = p.get("canonical_product_id")
            if not pid:
                continue
            key = p.get("exact_key") or pid
            if key not in aliases:
                aliases[key] = {
                    "alias_key": key, "canonical_product_id": pid,
                    "source_observation_id": observation_id(event["event_id"], position),
                }
            ts = raw["timestamp"]
            if pid not in products:
                products[pid] = {
                    "canonical_product_id": pid, "brand": p.get("brand"),
                    "product_title": p.get("canonical_name"),
                    "category_l1": p.get("category_l1"), "category_l2": p.get("category_l2"),
                    "first_seen": ts,
                }
            elif ts < products[pid]["first_seen"]:
                products[pid]["first_seen"] = ts

    frames = {
        "_raw_rows": pl.DataFrame(raw_rows, schema=_RAW_SCHEMA),
        "_fact_rows": pl.DataFrame(fact_rows, schema=_FACT_SCHEMA),
        "_obs_rows": pl.DataFrame(obs_rows, schema=_OBS_SCHEMA),
        "_product_rows": pl.DataFrame(list(products.values()), schema=_PRODUCT_SCHEMA),
        "_alias_rows": pl.DataFrame(list(aliases.values()), schema=_ALIAS_SCHEMA),
    }
    for name, frame in frames.items():
        con.register(name, frame.to_arrow())
    try:
        con.execute("INSERT INTO raw_events BY NAME SELECT * FROM _raw_rows")
        con.execute(
            f"""
            INSERT INTO fact_events BY NAME
            SELECT event_id, user_id, timestamp, app_name, canonical_product_id, action,
                   CAST(price AS DECIMAL(18,2)) AS price, currency, session_id,
                   CAST(mrp AS DECIMAL(18,2)) AS mrp, discount_pct, rating, extraction_confidence,
                   {price_inr_sql("CAST(price AS DECIMAL(18,2))")} AS price_inr
            FROM _fact_rows
            """
        )
        con.execute(
            f"""
            INSERT INTO fact_price_observations BY NAME
            SELECT observation_id, source_event_id, user_id, timestamp, app_name,
                   canonical_product_id,
                   {price_inr_sql("CAST(price AS DECIMAL(18,2))")} AS price_inr,
                   discount_pct, is_focal
            FROM _obs_rows
            """
        )
        con.execute(
            """
            INSERT INTO dim_products BY NAME
            SELECT n.* FROM _product_rows n
            WHERE NOT EXISTS (
                SELECT 1 FROM dim_products d WHERE d.canonical_product_id = n.canonical_product_id
            )
            """
        )
        con.execute(
            """
            INSERT INTO product_aliases BY NAME
            SELECT a.* FROM _alias_rows a
            WHERE NOT EXISTS (SELECT 1 FROM product_aliases p WHERE p.alias_key = a.alias_key)
            """
        )
        con.execute(
            """
            UPDATE dim_products SET first_seen = n.first_seen
            FROM _product_rows n
            WHERE dim_products.canonical_product_id = n.canonical_product_id
              AND (dim_products.first_seen IS NULL OR n.first_seen < dim_products.first_seen)
            """
        )
        con.execute(
            """
            UPDATE dim_products SET avg_price = a.avg_price
            FROM (
                SELECT o.canonical_product_id, CAST(AVG(o.price_inr) AS DECIMAL(18,2)) AS avg_price
                FROM fact_price_observations o
                WHERE o.canonical_product_id IN (SELECT canonical_product_id FROM _product_rows)
                GROUP BY o.canonical_product_id
            ) a
            WHERE dim_products.canonical_product_id = a.canonical_product_id
            """
        )
    finally:
        for name in frames:
            con.unregister(name)

    return {
        "raw_events": len(raw_rows), "fact_events": len(fact_rows),
        "fact_price_observations": len(obs_rows), "dim_products": len(products),
        "skipped": skipped,
    }


def ingest_event(
    con: duckdb.DuckDBPyConnection,
    event: Mapping[str, Any],
    registry: Any = None,
) -> Tuple[List[Any], Dict[str, int]]:
    """
    Extracts, resolves, and writes one raw event: the per-event path of the serving API.

    Returns the extracted `ExtractedProduct` list (with registry ids) and the write counts.
    """
    ts = _naive(event["timestamp"])
    screen = extract_screens([event["raw_text"]], [event["app_name"]], [ts])[0]
    slim = resolve_products([_slim(p) for p in screen], registry)
    for product, resolved in zip(screen, slim):
        product.canonical_product_id = resolved["canonical_product_id"]
    counts = write_events(con, [{**event, "timestamp": ts, "products": slim}])
    return screen, counts


# ---------------------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------------------

def _tasks(frame: pl.DataFrame, size: int) -> Iterator[Tuple[List[str], List[str], List[datetime]]]:
    """Splits the frame into extraction tasks of `size` screens, in order."""
    for start in range(0, frame.height, size):
        part = frame.slice(start, size)
        yield (part["raw_text"].to_list(), part["app_name"].to_list(), part["timestamp"].to_list())


def load_raw_events(
    con: duckdb.DuckDBPyConnection,
    raw_events: pl.DataFrame,
    workers: int = 1,
    registry: Optional[Any] = None,
    timings: Optional[Dict[str, float]] = None,
) -> Dict[str, int]:
    """
    Extracts and loads raw screen events; returns rows written per table.

    `raw_events` needs `event_id`, `user_id`, `timestamp`, `app_name`, and `raw_text`; any
    other column is ignored, so simulator ground truth never reaches the warehouse. The
    registry is rebuilt from `dim_products` when none is given. With `workers > 1`
    extraction runs in a process pool; resolution and writes stay in this process, in
    input order, so the tables are identical for any `workers`. `dim_users` is refreshed
    for every user in the warehouse at the end.

    When `timings` is given it is filled with wall-clock seconds: `extract_wait_seconds`
    (waiting for extracted screens), `resolve_seconds` (product resolution),
    `write_seconds` (warehouse writes including the `dim_users` refresh), and
    `total_seconds`.
    """
    started = time.perf_counter()
    clock = {"extract_wait_seconds": 0.0, "resolve_seconds": 0.0, "write_seconds": 0.0}
    missing = [c for c in REQUIRED_COLUMNS if c not in raw_events.columns]
    if missing:
        raise ValueError(f"raw_events is missing columns {missing}")
    frame = raw_events.select(REQUIRED_COLUMNS).with_columns(
        pl.col("timestamp").map_elements(_naive, return_dtype=pl.Datetime("us"))
        if raw_events.schema["timestamp"] == pl.Utf8 else pl.col("timestamp")
    )
    totals = {"raw_events": 0, "fact_events": 0, "fact_price_observations": 0, "skipped": 0}
    if frame.height == 0:
        return totals | {"dim_products": 0, "dim_users": 0}

    if registry is None:
        registry = registry_from_warehouse(con)

    ids = frame["event_id"].to_list()
    users = frame["user_id"].to_list()
    stamps = frame["timestamp"].to_list()
    apps = frame["app_name"].to_list()
    texts = frame["raw_text"].to_list()

    # Screens already in the memo are not sent to the workers.
    memo_hits = [_memo_get(t, a) for t, a in zip(texts, apps)]
    misses = frame.filter(pl.Series([hit is None for hit in memo_hits]))

    # Tasks are at most EVENT_LOADER_TASK_SIZE screens, and small enough that every worker
    # gets one, so a short frame still exercises the pool.
    workers = max(1, min(int(workers), misses.height))
    task_size = max(1, min(EVENT_LOADER_TASK_SIZE, -(-misses.height // workers)))
    tasks = _tasks(misses, task_size)
    executor = None
    if workers == 1:
        results: Iterator[List[List[Dict[str, Any]]]] = map(_extract_task, tasks)
    else:
        executor = _executor(workers)
        results = executor.map(_extract_task, tasks)

    def extracted() -> Iterator[List[Dict[str, Any]]]:
        for screens in results:
            yield from screens

    fresh = extracted()
    pending: List[Dict[str, Any]] = []

    def flush() -> None:
        t0 = time.perf_counter()
        counts = write_events(con, pending)
        clock["write_seconds"] += time.perf_counter() - t0
        for key in totals:
            totals[key] += counts[key]
        pending.clear()

    try:
        for row in range(frame.height):
            products = memo_hits[row]
            if products is None:
                t0 = time.perf_counter()
                products = next(fresh)
                clock["extract_wait_seconds"] += time.perf_counter() - t0
                _memo_put(texts[row], apps[row], products)
            t0 = time.perf_counter()
            resolved = resolve_products(products, registry)
            clock["resolve_seconds"] += time.perf_counter() - t0
            pending.append({
                "event_id": ids[row], "user_id": users[row], "timestamp": stamps[row],
                "app_name": apps[row], "raw_text": texts[row], "products": resolved,
            })
            if len(pending) >= EVENT_LOADER_CHUNK_SIZE:
                flush()
        if pending:
            flush()
    except BaseException:
        if executor is not None:
            _EXECUTORS.pop(workers, None)
            executor.shutdown(cancel_futures=True)
        raise

    t0 = time.perf_counter()
    refresh_dim_users(con)
    clock["write_seconds"] += time.perf_counter() - t0
    totals["dim_products"] = con.execute("SELECT COUNT(*) FROM dim_products").fetchone()[0]
    totals["dim_users"] = con.execute("SELECT COUNT(*) FROM dim_users").fetchone()[0]
    if timings is not None:
        timings.update({k: round(v, 2) for k, v in clock.items()})
        timings["total_seconds"] = round(time.perf_counter() - started, 2)
    return totals


def refresh_dim_users(con: duckdb.DuckDBPyConnection) -> None:
    """Rebuilds `dim_users` from `fact_events`: first event, most used app, event count."""
    con.execute("DELETE FROM dim_users")
    con.execute(
        """
        INSERT INTO dim_users (user_id, first_seen, primary_app, total_events)
        SELECT user_id, MIN(timestamp), MODE(app_name), COUNT(event_id)
        FROM fact_events
        GROUP BY user_id
        """
    )


if __name__ == "__main__":
    import argparse

    from src.config import DEFAULT_DB_PATH
    from src.features.migrations import migrate

    parser = argparse.ArgumentParser(description="Load a parquet of raw screen events into a warehouse.")
    parser.add_argument("raw_events", help="Parquet with event_id, user_id, timestamp, app_name, raw_text")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    connection = duckdb.connect(args.db_path)
    try:
        migrate(connection)
        print(load_raw_events(connection, pl.read_parquet(args.raw_events), workers=args.workers))
    finally:
        connection.close()
