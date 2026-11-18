"""Tests for the default-load timing benchmark (gate G1.3-11)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from src.config import DEFAULT_DB_PATH
from src.features import event_loader
from src.ingestion import load_benchmark


def _fast_extract(raw_texts, app_names, timestamps):
    """Stands in for the NLP extractor: regex price and action, first line as identity."""
    from src.ingestion.regex_extractor import extract_action, extract_price

    screens = []
    for text in raw_texts:
        price = extract_price(text)
        first = (text.strip().splitlines() or [""])[0]
        words = first.split()
        screens.append([{
            "raw_text": text, "brand": words[0] if words else None, "product_title": " ".join(words[:4]),
            "specs": [], "canonical_name": first, "canonical_product_id": "p_" + first.lower().replace(" ", "_")[:40],
            "category_l1": "x", "category_l2": "y", "price": price["price"], "currency": price["currency"],
            "mrp": price["mrp"], "discount_pct": price["discount_pct"], "rating": None,
            "extraction_confidence": 0.9, "action": extract_action(text) or "view", "is_focal": True,
        }])
    return screens


def test_load_timings_are_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(event_loader, "extract_screens", _fast_extract)
    result = load_benchmark.time_load(tmp_path, users=4, days=2, workers=1)
    for key in ("extract_wait_seconds", "resolve_seconds", "write_seconds", "total_seconds", "simulate_seconds"):
        assert result[key] >= 0.0
    assert result["total_seconds"] >= result["write_seconds"]
    assert result["counts"]["raw_events"] == result["events"] > 0
    con = duckdb.connect(str(tmp_path / "warehouse.duckdb"), read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0] == result["events"]
    finally:
        con.close()


def test_refuses_the_batch_warehouse(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        load_benchmark.time_load(Path(DEFAULT_DB_PATH).parent, users=1, days=1)


def test_load_raw_events_timings_are_optional(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import polars as pl
    from datetime import datetime
    from src.features.migrations import migrate

    monkeypatch.setattr(event_loader, "extract_screens", _fast_extract)
    con = duckdb.connect(str(tmp_path / "w.duckdb"))
    migrate(con)
    raw = pl.DataFrame({"event_id": ["e1"], "user_id": ["u1"], "timestamp": [datetime(2026, 10, 1)],
                        "app_name": ["flipkart"], "raw_text": ["Acme Widget In stock"]})
    counts = event_loader.load_raw_events(con, raw)
    assert counts["raw_events"] == 1
    timings: dict = {}
    event_loader.load_raw_events(con, raw.with_columns(event_id=pl.lit("e2")), timings=timings)
    assert set(timings) == {"extract_wait_seconds", "resolve_seconds", "write_seconds", "total_seconds"}
    con.close()
