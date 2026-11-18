"""Tests for the noise propagation experiment: accuracy scoring, paired deltas, isolation and a tiny end-to-end run."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest

from src.config import DEFAULT_DB_PATH, USD_TO_INR_RATE
from src.experiments import noise_propagation as npx


def _make_db(path: Path, rows: list) -> None:
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE fact_events (event_id VARCHAR, action VARCHAR, price DOUBLE, "
                "currency VARCHAR, canonical_product_id VARCHAR)")
    for r in rows:
        con.execute("INSERT INTO fact_events VALUES (?, ?, ?, ?, ?)", r)
    con.close()


def _truth(path: Path, rows: list) -> None:
    schema = {"event_id": pl.String, "true_sku_key": pl.String, "true_action": pl.String,
              "true_price": pl.Float64, "true_currency": pl.String}
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(path)


def test_event_accuracy_scores_each_field(tmp_path: Path) -> None:
    _make_db(tmp_path / "w.duckdb", [
        ("e1", "view", 100.0, "INR", "P1"),
        ("e2", "buy", 100.0 * USD_TO_INR_RATE * 1.005, "INR", "P1"),
        ("e3", "cart", 150.0, "INR", "P1"),   # wrong price
        ("e4", "view", 10.0, "INR", "P9"),    # wrong identity, wrong action
    ])
    _truth(tmp_path / "t.parquet", [
        ("e1", "S", "view", 100.0, "INR"),
        ("e2", "S", "buy", 100.0, "USD"),
        ("e3", "S", "cart", 100.0, "INR"),
        ("e4", "S", "buy", 10.0, "INR"),
        ("e5", "S", "view", 10.0, "INR"),     # missing from warehouse
    ])
    acc = npx.event_accuracy(tmp_path / "w.duckdb", tmp_path / "t.parquet")
    assert acc["events"] == 5 and acc["matched"] == 4
    assert acc["action"] == pytest.approx(3 / 5)
    assert acc["price"] == pytest.approx(3 / 5)
    assert acc["identity"] == pytest.approx(3 / 5)


def test_event_accuracy_empty_truth(tmp_path: Path) -> None:
    _make_db(tmp_path / "w.duckdb", [])
    _truth(tmp_path / "t.parquet", [])
    acc = npx.event_accuracy(tmp_path / "w.duckdb", tmp_path / "t.parquet")
    assert acc["events"] == 0 and math.isnan(acc["action"])


def test_paired_delta_identical_scores_is_zero() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 60)
    p = rng.random(60)
    users = np.array([f"u{i}" for i in range(60)])
    run = {"_scores": {"users": users, "y": y, "p": p}}
    out = npx.paired_delta(run, run, n_boot=50)
    assert out["delta"] == 0.0 and out["shared_users"] == 60


def test_paired_delta_uses_shared_users_only() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 40)
    p = rng.random(40)
    a = {"_scores": {"users": np.array([f"u{i}" for i in range(40)]), "y": y, "p": p}}
    b = {"_scores": {"users": np.array([f"u{i}" for i in range(20, 60)]), "y": y, "p": p}}
    assert npx.paired_delta(a, b, n_boot=20)["shared_users"] == 20
    c = {"_scores": {"users": np.array(["z1", "z2"]), "y": y[:2], "p": p[:2]}}
    assert npx.paired_delta(a, c, n_boot=5)["shared_users"] == 0


def _fast_extract_screens(texts, apps, stamps):
    """Cheap stand-in for the NLP extractor: regex price and action, first line as identity."""
    from src.ingestion.regex_extractor import extract_action, extract_price

    screens = []
    for text in texts:
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


def test_tiny_run_is_isolated_and_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Whole chain on 24 users; only the NLP extractor is replaced so the test runs in seconds."""
    from src.features import event_loader

    monkeypatch.setattr(event_loader, "extract_screens", _fast_extract_screens)
    monkeypatch.setattr(npx, "LOAD_WORKERS", 1)
    from functools import partial
    from src.models import propensity

    # The 2000-resample bootstrap inside the propensity report dominates the runtime.
    monkeypatch.setattr(propensity, "baseline_report", partial(propensity.baseline_report, n_boot=20))
    before = DEFAULT_DB_PATH.stat().st_mtime_ns if DEFAULT_DB_PATH.exists() else None
    res = npx.run_noise_propagation(tmp_path, noise_levels=(0.0, 0.05), users=24, days=30, n_boot=30)
    assert set(res["levels"]) == {"0", "0.05"}
    for key, lvl in res["levels"].items():
        assert (tmp_path / key / "warehouse.duckdb").exists()
        assert (tmp_path / key / "truth.parquet").exists()
        assert {"action", "price", "identity"} <= set(lvl["accuracy"])
        assert 0.0 <= lvl["propensity_test"]["roc_auc"] <= 1.0
        assert "_scores" not in lvl
    assert res["levels"]["0"]["delta_vs_baseline"]["delta"] == 0.0
    after = DEFAULT_DB_PATH.stat().st_mtime_ns if DEFAULT_DB_PATH.exists() else None
    assert before == after
