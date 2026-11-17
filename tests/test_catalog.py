"""Tests for the product-family catalog in src/ingestion/catalog.py."""

from __future__ import annotations

from src.config import USD_TO_INR_RATE
from src.ingestion.catalog import FAMILIES, build_catalog
from src.config import PRICE_TIERS
from src.ingestion.synthetic_data import CATALOG

KEYS = {"brand", "title", "specs", "base_inr", "base_usd", "category", "sku_id", "family_id", "variant", "tier"}


def test_at_least_300_skus() -> None:
    assert len(build_catalog()) >= 300


def test_sku_id_unique() -> None:
    ids = [s["sku_id"] for s in build_catalog()]
    assert len(ids) == len(set(ids))


def test_deterministic() -> None:
    assert build_catalog() == build_catalog()
    assert build_catalog(1) == build_catalog(2)


def test_prices_and_tiers_valid() -> None:
    for s in build_catalog():
        assert set(s) >= KEYS
        assert s["base_inr"] > 0 and s["base_usd"] > 0
        assert s["tier"] in PRICE_TIERS
        low, high = PRICE_TIERS[s["tier"]]
        assert low <= s["base_inr"] < high
        assert s["base_usd"] == round(s["base_inr"] / USD_TO_INR_RATE, 2)
        assert s["sku_id"].startswith(s["family_id"] + "-")


def test_structural_minimums() -> None:
    assert len(FAMILIES) >= 40
    assert sum(1 for f in FAMILIES if len(f.get("storage", [])) >= 2) >= 15
    lines = {(f["brand"], f["line"]) for f in FAMILIES}
    pairs = [
        (a, b)
        for a in lines
        for b in lines
        if a[0] == b[0] and b[1].startswith(a[1] + " ") and len(b[1].split()) == len(a[1].split()) + 1
    ]
    assert len(pairs) >= 5


def test_larger_storage_costs_more() -> None:
    by_family: dict = {}
    for s in build_catalog():
        if "storage" in s["variant"]:
            by_family.setdefault((s["family_id"], s["variant"].get("colour")), []).append(s)
    assert by_family
    for group in by_family.values():
        prices = [s["base_inr"] for s in group]  # product order is cheapest storage first
        assert prices == sorted(prices) and prices[0] < prices[-1]


def test_categories_come_from_v1_catalog() -> None:
    allowed = {c["category"] for c in CATALOG}
    assert {s["category"] for s in build_catalog()} <= allowed


def test_v1_products_present() -> None:
    present = {(s["brand"], s["title"]) for s in build_catalog()}
    assert len(CATALOG) == 17
    for item in CATALOG:
        assert (item["brand"], item["title"]) in present
