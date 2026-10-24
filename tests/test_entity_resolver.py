import pytest
from src.ingestion.entity_resolver import (
    compute_canonical_key,
    compute_canonical_name,
    are_prices_compatible,
    resolve_entities,
)


def test_canonical_key_identical_for_cross_app_variants():
    # Amazon format
    key_amazon = compute_canonical_key("Apple", "iPhone 15", ["(128 GB)", "Blue"])
    # Flipkart format (different word order and formatting)
    key_flipkart = compute_canonical_key("Apple", "iPhone 15", ["128 GB", "Blue"])

    assert key_amazon == key_flipkart


def test_canonical_key_distinguishes_different_specs():
    key_128 = compute_canonical_key("Apple", "iPhone 15", ["128 GB"])
    key_256 = compute_canonical_key("Apple", "iPhone 15", ["256 GB"])

    assert key_128 != key_256


def test_canonical_key_distinguishes_different_products():
    key_iphone = compute_canonical_key("Apple", "iPhone 15", [])
    key_galaxy = compute_canonical_key("Samsung", "Galaxy S24", [])

    assert key_iphone != key_galaxy


def test_compute_canonical_name():
    assert compute_canonical_name("Apple", "iPhone 15") == "Apple iPhone 15"
    assert compute_canonical_name("Apple", "Apple iPhone 15") == "Apple iPhone 15"
    assert compute_canonical_name(None, "boAt Rockerz 450") == "boAt Rockerz 450"
    assert compute_canonical_name("", "") == "Unknown Product"


def test_are_prices_compatible():
    assert are_prices_compatible(71999.0, 72500.0, tolerance_pct=0.10) is True
    assert are_prices_compatible(71999.0, 95000.0, tolerance_pct=0.10) is False
    assert are_prices_compatible(None, 500.0) is True


def test_resolve_entities_pipeline():
    products = [
        {
            "brand": "boAt",
            "product_title": "Airdopes 141",
            "specs": ["True Wireless Earbuds", "Black"],
            "price": 22.90,
            "app_source": "walmart"
        },
        {
            "brand": "boAt",
            "product_title": "Airdopes 141",
            "specs": ["True Wireless Earbuds", "Black"],
            "price": 24.50,
            "app_source": "amazon_us"
        },
        {
            "brand": "Apple",
            "product_title": "iPhone 16 Pro",
            "specs": ["256 GB", "5G"],
            "price": 999.0,
            "app_source": "amazon_us"
        }
    ]

    resolved = resolve_entities(products)
    assert len(resolved) == 3
    assert "canonical_product_id" in resolved[0]
    assert "canonical_name" in resolved[0]

    # The two boAt items from different apps must map to the exact same canonical_product_id
    assert resolved[0]["canonical_product_id"] == resolved[1]["canonical_product_id"]
    # iPhone must map to a different canonical_product_id
    assert resolved[0]["canonical_product_id"] != resolved[2]["canonical_product_id"]
