import pytest
from src.ingestion.taxonomy_classifier import classify_product


def test_classify_smartphone():
    res = classify_product("iPhone 15 Pro Max 256GB", brand="Apple")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Smartphones"
    assert "Smartphones" in res["full_category"]
    assert res["confidence"] > 0.25


def test_classify_audio_headphones():
    res = classify_product("Airdopes 141 True Wireless Earbuds", brand="boAt")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Audio"
    assert "Headphones" in res["full_category"]
    assert res["confidence"] > 0.50


def test_classify_smartwatch():
    res = classify_product("Galaxy Watch 6 Bluetooth 44mm", brand="Samsung")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Wearables"
    assert "Smartwatches" in res["full_category"]
    assert res["confidence"] > 0.35


def test_classify_footwear():
    res = classify_product("Air Monarch IV Cross Training Shoes", brand="Nike")
    assert res["category_l1"] == "Apparel & Accessories"
    assert res["category_l2"] == "Footwear"
    assert "Shoes" in res["full_category"]
    assert res["confidence"] > 0.50


def test_classify_storage():
    res = classify_product("Extreme Portable SSD 1TB", brand="SanDisk")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Storage"
    assert "Storage" in res["full_category"]
    assert res["confidence"] > 0.40


def test_classify_accessories():
    res = classify_product("MX Master 3S Wireless Performance Mouse", brand="Logitech")
    assert res["category_l1"] == "Electronics"
    assert res["category_l2"] == "Accessories"
    assert res["confidence"] > 0.35


def test_classify_empty():
    res = classify_product("", brand="")
    assert res["category_l1"] == "Unknown"
    assert res["confidence"] == 0.0


from src.config import TAXONOMY_MIN_SIMILARITY


def test_classifier_abstains_below_similarity_floor():
    """
    Verifies a product unlike every taxonomy description is labelled Unknown.

    Nearest-neighbour search over a closed label set always returns something, which
    once put a quarter of real phone accessories in categories such as Footwear.
    """
    res = classify_product("Ryleigh Name on Geometric Red Pink Purple Black PopGrip")
    assert res["confidence"] < TAXONOMY_MIN_SIMILARITY
    assert res["abstained"] is True
    assert res["category_l1"] == "Unknown"
    assert res["category_l2"] == "Unknown"
    assert res["nearest_category"] is not None


def test_classifier_floor_is_configurable():
    """Verifies the floor can be overridden per call, and reports the runner-up margin."""
    strict = classify_product("Extreme Portable SSD 1TB", brand="SanDisk", min_similarity=0.99)
    assert strict["abstained"] is True
    assert "Storage" in strict["nearest_category"]

    lenient = classify_product("Extreme Portable SSD 1TB", brand="SanDisk", min_similarity=0.0)
    assert lenient["abstained"] is False
    assert lenient["category_l2"] == "Storage"
    assert lenient["margin"] >= 0.0
