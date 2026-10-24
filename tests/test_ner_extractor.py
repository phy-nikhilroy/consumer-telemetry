import pytest
from src.ingestion.ner_extractor import (
    extract_brand,
    extract_specs,
    extract_product_title,
    extract_all,
)


def test_extract_brand_known_brands():
    assert extract_brand("Apple iPhone 15 128GB") == "Apple"
    assert extract_brand("Samsung Galaxy S24 Ultra") == "Samsung"
    assert extract_brand("boAt Airdopes 141 True Wireless") == "boAt"
    assert extract_brand("Sony WH-1000XM5 Noise Cancelling") == "Sony"
    assert extract_brand("Nike Air Max Impact 4 Basketball Shoes") == "Nike"


def test_extract_brand_case_insensitive():
    assert extract_brand("apple iphone 13") == "Apple"
    assert extract_brand("BOAT airdopes 131") == "boAt"
    assert extract_brand("samsung galaxy watch 6") == "Samsung"


def test_extract_brand_unknown_returns_none():
    assert extract_brand("Generic Unbranded Phone Case Transparent") is None
    assert extract_brand("") is None
    assert extract_brand(None) is None


def test_extract_specs_storage_and_connectivity():
    text = "Apple iPhone 15 (128 GB) 5G Blue"
    specs = extract_specs(text)
    assert "(128 GB)" in specs
    assert "128 GB" in specs or "(128 GB)" in specs
    assert "5G" in specs
    assert "Blue" in specs


def test_extract_specs_audio_features():
    text = "Sony WH-1000XM5 Active Noise Cancellation Bluetooth Black"
    specs = extract_specs(text)
    assert "Active Noise Cancellation" in specs
    assert "Bluetooth" in specs
    assert "Black" in specs


def test_extract_product_title_clean():
    text = "Sponsored Apple iPhone 15 (128 GB) ₹68,330 Buy Now"
    title = extract_product_title(text, brand="Apple")
    assert title == "iPhone 15"


def test_extract_product_title_boat():
    text = "boAt Airdopes 141 True Wireless Earbuds $22.90"
    title = extract_product_title(text, brand="boAt")
    assert "Airdopes 141" in title


def test_extract_all_unified():
    raw_text = "Sponsored 4.7 out of 5 stars Apple iPhone 15 (128 GB) ₹68,330 M.R.P: ₹75,920 (10% off) FREE delivery Buy Now"
    res = extract_all(raw_text)
    assert res["brand"] == "Apple"
    assert "iPhone 15" in res["product_title"]
    assert res["price"] == 68330.0
    assert res["currency"] == "INR"
    assert res["mrp"] == 75920.0
    assert res["discount_pct"] == 10
    assert res["rating"] == 4.7
    assert res["action"] == "buy"
    assert len(res["specs"]) > 0


def test_extract_all_empty():
    res = extract_all("")
    assert res["brand"] is None
    assert res["product_title"] is None
    assert res["price"] is None
    assert res["action"] == "view"
