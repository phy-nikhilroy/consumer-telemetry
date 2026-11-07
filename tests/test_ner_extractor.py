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


from src.ingestion.ner_extractor import (
    BRAND_SOURCE_LEADING_MATCH,
    BRAND_SOURCE_LEADING_TOKEN,
    extract_brand_with_source,
    extract_price_mentions,
    register_brands,
    reset_registered_brands,
)


def test_compatibility_mention_is_not_the_brand():
    """
    Verifies a brand named only as the device an accessory fits is not returned.

    On real Amazon listings the closed-vocabulary matcher attributed accessories to the
    brand of the phone they fit, confidently wrong on about half of all products.
    """
    assert extract_brand("Case for Samsung Galaxy S6 Slim Translucent Mirror Flip") != "Samsung"
    assert extract_brand("Leather bands compatible with Apple Watch Series 7") is None


def test_brand_named_as_compatibility_anywhere_is_rejected_throughout():
    """Verifies an unqualified device-brand mention is rejected when it is also a target."""
    text = "Samsung Galaxy S10 Wallet Case Glitter Cover for Samsung Galaxy S10"
    assert extract_brand(text) != "Samsung"


def test_open_vocabulary_leading_word():
    """Verifies an unlisted seller brand leading the title is recovered."""
    brand, source = extract_brand_with_source("Spigen Tough Armor Case for iPhone 15 Pro $19.99")
    assert brand == "Spigen"
    assert source == BRAND_SOURCE_LEADING_TOKEN


def test_leading_seller_beats_mid_title_device_brand():
    """Verifies the leading seller wins over a device brand that appears later."""
    assert extract_brand("Bonoma Samsung Galaxy S8 Case Geometric Marble") == "Bonoma"


def test_leading_word_filters():
    """Verifies generic words, model numbers, and device lines are not read as brands."""
    assert extract_brand("Premium Rugged Heavy Duty Case") is None
    assert extract_brand("R2111 Dark Samurai Case Cover") is None
    assert extract_brand("iPhone 7 Plus Adventure Skull Face Case") is None


def test_all_caps_two_word_seller_name():
    """Verifies a two-word all-capital seller name is kept together."""
    assert extract_brand("MODOS LOGICOS Case for Samsung Galaxy S21") == "MODOS LOGICOS"


def test_known_brand_leading_title_reports_strongest_source():
    """Verifies a known brand at the start of the title carries the strongest evidence."""
    brand, source = extract_brand_with_source("Sponsored 4.5★ (2,847) Apple iPhone 15 (128 GB)")
    assert brand == "Apple"
    assert source == BRAND_SOURCE_LEADING_MATCH


def test_registered_gazetteer_extends_and_resets():
    """Verifies a registered gazetteer is matched with its own spelling, then removed."""
    text = "Velvet Caviar Compatible with Samsung Galaxy S21 Case Pink"
    reset_registered_brands()
    try:
        assert extract_brand(text) == "Velvet"
        assert register_brands(["Velvet Caviar", "Apple", "Generic", ""]) == 1
        brand, source = extract_brand_with_source(text)
        assert brand == "Velvet Caviar"
        assert source == BRAND_SOURCE_LEADING_MATCH
    finally:
        reset_registered_brands()
    assert extract_brand(text) == "Velvet"


def test_extract_price_mentions_reads_ner_money():
    """Verifies spaCy NER money mentions are parsed into amounts."""
    amounts = extract_price_mentions("Samsung Galaxy S24 Ultra $1,299.99 Add to Cart")
    assert 1299.99 in amounts
    assert extract_price_mentions("") == []
