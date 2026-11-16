import pytest
from src.ingestion.regex_extractor import extract_price, extract_rating, extract_action


def test_extract_price_inr_basic():
    text = "Sponsored Apple iPhone 15 ₹68,330 M.R.P: ₹75,920 (10% off) Buy Now"
    res = extract_price(text)
    assert res["price"] == 68330.0
    assert res["currency"] == "INR"
    assert res["mrp"] == 75920.0
    assert res["discount_pct"] == 10


def test_extract_price_usd_with_cents():
    text = "boAt Airdopes 141 $22.90 Was: $29.99 (24% off)"
    res = extract_price(text)
    assert res["price"] == 22.90
    assert res["currency"] == "USD"
    assert res["mrp"] == 29.99
    assert res["discount_pct"] == 24


def test_extract_price_no_discount():
    text = "Sony WH-1000XM5 ₹29,990 FREE Delivery"
    res = extract_price(text)
    assert res["price"] == 29990.0
    assert res["currency"] == "INR"
    assert res["discount_pct"] == 0
    assert res["mrp"] == 29990.0


def test_extract_price_rs_prefix():
    text = "Samsung Galaxy S24 Ultra Rs. 1,29,999 M.R.P: Rs. 1,39,999"
    res = extract_price(text)
    assert res["price"] == 129999.0
    assert res["currency"] == "INR"
    assert res["mrp"] == 139999.0


def test_extract_price_empty_and_none():
    assert extract_price("") == {"price": None, "currency": None, "mrp": None, "discount_pct": None}
    assert extract_price(None) == {"price": None, "currency": None, "mrp": None, "discount_pct": None}


def test_extract_price_list_price_format():
    text = "Dell XPS 13 $1,199.00 List Price: $1,399.00"
    res = extract_price(text)
    assert res["price"] == 1199.00
    assert res["currency"] == "USD"
    assert res["mrp"] == 1399.00


def test_extract_price_discount_without_mrp():
    text = "Noise ColorFit Pulse $50.00 (50% off)"
    res = extract_price(text)
    assert res["price"] == 50.0
    assert res["currency"] == "USD"
    assert res["discount_pct"] == 50
    assert res["mrp"] == 100.0


def test_extract_rating_out_of_5():
    text = "4.7 out of 5 stars Apple iPhone 15 (2,847)"
    res = extract_rating(text)
    assert res["rating"] == 4.7
    assert res["num_ratings"] == 2847


def test_extract_rating_star_symbol():
    text = "boAt Airdopes 141 4.5★ (43,981 reviews)"
    res = extract_rating(text)
    assert res["rating"] == 4.5
    assert res["num_ratings"] == 43981


def test_extract_rating_unicode_stars():
    text = "OnePlus Nord CE 3 ★★★★☆ (12,400 ratings)"
    res = extract_rating(text)
    assert res["rating"] == 4.0
    assert res["num_ratings"] == 12400


def test_extract_rating_slash_5():
    text = "Rating: 4.2 / 5 (850)"
    res = extract_rating(text)
    assert res["rating"] == 4.2
    assert res["num_ratings"] == 850


def test_extract_rating_missing():
    text = "Just a product title with no rating info"
    res = extract_rating(text)
    assert res["rating"] is None
    assert res["num_ratings"] is None


def test_extract_rating_empty():
    assert extract_rating("") == {"rating": None, "num_ratings": None}
    assert extract_rating(None) == {"rating": None, "num_ratings": None}


def test_extract_action_buy():
    assert extract_action("Apple iPhone 15 Buy Now FREE delivery") == "buy"
    assert extract_action("Proceed to buy item in your account") == "buy"
    assert extract_action("Order now and get 10% discount") == "buy"


def test_extract_action_cart():
    assert extract_action("boAt Rockerz 450 Add to Cart") == "cart"
    assert extract_action("Item added to cart successfully") == "cart"
    assert extract_action("Go to cart to review") == "cart"


def test_extract_action_wishlist():
    assert extract_action("Save for later or Add to Wishlist") == "wishlist"
    assert extract_action("Moved to wishlist") == "wishlist"


def test_extract_action_search():
    assert extract_action("Searching for wireless headphones with noise cancelling") == "search"
    assert extract_action("Search results for iPhone 15") == "search"
    assert extract_action("Popular pick in electronics") == "search"
    assert extract_action("Pickup today at Supercenter") == "search"


def test_extract_action_view_fallback():
    assert extract_action("Sponsored product details overview specifications") == "view"
    assert extract_action("Random unclassified screen text") == "view"
    assert extract_action("") == "view"
    assert extract_action(None) == "view"


def test_extract_price_inr_lakhs_notation():
    text = "MacBook Pro M3 Max ₹3,49,900 M.R.P: ₹3,99,900"
    res = extract_price(text)
    assert res["price"] == 349900.0
    assert res["currency"] == "INR"
    assert res["mrp"] == 399900.0


def test_extract_rating_large_number():
    text = "Kindle Paperwhite 4.6 out of 5 stars (182,900 ratings)"
    res = extract_rating(text)
    assert res["rating"] == 4.6
    assert res["num_ratings"] == 182900


def test_extract_price_multiple_discounts_picks_match():
    text = "Nike Air Max ₹6,995 M.R.P: ₹9,995 (30% off)"
    res = extract_price(text)
    assert res["price"] == 6995.0
    assert res["discount_pct"] == 30


@pytest.mark.parametrize(
    "text, expected_price, expected_currency",
    [
        ("Samsung Galaxy S24 Ultra 256GB Titanium $1299.99 4.7★ Add to Cart", 1299.99, "USD"),
        ("Sony Bravia 55 inch $12999", 12999.0, "USD"),
        ("OnePlus Nord CE ₹15999", 15999.0, "INR"),
        ("Boat Rockerz Rs. 2499", 2499.0, "INR"),
    ],
)
def test_extract_price_without_thousands_separator(text, expected_price, expected_currency):
    # Regression: the grouped branch of the price pattern once matched only the first three
    # digits of an unseparated price, so "$1299.99" was read as 129.0.
    res = extract_price(text)
    assert res["price"] == expected_price
    assert res["currency"] == expected_currency


def test_extract_price_separated_formats_unchanged():
    assert extract_price("$1,299.99")["price"] == 1299.99
    assert extract_price("₹15,999")["price"] == 15999.0
    assert extract_price("₹1,23,456")["price"] == 123456.0
    assert extract_price("$22.90")["price"] == 22.9


def test_extract_price_unseparated_mrp_and_selling_price():
    res = extract_price("List Price: $1499.00 $1299.99")
    assert res["price"] == 1299.99
    assert res["mrp"] == 1499.0
    assert res["discount_pct"] == 13
