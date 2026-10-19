"""Synthetic OCR Screen Dump Generator for Day 1.

Generates realistic, noisy screen strings simulating raw OCR telemetry from e-commerce apps
(Amazon India, Amazon US, Flipkart, Walmart) with varying layouts, currencies, ratings,
badges, actions, and specifications.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Product catalogs across categories
CATALOG = [
    # Electronics / Smartphones
    {
        "brand": "Apple",
        "title": "iPhone 15",
        "specs": ["(128 GB)", "Blue", "A16 Bionic"],
        "base_inr": 71999,
        "base_usd": 799,
        "category": "Smartphones",
    },
    {
        "brand": "Apple",
        "title": "iPhone 16 Pro",
        "specs": ["256 GB", "Natural Titanium", "5G"],
        "base_inr": 119900,
        "base_usd": 999,
        "category": "Smartphones",
    },
    {
        "brand": "Samsung",
        "title": "Galaxy S24 Ultra",
        "specs": ["Titanium Gray", "12GB RAM", "256GB Storage", "5G AI Smartphone"],
        "base_inr": 129999,
        "base_usd": 1299.99,
        "category": "Smartphones",
    },
    {
        "brand": "Samsung",
        "title": "Galaxy M34 5G",
        "specs": ["Midnight Blue", "6GB RAM", "128GB Storage"],
        "base_inr": 16999,
        "base_usd": 210,
        "category": "Smartphones",
    },
    {
        "brand": "OnePlus",
        "title": "12R 5G",
        "specs": ["Cool Blue", "16GB RAM", "256GB Storage"],
        "base_inr": 42999,
        "base_usd": 499,
        "category": "Smartphones",
    },
    {
        "brand": "Google",
        "title": "Pixel 8a",
        "specs": ["Obsidian", "128 GB", "Unlocked"],
        "base_inr": 52999,
        "base_usd": 499,
        "category": "Smartphones",
    },
    # Audio
    {
        "brand": "boAt",
        "title": "Rockerz 450",
        "specs": ["Bluetooth On Ear Headphones", "with Mic", "Luscious Black"],
        "base_inr": 1099,
        "base_usd": 19.99,
        "category": "Audio",
    },
    {
        "brand": "boAt",
        "title": "Airdopes 141",
        "specs": ["True Wireless Earbuds", "42H Playtime", "Bold Black"],
        "base_inr": 1299,
        "base_usd": 22.50,
        "category": "Audio",
    },
    {
        "brand": "Sony",
        "title": "WH-1000XM5",
        "specs": ["Wireless Industry Leading Noise Canceling Headphones", "Black"],
        "base_inr": 29990,
        "base_usd": 398.00,
        "category": "Audio",
    },
    {
        "brand": "JBL",
        "title": "Flip 6",
        "specs": ["Portable Bluetooth Speaker", "IP67 Waterproof", "Squad"],
        "base_inr": 8999,
        "base_usd": 119.95,
        "category": "Audio",
    },
    # Wearables
    {
        "brand": "Noise",
        "title": "ColorFit Pulse 3",
        "specs": ["1.96 Display Smart Watch", "Bluetooth Calling", "Jet Black"],
        "base_inr": 1499,
        "base_usd": 25.0,
        "category": "Wearables",
    },
    {
        "brand": "Apple",
        "title": "Watch Series 9",
        "specs": ["GPS 41mm", "Midnight Aluminum Case", "Sport Band"],
        "base_inr": 41900,
        "base_usd": 399.0,
        "category": "Wearables",
    },
    # Footwear & Apparel
    {
        "brand": "Nike",
        "title": "Air Monarch IV",
        "specs": ["Men's Cross Training Shoes", "White/Metallic Silver"],
        "base_inr": 5495,
        "base_usd": 75.0,
        "category": "Footwear",
    },
    {
        "brand": "Adidas",
        "title": "Ultraboost Light",
        "specs": ["Running Shoes", "Core Black", "Size 9"],
        "base_inr": 11999,
        "base_usd": 150.0,
        "category": "Footwear",
    },
    {
        "brand": "Puma",
        "title": "Smash v2",
        "specs": ["Leather Sneaker", "Puma White", "Unisex"],
        "base_inr": 2499,
        "base_usd": 45.0,
        "category": "Footwear",
    },
    # Computing & Storage
    {
        "brand": "SanDisk",
        "title": "Extreme Portable SSD",
        "specs": ["1TB", "Up to 1050MB/s", "USB-C"],
        "base_inr": 9499,
        "base_usd": 119.99,
        "category": "Storage",
    },
    {
        "brand": "Logitech",
        "title": "MX Master 3S",
        "specs": ["Wireless Performance Mouse", "Quiet Clicks", "Graphite"],
        "base_inr": 8995,
        "base_usd": 99.99,
        "category": "Accessories",
    },
]

ACTIONS = ["view", "cart", "buy", "search", "wishlist"]

ACTION_SNIPPETS = {
    "amazon_in": {
        "view": ["See More Details", "Explore related items"],
        "cart": ["Add to Cart"],
        "buy": ["Buy Now"],
        "search": ["Search results for", "Sponsored result"],
        "wishlist": ["Add to Wish List"],
    },
    "amazon_us": {
        "view": ["Details", "Compare with similar items"],
        "cart": ["Add to Cart"],
        "buy": ["Buy Now"],
        "search": ["Sponsored", "Best Seller in"],
        "wishlist": ["Add to List"],
    },
    "flipkart": {
        "view": ["View Details", "Similar Products"],
        "cart": ["GO TO CART", "ADD TO CART"],
        "buy": ["BUY NOW"],
        "search": ["Showing results for", "Flipkart Trending"],
        "wishlist": ["Add to Wishlist"],
    },
    "walmart": {
        "view": ["View product details", "Free pickup today"],
        "cart": ["Add to cart"],
        "buy": ["Buy now with 1-click"],
        "search": ["Results for", "Popular pick in"],
        "wishlist": ["Save for later"],
    },
}

DELIVERY_BADGES = {
    "amazon_in": [
        "FREE delivery Thu, 18 Jan",
        "FREE delivery Tomorrow by 10 PM",
        "Prime delivery tomorrow",
        "Fastest delivery Today 5 PM",
        "",
    ],
    "amazon_us": [
        "Prime FREE One-Day",
        "FREE delivery Wed, Oct 22 on $35 of items shipped by Amazon",
        "FREE delivery Tomorrow",
        "Ships to India",
        "",
    ],
    "flipkart": [
        "Flipkart Assured",
        "Free delivery",
        "Delivery by Tomorrow, 11 PM",
        "Bank Offer 10% off on SBI Credit Card",
        "",
    ],
    "walmart": [
        "Free 90-day returns",
        "Free shipping, arrives in 2 days",
        "Pickup today at Supercenter",
        "Free delivery with Walmart+",
        "",
    ],
}


def _format_inr(val: float) -> str:
    s = str(int(round(val)))
    if len(s) <= 3:
        return s
    first, rest = s[:-3], s[-3:]
    # Indian numbering: commas every 2 digits after the last 3
    parts = []
    while len(first) > 2:
        parts.insert(0, first[-2:])
        first = first[:-2]
    if first:
        parts.insert(0, first)
    return ",".join(parts) + "," + rest


def generate_single_sample(sample_id: int) -> Dict[str, Any]:
    """Generates a single synthetic sample with ground truth metadata."""
    app = random.choice(["amazon_in", "amazon_us", "flipkart", "walmart"])
    item = random.choice(CATALOG)
    action = random.choice(ACTIONS)

    # Price calculation & currency
    currency = "INR" if app in ["amazon_in", "flipkart"] else "USD"
    curr_symbol = "₹" if currency == "INR" else "$"

    base_price = item["base_inr"] if currency == "INR" else item["base_usd"]
    # Slight price fluctuation
    price = round(base_price * random.uniform(0.90, 1.10), 2 if currency == "USD" else -1)
    if currency == "INR":
        price = float(int(price))

    has_discount = random.random() < 0.75
    discount_pct = random.choice([5, 10, 15, 20, 25, 30, 45, 72]) if has_discount else 0
    if discount_pct > 0:
        mrp = round(price / (1 - (discount_pct / 100)), 2 if currency == "USD" else -1)
        if currency == "INR":
            mrp = float(int(mrp))
    else:
        mrp = price

    # Format price strings
    if currency == "INR":
        price_str = f"{curr_symbol}{_format_inr(price)}"
        mrp_str = f"{curr_symbol}{_format_inr(mrp)}"
    else:
        price_str = f"{curr_symbol}{price:,.2f}"
        mrp_str = f"{curr_symbol}{mrp:,.2f}"

    # Ratings
    has_rating = random.random() < 0.90
    rating_val = round(random.uniform(3.5, 4.9), 1) if has_rating else None
    num_ratings = (
        random.choice([2847, 123456, 1420, 52100, 890, 43981]) if has_rating else None
    )

    rating_str = ""
    if has_rating:
        style = random.choice(["star_symbol", "unicode_stars", "text_rating"])
        if style == "star_symbol":
            rating_str = f"{rating_val} ★"
            if random.random() < 0.6:
                rating_str += f" ({num_ratings:,})" if currency == "USD" else f" {_format_inr(num_ratings)} ratings"
        elif style == "unicode_stars":
            rating_str = f"★★★★☆ ({num_ratings:,})"
        else:
            rating_str = f"{rating_val} out of 5 stars"

    # Action and delivery noise
    action_phrase = random.choice(ACTION_SNIPPETS[app][action])
    delivery_phrase = random.choice(DELIVERY_BADGES[app])
    sponsored = "Sponsored " if random.random() < 0.3 else ""

    # Selected specs
    specs_picked = random.sample(item["specs"], k=random.randint(1, len(item["specs"])))
    specs_str = " ".join(specs_picked)

    # Discount string format variations
    if discount_pct > 0:
        disc_fmt = random.choice([
            f"M.R.P: {mrp_str} ({discount_pct}% off)",
            f"{mrp_str} {discount_pct}% off",
            f"List Price: {mrp_str} Save {discount_pct}%",
            f"({discount_pct}% off)",
        ])
    else:
        disc_fmt = ""

    # Build raw screen string per app style
    if app == "amazon_in":
        chunks = [
            sponsored,
            rating_str,
            item["brand"],
            item["title"],
            specs_str,
            price_str,
            disc_fmt,
            delivery_phrase,
            action_phrase,
        ]
    elif app == "amazon_us":
        chunks = [
            sponsored,
            item["brand"],
            item["title"],
            specs_str,
            price_str,
            disc_fmt,
            rating_str,
            delivery_phrase,
            "In Stock" if random.random() < 0.5 else "",
            action_phrase,
        ]
    elif app == "flipkart":
        chunks = [
            item["brand"],
            item["title"],
            specs_str,
            price_str,
            mrp_str if disc_fmt else "",
            f"{discount_pct}% off" if discount_pct > 0 else "",
            rating_str,
            delivery_phrase,
            action_phrase,
        ]
    else:  # walmart
        chunks = [
            sponsored,
            item["brand"],
            item["title"],
            specs_str,
            price_str,
            disc_fmt,
            rating_str,
            delivery_phrase,
            action_phrase,
        ]

    # Clean whitespace
    raw_text = " ".join(c.strip() for c in chunks if c and c.strip())

    # Simulated timestamp within the last 30 days
    base_time = datetime(2026, 1, 15, 8, 0, 0)
    rand_offset = timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    timestamp = (base_time + rand_offset).isoformat()

    return {
        "id": sample_id,
        "app": app,
        "raw_text": raw_text,
        "timestamp": timestamp,
        "ground_truth": {
            "brand": item["brand"],
            "product_title": item["title"],
            "category": item["category"],
            "specs": specs_picked,
            "price": price,
            "currency": currency,
            "mrp": mrp,
            "discount_pct": discount_pct,
            "rating": rating_val,
            "num_ratings": num_ratings,
            "action": action,
        },
    }


def generate_dataset(
    count: int = 250, output_path: str = "data/raw/screen_dumps.jsonl", seed: int = 42
) -> List[Dict[str, Any]]:
    """Generates `count` screen dumps and saves to JSONL."""
    random.seed(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    records = [generate_single_sample(i + 1) for i in range(count)]

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OCR screen dumps")
    parser.add_argument("--count", type=int, default=250, help="Number of records to generate")
    parser.add_argument(
        "--output", type=str, default="data/raw/screen_dumps.jsonl", help="Output JSONL path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    data = generate_dataset(count=args.count, output_path=args.output, seed=args.seed)
    print(f"Successfully generated {len(data)} screen dumps to {args.output}")
