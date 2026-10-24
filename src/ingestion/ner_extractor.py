import re
from typing import Any, Dict, List, Optional, Tuple
import spacy
from spacy.matcher import PhraseMatcher

from src.ingestion.regex_extractor import extract_price, extract_rating, extract_action

# Load spaCy English pipeline
nlp = spacy.load("en_core_web_sm")

# Curated list of 60+ common consumer electronics and e-commerce brands
POPULAR_BRANDS = [
    # Smartphones & Computing
    "Apple", "Samsung", "Google", "OnePlus", "Xiaomi", "Redmi", "Realme", "Oppo", "Vivo",
    "Motorola", "Nothing", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Microsoft",
    # Audio & Wearables
    "Sony", "boAt", "Noise", "Fire-Boltt", "Boult", "JBL", "Bose", "Sennheiser",
    "Marshall", "Skullcandy", "Zebronics", "pTron", "Mivi", "Amazfit",
    # Lifestyle, Apparel & Footwear
    "Nike", "Adidas", "Puma", "Reebok", "Under Armour", "Levi's", "Zara", "H&M",
    "Tommy Hilfiger", "Calvin Klein", "Fossil", "Casio", "Titan", "Fastrack",
    # Home & Appliances
    "LG", "Whirlpool", "Dyson", "Philips", "Panasonic", "Havells", "Bajaj", "Prestige",
    "Eureka Forbes", "Mi", "AmazonBasics", "Logitech", "SanDisk", "Anker"
]

# Initialize PhraseMatcher with brand list
brand_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
brand_patterns = [nlp.make_doc(brand) for brand in POPULAR_BRANDS]
brand_matcher.add("BRAND", brand_patterns)

# Common specifications regex patterns
SPECS_PATTERNS = [
    re.compile(r"\b\d+\s*(?:GB|TB|MB)\b", re.IGNORECASE),  # Storage/RAM: 128 GB, 16GB, 1TB
    re.compile(r"\b(?:4G|5G|LTE|Wi-Fi|Bluetooth|NFC)\b", re.IGNORECASE),  # Connectivity
    re.compile(r"\b(?:AMOLED|OLED|Retina|FHD\+?|4K|QLED)\b", re.IGNORECASE),  # Display
    re.compile(r"\b(?:Active Noise Cancellation|ANC|True Wireless|TWS)\b", re.IGNORECASE),
    re.compile(r"\b(?:Black|White|Blue|Green|Titanium|Silver|Midnight|Starlight|Purple|Red|Space Grey)\b", re.IGNORECASE),
    re.compile(r"\((?:[\w\s,]+)\)"),  # Content in parens: (128 GB), (Midnight Blue)
]

# Noise tokens to strip out from product titles
NOISE_PREFIXES = re.compile(
    r"^(?:Sponsored\b|Deal\s+of\s+the\s+Day|Limited\s+time\s+deal|Best\s+seller\b|Amazon\'s\s+Choice\b|Choice\b|Popular\s+pick\b|Overall\s+Pick\b|Featured\b)\s*",
    re.IGNORECASE
)


def extract_brand(text: str) -> Optional[str]:
    """
    Extracts brand from text using spaCy PhraseMatcher.
    """
    if not text:
        return None

    doc = nlp(text)
    matches = brand_matcher(doc)
    if matches:
        # Sort matches by length (prefer BoAt / Fire-Boltt over shorter substrings)
        matches.sort(key=lambda m: m[2] - m[1], reverse=True)
        match_id, start, end = matches[0]
        # Return properly capitalized canonical name
        matched_span = doc[start:end].text
        for brand in POPULAR_BRANDS:
            if brand.lower() == matched_span.lower():
                return brand
        return matched_span

    return None


def extract_specs(text: str) -> List[str]:
    """
    Extracts technical specifications and variant attributes from text.
    """
    if not text:
        return []

    specs = set()
    for pattern in SPECS_PATTERNS:
        for match in pattern.finditer(text):
            val = match.group(0).strip()
            # Filter out ratings in parens like (2,847)
            if re.match(r"^\(\s*[\d,]+\s*\)$", val):
                continue
            if len(val) > 1:
                specs.add(val)

    return sorted(list(specs))


def extract_product_title(text: str, brand: Optional[str] = None) -> Optional[str]:
    """
    Extracts the core product title noun phrase.
    Locates the title around the brand name or by removing ratings, prices, and banners.
    """
    if not text:
        return None

    cleaned = text

    # Remove rating substrings: "4.7 out of 5 stars", "4.5★", "★★★★☆"
    cleaned = re.sub(r"\b[1-5](?:\.\d)?\s*(?:out of 5(?:\s*stars?)?|/ 5|★|stars?\b)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[★☆]{3,5}", "", cleaned)
    # Remove review counts in parens e.g. (2,847) or (43,981)
    cleaned = re.sub(r"\(\s*[\d,]+\s*(?:ratings?|reviews?)?\s*\)", "", cleaned, flags=re.IGNORECASE)

    # Remove price and MRP strings
    cleaned = re.sub(r"(?:M\.?R\.?P\.?|List\s+Price|Was)?[:\s]*(?:₹|\$|\bRs\.?|\bINR\b|\bUSD\b)\s*[\d,]+(?:\.\d+)?", "", cleaned, flags=re.IGNORECASE)
    # Remove discount expressions: (10% off), 25% off
    cleaned = re.sub(r"\(?\d{1,2}%\s*(?:off|savings?)\)?", "", cleaned, flags=re.IGNORECASE)

    # Remove delivery / action phrases
    cleaned = re.sub(
        r"\b(?:Buy Now|Add to Cart|Proceed to buy|Order now|Save for later|Add to Wishlist|Pickup today|See options|FREE delivery(?:\s+[A-Za-z]+,?\s+\d+.*?)?|In Stock|Compare with similar items|Popular pick in.*?|Overview|Specifications)\b",
        "", cleaned, flags=re.IGNORECASE
    )

    # Strip noisy leading banners
    cleaned = NOISE_PREFIXES.sub("", cleaned).strip()

    # Remove specs in parens like (128 GB)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned).strip()

    # Remove brand name if present
    if brand:
        brand_regex = re.compile(r"\b" + re.escape(brand) + r"\b", re.IGNORECASE)
        cleaned = brand_regex.sub("", cleaned).strip()

    # Remove common spec keywords
    cleaned = re.sub(r"\b(?:4G|5G|LTE|Bluetooth|Wi-Fi|ANC|TWS|Active Noise Cancellation|True Wireless Earbuds)\b", "", cleaned, flags=re.IGNORECASE).strip()

    # Clean punctuation and extra whitespace
    cleaned = re.sub(r"^[,\-\|\s]+|[,\-\|\s]+$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return None

    # Keep the primary product name chunk (first 1 to 4 noun tokens)
    doc = nlp(cleaned)
    tokens = [t.text for t in doc if not t.is_punct and t.text.strip()]
    if tokens:
        return " ".join(tokens[:4]).strip()

    return cleaned


def extract_all(raw_text: str) -> Dict[str, Any]:
    """
    Unified extraction pipeline combining rule-based regex extractors
    and spaCy linguistic/phrase matchers.
    """
    if not raw_text:
        return {
            "brand": None,
            "product_title": None,
            "specs": [],
            "price": None,
            "currency": None,
            "mrp": None,
            "discount_pct": None,
            "rating": None,
            "num_ratings": None,
            "action": "view"
        }

    # 1. Regex deterministic extraction
    price_info = extract_price(raw_text)
    rating_info = extract_rating(raw_text)
    action = extract_action(raw_text)

    # 2. Linguistic and NER extraction
    brand = extract_brand(raw_text)
    product_title = extract_product_title(raw_text, brand=brand)
    specs = extract_specs(raw_text)

    return {
        "brand": brand,
        "product_title": product_title,
        "specs": specs,
        "price": price_info.get("price"),
        "currency": price_info.get("currency"),
        "mrp": price_info.get("mrp"),
        "discount_pct": price_info.get("discount_pct"),
        "rating": rating_info.get("rating"),
        "num_ratings": rating_info.get("num_ratings"),
        "action": action
    }
