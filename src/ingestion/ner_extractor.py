"""
Linguistic extraction of brand, product title, specifications, and NER price mentions.

Brand detection combines three sources, in order of trust:

1. A phrase matcher over a known-brand vocabulary, extended at runtime with any brand
   gazetteer registered through `register_brands()`.
2. A guard that discards a vocabulary match when the listing names that brand only as the
   device it is compatible with ("Case for Samsung Galaxy S6"). Without it, accessory
   listings were systematically attributed to the brand of the phone they fit.
3. An open-vocabulary fallback that takes the leading word of the listing title, which on
   real marketplace data is the seller's brand far more often than any closed list can
   anticipate.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
import spacy
from spacy.matcher import PhraseMatcher

from src.ingestion.regex_extractor import extract_price, extract_rating, extract_action

# Load spaCy English pipeline
# Tagger, parser, lemmatizer and attribute ruler are excluded: this module reads only tokens,
# punctuation flags, and the statistical NER, so they were pure cost.
NLP_EXCLUDED_COMPONENTS = ["tagger", "parser", "lemmatizer", "attribute_ruler"]
nlp = spacy.load("en_core_web_sm", exclude=NLP_EXCLUDED_COMPONENTS)


def _get_nlp() -> Any:
    """Returns the shared spaCy pipeline (tok2vec and ner, without the unused components)."""
    return nlp

# Brand evidence labels, ordered from strongest to weakest. The pipeline uses these to
# score how far an extracted brand can be trusted.
BRAND_SOURCE_LEADING_MATCH = "known_brand_leading"
BRAND_SOURCE_MATCH = "known_brand"
BRAND_SOURCE_LEADING_TOKEN = "leading_token"

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

# Canonical spelling for every brand the matcher knows, keyed by lower case. Starts as
# the curated vocabulary and grows when a gazetteer is registered.
_known_brands: Dict[str, str] = {brand.lower(): brand for brand in POPULAR_BRANDS}

# Initialize PhraseMatcher with brand list
brand_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
brand_patterns = [nlp.make_doc(brand) for brand in POPULAR_BRANDS]
brand_matcher.add("BRAND", brand_patterns)

# Words that, when they precede a brand, mark it as the device an accessory fits rather
# than the maker of the product: "for", "compatible with", "fits", "designed for",
# "replacement for", "works with".
COMPATIBILITY_CUES = frozenset({"for", "compatible", "fits", "fit", "replacement", "works", "designed"})
COMPATIBILITY_WINDOW_TOKENS = 3

# Leading title words that describe the product rather than name its maker. The
# open-vocabulary fallback refuses these, so "Leather bands compatible with..." does not
# yield the brand "Leather". "generic" and "unbranded" are how marketplaces mark an
# unbranded product, which is an absence of brand rather than a brand.
GENERIC_LEADING_WORDS = frozenset({
    "a", "an", "the", "for", "new", "and", "with", "by", "of", "in", "on", "to",
    "generic", "unbranded", "unknown", "brand", "official", "original", "genuine",
    "case", "cases", "cover", "covers", "battery", "leather", "premium", "slim", "bumper",
    "compatible", "designed", "car", "sport", "sports", "replacement", "universal",
    "wireless", "magnetic", "clear", "glass", "tempered", "screen", "phone", "phones",
    "cell", "mobile", "charger", "charging", "cable", "cables", "holder", "stand", "mount",
    "band", "bands", "strap", "straps", "protector", "protectors", "pack", "set", "kit",
    "luxury", "cute", "fashion", "designer", "personalized", "custom", "women", "womens",
    "men", "mens", "kids", "girls", "boys", "black", "white", "red", "blue", "pink",
    "green", "gold", "silver", "purple", "brown", "rose", "grey", "gray", "yellow",
    "thin", "ultra", "heavy", "durable", "shockproof", "waterproof", "rugged", "wallet",
    "flip", "folio", "soft", "hard", "silicone", "tpu", "metal", "crystal", "glitter",
    "bling", "floral", "flower", "protective", "privacy", "anti", "matte", "upgraded",
    "upgrade", "latest", "extended", "portable", "fast", "quick", "usb", "magsafe", "lens",
    "camera", "ring", "grip", "first", "name", "capacity", "awesome", "funny", "smart",
    "watch", "headphones", "earbuds", "speaker", "adapter", "dock", "pouch", "sleeve",
    "skin", "skins", "film", "stylus", "pen", "keyboard", "mouse", "tablet", "laptop",
    "shoes", "shirt", "men's", "women's", "kid's", "unisex", "classic", "vintage",
    "lightweight", "heavy-duty", "dual", "triple", "double", "single", "mini", "pro",
    "max", "plus", "lite", "wired", "bluetooth", "led", "lcd", "oled", "digital",
})

# Device and product-line names. When one leads a title the listing is almost always an
# accessory for that device, so the leading word cannot be read as the seller's brand.
DEVICE_LINE_WORDS = frozenset({
    "iphone", "ipad", "ipod", "airpods", "macbook", "imac", "iwatch", "galaxy", "pixel",
    "note", "moto", "kindle", "surface", "xperia", "redmi", "nexus",
})

LEADING_TOKEN_PATTERN = re.compile(r"^[\W_]*([A-Za-z][\w&'\-\.]*)")
UNBRACKETED_COUNT_PATTERN = re.compile(r"^\s*[\d,]+\s+(?:ratings?|reviews?)\b", re.IGNORECASE)
LEADING_BRACKET_PATTERN = re.compile(r"^\s*(?:[\[(][^\])]{0,20}[\])]\s*)+")
ALL_CAPS_WORD = re.compile(r"^[A-Z][A-Z&'\-]+$")

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


def register_brands(brands: Iterable[str]) -> int:
    """
    Extends the brand matcher with an external gazetteer, such as a marketplace's
    seller or store names.

    Names already known are skipped. Returns how many new brands were added. The
    extension is process-wide and lasts until `reset_registered_brands()`.
    """
    added = 0
    patterns = []
    for raw in brands:
        if not raw:
            continue
        name = str(raw).strip()
        key = name.lower()
        if len(key) < 2 or key in _known_brands or key in GENERIC_LEADING_WORDS:
            continue
        _known_brands[key] = name
        patterns.append(nlp.make_doc(name))
        added += 1
    if patterns:
        brand_matcher.add("BRAND", patterns)
    return added


def reset_registered_brands() -> None:
    """Restores the brand matcher to the curated `POPULAR_BRANDS` vocabulary only."""
    global brand_matcher
    _known_brands.clear()
    _known_brands.update({brand.lower(): brand for brand in POPULAR_BRANDS})
    brand_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    brand_matcher.add("BRAND", [nlp.make_doc(brand) for brand in POPULAR_BRANDS])


def _strip_listing_noise(text: str) -> str:
    """
    Removes ratings, review counts, prices, discounts, action phrases, and leading
    banners from a listing, leaving the product title and its specifications.
    """
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
    return NOISE_PREFIXES.sub("", cleaned).strip()


def _in_compatibility_context(doc: Any, start: int) -> bool:
    """Returns whether a span starting at `start` follows a compatibility cue word."""
    window_start = max(0, start - COMPATIBILITY_WINDOW_TOKENS)
    return any(tok.lower_ in COMPATIBILITY_CUES for tok in doc[window_start:start])


# OCR confuses these digits with letters inside words ("App1e", "Sams0ng").
_DIGIT_TO_LOOKALIKE = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "|": "l"})
# The leading word for brand recovery may start with a digit OCR put in place of a letter
# ("5amsung").
RECOVERY_TOKEN_PATTERN = re.compile(r"^[\W_]*([A-Za-z0-9][\w&'\-\.]*)")
# Fuzzy (one-edit) brand recovery only for brands this long: a one-edit near miss of a 2 or 3 letter
# brand ("LX" for LG, "MSX" for MSI) is as likely another maker as a misread.
BRAND_RECOVERY_MIN_LENGTH = 4


def _within_one_edit(a: str, b: str) -> bool:
    """Whether `a` and `b` differ by at most one insertion, deletion, or substitution."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    if len(a) == len(b):
        return a[i + 1:] == b[i + 1:]
    return a[i:] == b[i + 1:]


def _recover_known_brand(title_text: str) -> Optional[str]:
    """
    Recovers a known brand from an OCR-damaged leading word.

    The word is first folded (digits read as the letters they resemble); an exact known
    brand after folding is returned ("J8L" is JBL), otherwise the single known brand within
    one edit of it. The one-edit step needs at least `BRAND_RECOVERY_MIN_LENGTH`
    characters in both the word and the brand. Generic or device-line words are never
    recovered.
    """
    match = RECOVERY_TOKEN_PATTERN.match(title_text)
    if not match:
        return None
    lowered = match.group(1).rstrip(".-'").lower()
    folded = lowered.translate(_DIGIT_TO_LOOKALIKE)
    if (
        # A brand spelled correctly was already judged by the matcher (and possibly
        # rejected as a compatibility target); recovery must not readmit it.
        lowered in _known_brands
        or len(folded) < 2
        or not folded.isalpha()
        or folded in GENERIC_LEADING_WORDS
        or folded in DEVICE_LINE_WORDS
    ):
        return None
    if folded in _known_brands:
        return _known_brands[folded]
    if len(folded) < BRAND_RECOVERY_MIN_LENGTH:
        return None
    candidates = [
        name for key, name in _known_brands.items()
        if len(key) >= BRAND_RECOVERY_MIN_LENGTH and key.isalpha() and _within_one_edit(folded, key)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _leading_token_brand(title_text: str) -> Optional[str]:
    """
    Reads the leading word of a listing title as an open-vocabulary brand guess.

    Two consecutive all-capital words are kept together ("MODOS LOGICOS", "CG MOBILE"),
    since that is how multi-word seller names are commonly written in titles.
    """
    match = LEADING_TOKEN_PATTERN.match(title_text)
    if not match:
        return None

    token = match.group(1).rstrip(".-'")
    lowered = token.lower()
    if (
        len(token) < 2
        # Alphanumeric leading words are model numbers ("R2111", "ip12"), not makers.
        or any(ch.isdigit() for ch in token)
        or lowered in GENERIC_LEADING_WORDS
        or lowered in DEVICE_LINE_WORDS
        or lowered in _known_brands
    ):
        # A known brand reaching this point was already rejected by the compatibility
        # guard, so it must not be readmitted through the fallback.
        return None

    if ALL_CAPS_WORD.match(token):
        rest = title_text[match.end():].split()
        if rest and ALL_CAPS_WORD.match(rest[0]):
            return f"{token} {rest[0]}"
    return token


def extract_brand_with_source(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts a brand and reports which kind of evidence produced it.

    Returns `(brand, source)`, where source is one of `BRAND_SOURCE_LEADING_MATCH`
    (a known brand at the start of the title), `BRAND_SOURCE_MATCH` (a known brand
    elsewhere in the title), `BRAND_SOURCE_LEADING_TOKEN` (the open-vocabulary
    fallback), or `None` when no brand was found.
    """
    if not text:
        return None, None

    title_text = _strip_listing_noise(text)
    # Brand position matters here in a way it does not for title extraction, so bare
    # review counts ("1,23,456 ratings") and leading bracketed pack sizes ("[2 Pack]")
    # are also removed before deciding what leads the title.
    title_text = UNBRACKETED_COUNT_PATTERN.sub("", title_text)
    title_text = LEADING_BRACKET_PATTERN.sub("", title_text).strip()
    if not title_text:
        return None, None

    doc = nlp.make_doc(title_text)
    matches = brand_matcher(doc)

    # A brand named anywhere as a compatibility target is a device reference throughout
    # the listing: "Galaxy S10 Case ... Cover for Samsung Galaxy S10" is not a Samsung
    # product just because "Samsung" also appears unqualified.
    compatibility_brands = {
        doc[start:end].text.lower()
        for _, start, end in matches
        if _in_compatibility_context(doc, start)
    }
    usable = [
        (start, end)
        for _, start, end in matches
        if doc[start:end].text.lower() not in compatibility_brands
    ]

    leading_match: Optional[str] = None
    later_match: Optional[str] = None
    if usable:
        # Earliest mention wins, then the longest span at that position (Fire-Boltt
        # over Fire).
        usable.sort(key=lambda span: (span[0], -(span[1] - span[0])))
        start, end = usable[0]
        surface = doc[start:end].text
        brand = _known_brands.get(surface.lower(), surface)
        if all(not tok.is_alpha and not tok.is_digit for tok in doc[:start]):
            leading_match = brand
        else:
            later_match = brand

    if leading_match:
        return leading_match, BRAND_SOURCE_LEADING_MATCH

    recovered = _recover_known_brand(title_text)
    if recovered:
        return recovered, BRAND_SOURCE_LEADING_MATCH

    # A known brand that appears only mid-title is usually the device an accessory is
    # for, even without an explicit cue ("Bonoma Samsung Galaxy S8 Case"). On real
    # Amazon listings it was correct 0.3 percent of the time, so the title's own leading
    # word is preferred and the mid-title match is the last resort.
    fallback = _leading_token_brand(title_text)
    if fallback:
        return fallback, BRAND_SOURCE_LEADING_TOKEN
    if later_match:
        return later_match, BRAND_SOURCE_MATCH
    return None, None


def extract_brand(text: str) -> Optional[str]:
    """
    Extracts the product's brand from listing text.

    See `extract_brand_with_source` for the evidence model; this wrapper returns only
    the brand.
    """
    return extract_brand_with_source(text)[0]


def extract_price_mentions(text: str) -> List[float]:
    """
    Returns every monetary amount spaCy's statistical NER tags as MONEY.

    This is an extractor independent of the regex price parser, so agreement between
    the two is evidence that the extracted price is genuinely a price.
    """
    if not text:
        return []

    return _money_amounts(nlp(text))


def _money_amounts(doc: Any) -> List[float]:
    """The numeric amounts inside the MONEY entities of a processed doc."""
    amounts: List[float] = []
    for ent in doc.ents:
        if ent.label_ != "MONEY":
            continue
        for number in re.finditer(r"\d[\d,]*(?:\.\d+)?", ent.text):
            try:
                amounts.append(float(number.group(0).replace(",", "")))
            except ValueError:
                continue
    return amounts


def extract_price_mentions_batch(texts: List[str], batch_size: int = 256) -> List[List[float]]:
    """`extract_price_mentions` for many texts through `nlp.pipe`, in input order."""
    results: List[List[float]] = [[] for _ in texts]
    positions = [i for i, t in enumerate(texts) if t]
    docs = nlp.pipe([texts[i] for i in positions], batch_size=batch_size)
    for i, doc in zip(positions, docs):
        results[i] = _money_amounts(doc)
    return results


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

    cleaned = _strip_listing_noise(text)

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
    doc = nlp.make_doc(cleaned)  # tokens and punctuation flags need only the tokenizer
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
