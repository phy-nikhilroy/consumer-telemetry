import re
from typing import Any, Dict, Optional


# Pre-compiled regex patterns for performance and clarity

# Currency patterns with proper word boundaries and symbol definitions
CURRENCY_INR_PATTERN = re.compile(r"(?:₹|\bRs\.?|\bINR\b)", re.IGNORECASE)
CURRENCY_USD_PATTERN = re.compile(r"(?:\$|\bUSD\b)", re.IGNORECASE)

# Price numeric pattern: e.g. 1,23,456 or 123,456.78 or 1234 or 1299.99
# The grouped branch requires at least one comma group. When it allowed zero groups it
# matched the first three digits of an unseparated price ("$1299.99" read as 129.0),
# and the regex engine never fell through to the plain-digits branch.
NUMERIC_PRICE_RE = r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"

# Discount pattern: e.g. 10% off, 72% OFF, (15% off), save 20%
DISCOUNT_PATTERN = re.compile(r"(?:(?:save\s+)?(\d{1,2})%\s*off|\b(\d{1,2})%\s*savings?\b)", re.IGNORECASE)

# Rating patterns:
# 1. Decimal rating: "4.7 out of 5 stars", "4.5 out of 5", "Rating: 4.2", "4.3 / 5", "4.5★"
# Note: require non-digit before rating or start of word to avoid matching "CE 3 ★"
DECIMAL_RATING_PATTERN = re.compile(
    r"(?:(?:\b|Rating:\s*)([1-5]\.\d)\s*(?:out of 5(?:\s*stars?)?|/ 5|★|stars?\b))",
    re.IGNORECASE
)
# 2. Integer rating out of 5: e.g. "4 out of 5 stars"
INTEGER_RATING_PATTERN = re.compile(
    r"\b([1-5])\s+out of 5(?:\s*stars?)?\b",
    re.IGNORECASE
)
# 3. Unicode stars: ★★★★☆ (count filled stars)
UNICODE_STARS_PATTERN = re.compile(r"([★☆]{3,5})")

# Num ratings pattern: e.g. (2,847), (43,981 ratings), 1,234 reviews
NUM_RATINGS_PATTERN = re.compile(
    r"\(\s*([\d,]+)\s*(?:ratings?|reviews?)?\s*\)|(?:based on\s+([\d,]+)\s+ratings?)|(?:([\d,]+)\s+(?:ratings?|reviews?))",
    re.IGNORECASE
)

# User intent actions
ACTION_PATTERNS = [
    ("buy", re.compile(r"\b(?:buy\s+now|proceed\s+to\s+buy|order\s+now)\b", re.IGNORECASE)),
    ("cart", re.compile(r"\b(?:add\s+to\s+cart|added\s+to\s+cart|go\s+to\s+cart|in\s+cart)\b", re.IGNORECASE)),
    ("wishlist", re.compile(r"\b(?:add\s+to\s+wishlist|save\s+for\s+later|wishlist)\b", re.IGNORECASE)),
    ("search", re.compile(r"\b(?:search(?:ed|ing)?(?:\s+for)?|results\s+for|filters?|popular\s+pick|pickup\s+today)\b", re.IGNORECASE)),
    ("view", re.compile(r"\b(?:sponsored|product\s+details|overview|specifications|free\s+delivery|in\s+stock|ratings?|reviews?)\b", re.IGNORECASE)),
]


def _clean_number(val_str: str) -> float:
    return float(val_str.replace(",", "").strip())


# OCR confuses these letters with digits inside prices ("₹1,l9,900", "₹36,S49").
_LOOKALIKE_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5", "B": "8"})
_LOOKALIKE_CHARS = "OolI|SB"
# A currency marker or an MRP-style label, then a token of digits, separators, and lookalikes.
# The marker part is case-insensitive; the token class is not, so `s` and `b` never fold.
PRICE_TOKEN_PATTERN = re.compile(
    r"((?i:₹|\bRs\.?|\bINR\b|\$|\bUSD\b"
    r"|(?:M\.?R\.?P\.?|List\s+Price|Was|Original\s+Price)[:\s]+(?:₹|Rs\.?|INR|\$|USD)?)\s*)"
    r"([0-9OolI|SB][0-9OolI|SB,.]*)"
)


def _fold_price_token(match: "re.Match[str]") -> str:
    """Folds OCR lookalikes to digits inside one price token, leaving following words intact."""
    prefix, token = match.group(1), match.group(2)
    following = match.string[match.end():match.end() + 1]
    tail = ""
    if following.isalpha():
        # The token runs into a word ("$20Off"): its trailing lookalikes belong to that word.
        cut = len(token)
        while cut > 0 and (token[cut - 1] in _LOOKALIKE_CHARS or token[cut - 1] in ",."):
            cut -= 1
        token, tail = token[:cut], token[cut:]
    if not any(c.isdigit() for c in token):
        return prefix + token + tail
    return prefix + token.translate(_LOOKALIKE_TO_DIGIT) + tail


def fold_price_lookalikes(text: str) -> str:
    """
    Rewrites OCR letter-for-digit confusions inside price tokens: `O`/`o` to 0, `l`/`I`/`|`
    to 1, `S` to 5, `B` to 8. Only a token after a currency marker or an M.R.P.-style label
    that already holds a digit is changed, so words elsewhere are never altered.
    """
    if not text:
        return text
    return PRICE_TOKEN_PATTERN.sub(_fold_price_token, text)


# A percentage inside a payment offer ("Bank Offer 10% off on SBI Credit Card") is not the
# listing's discount. A match is skipped when one of these words sits just before it or a card
# is named just after it.
_OFFER_BEFORE_RE = re.compile(r"(?i)\b(?:bank|offer|cashback|coupon)\b")
_OFFER_AFTER_RE = re.compile(r"(?i)\bcard\b")
_OFFER_BEFORE_CHARS = 14
# A one or two character percentage token with an OCR lookalike ("(l0% off)"): `\d{1,2}%` alone
# matched its "0%" and reported no discount.
_DISCOUNT_TOKEN_RE = re.compile(r"(?<![0-9A-Za-z])([0-9OolI|SB]{1,2})(?=\s?%)")


def _fold_discount_token(match: "re.Match[str]") -> str:
    """Folds lookalikes in a percentage token that already holds a digit."""
    token = match.group(1)
    if not any(c.isdigit() for c in token):
        return token
    return token.translate(_LOOKALIKE_TO_DIGIT)

_OFFER_AFTER_CHARS = 30


def _listing_discount_match(text: str) -> Optional["re.Match[str]"]:
    """The first discount match that is not part of a payment offer, or None."""
    for match in DISCOUNT_PATTERN.finditer(text):
        before = text[max(0, match.start() - _OFFER_BEFORE_CHARS):match.start()]
        after = text[match.end():match.end() + _OFFER_AFTER_CHARS]
        if _OFFER_BEFORE_RE.search(before) or _OFFER_AFTER_RE.search(after):
            continue
        return match
    return None


def extract_price(text: str) -> Dict[str, Any]:
    """
    Extracts price, currency, mrp, and discount_pct from noisy screen text.
    Returns:
        {
            'price': float or None,
            'currency': str or None,
            'mrp': float or None,
            'discount_pct': int or None
        }
    """
    if not text:
        return {"price": None, "currency": None, "mrp": None, "discount_pct": None}

    text = fold_price_lookalikes(text)
    text = _DISCOUNT_TOKEN_RE.sub(_fold_discount_token, text)

    # Determine currency
    currency = None
    if CURRENCY_INR_PATTERN.search(text):
        currency = "INR"
    elif CURRENCY_USD_PATTERN.search(text):
        currency = "USD"

    # Extract discount percentage if present
    discount_pct: Optional[int] = None
    discount_match = _listing_discount_match(text)
    if discount_match:
        val = discount_match.group(1) or discount_match.group(2)
        if val:
            try:
                discount_pct = int(val)
            except ValueError:
                discount_pct = None

    # Find explicitly labeled MRP / list price / original price / struck-through price
    # e.g., M.R.P: ₹75,920 or MRP: $100 or List Price: $99.99 or Was: $89.00
    mrp: Optional[float] = None
    mrp_pattern = re.compile(
        r"(?:M\.?R\.?P\.?|List\s+Price|\bWas|Original\s+Price)[:\s]+(?:₹|\bRs\.?|\bINR\b|\$|\bUSD\b)?\s*("
        + NUMERIC_PRICE_RE + r")",
        re.IGNORECASE
    )
    mrp_match = mrp_pattern.search(text)
    if mrp_match:
        try:
            mrp = _clean_number(mrp_match.group(1))
        except ValueError:
            mrp = None

    # Find currency-tagged amounts: (₹|Rs|INR|$)\s*(\d[,\d.]*)
    tagged_prices = []
    symbol_price_pattern = re.compile(
        # Word boundaries keep "Rs" and "INR" from matching inside words: "stars 5imilar"
        # (OCR S to 5) once parsed as a 5 rupee price.
        r"(?:₹|\bRs\.?|\bINR\b|\$)\s*(" + NUMERIC_PRICE_RE + r")",
        re.IGNORECASE
    )
    for m in symbol_price_pattern.finditer(text):
        try:
            val = _clean_number(m.group(1))
            tagged_prices.append((m.start(), val))
        except ValueError:
            continue

    price: Optional[float] = None
    if tagged_prices:
        # If we found an MRP, the selling price is typically the other tagged price
        if mrp is not None:
            # Pick the tagged price that is distinct from MRP, or lowest if multiple
            non_mrp = [p[1] for p in tagged_prices if abs(p[1] - mrp) > 0.01]
            if non_mrp:
                price = non_mrp[0]
            else:
                price = mrp
        else:
            # If no MRP label, usually first tagged price is the selling price
            price = tagged_prices[0][1]
            if len(tagged_prices) > 1 and discount_pct and discount_pct > 0:
                # If there are two prices, higher is likely mrp, lower is price
                sorted_prices = sorted([p[1] for p in tagged_prices])
                price = sorted_prices[0]
                mrp = sorted_prices[-1]
    else:
        # Fallback: look for standalone price pattern if currency was detected
        if currency:
            standalone_pattern = re.compile(r"\b(" + NUMERIC_PRICE_RE + r")\b")
            candidates = []
            for m in standalone_pattern.finditer(text):
                try:
                    candidates.append(_clean_number(m.group(1)))
                except ValueError:
                    continue
            if candidates:
                price = candidates[0]

    # If MRP is not explicit but we have a price
    if mrp is None and price is not None:
        if discount_pct is not None and discount_pct > 0:
            mrp = round(price / (1 - (discount_pct / 100.0)), 2)
        else:
            mrp = price

    if discount_pct is None and price is not None and mrp is not None and mrp > price:
        discount_pct = int(round(((mrp - price) / mrp) * 100))

    return {
        "price": price,
        "currency": currency,
        "mrp": mrp,
        "discount_pct": discount_pct if discount_pct is not None else 0
    }


def extract_rating(text: str) -> Dict[str, Any]:
    """
    Extracts rating (float, 0-5) and review/rating count (int).
    Returns:
        {
            'rating': float or None,
            'num_ratings': int or None
        }
    """
    if not text:
        return {"rating": None, "num_ratings": None}

    rating: Optional[float] = None
    num_ratings: Optional[int] = None

    # Check decimal rating first
    dec_match = DECIMAL_RATING_PATTERN.search(text)
    if dec_match:
        try:
            rating = float(dec_match.group(1))
        except ValueError:
            rating = None
    else:
        int_match = INTEGER_RATING_PATTERN.search(text)
        if int_match:
            try:
                rating = float(int_match.group(1))
            except ValueError:
                rating = None

    # Check star symbols if not found
    if rating is None:
        star_match = UNICODE_STARS_PATTERN.search(text)
        if star_match:
            stars_str = star_match.group(1)
            filled_count = stars_str.count("★")
            total_count = len(stars_str)
            if total_count >= 3:
                # Rough estimate: filled count out of total scaled to 5.0
                rating = round((filled_count / total_count) * 5.0, 1)

    # Extract number of ratings
    for m in NUM_RATINGS_PATTERN.finditer(text):
        matched_str = m.group(1) or m.group(2) or m.group(3)
        if matched_str:
            clean_str = matched_str.replace(",", "").strip()
            # Avoid picking up year like 2026 or small single numbers unless in parens
            if clean_str.isdigit():
                val = int(clean_str)
                # Discard if it looks like a year (e.g. 2024-2027) unless explicitly in parens
                if 2020 <= val <= 2030 and not m.group(1):
                    continue
                num_ratings = val
                break

    return {
        "rating": rating,
        "num_ratings": num_ratings
    }


def extract_action(text: str) -> str:
    """
    Determines user intent action from the text.
    Returns one of: 'buy', 'cart', 'wishlist', 'search', 'view'
    Defaults to 'view'.
    """
    if not text:
        return "view"

    for action_label, pattern in ACTION_PATTERNS:
        if pattern.search(text):
            return action_label

    return "view"
