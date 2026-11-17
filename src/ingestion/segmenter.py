"""
Screen segmenter: splits one multi-listing screen into one text segment per listing.

A listing ends at its action or stock line: a line matching a buy or cart pattern of
`regex_extractor.ACTION_PATTERNS`, or a stock phrase. Lines after the last end marker join
the last listing. A single-line input is returned unchanged as the only segment, so every
v1.0 one-product string still flows through the pipeline exactly as before.

OCR noise can turn "Buy Now" into "8uy Now" or "BuyNow". When a line does not match as
written, it is matched again after undoing the digit-for-letter confusions and removing
spaces, so noisy screens still split at the right place.
"""

from __future__ import annotations

import re
from typing import List

from src.ingestion.regex_extractor import ACTION_PATTERNS

END_ACTIONS = ("buy", "cart")

STOCK_PATTERN = re.compile(r"\b(?:in\s+stock|only\s+\d+\s+left\s+in\s+stock)\b", re.IGNORECASE)

END_PATTERNS: List[re.Pattern] = [p for name, p in ACTION_PATTERNS if name in END_ACTIONS] + [STOCK_PATTERN]

# Undo the OCR confusions that replace letters with digits.
_DIGIT_TO_LETTER = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b"})

# The end phrases with their spaces removed, for lines whose spaces OCR dropped.
_COMPACT_END_PHRASES = (
    "buynow", "proceedtobuy", "ordernow", "addtocart", "addedtocart", "gotocart", "instock",
)


def _compact(line: str) -> str:
    """Lower-cases the line, undoes digit confusions, and drops whitespace."""
    return re.sub(r"\s+", "", line.lower().translate(_DIGIT_TO_LETTER))


def is_end_marker(line: str) -> bool:
    """Returns whether a line closes a listing (an action or stock line)."""
    if any(p.search(line) for p in END_PATTERNS):
        return True
    compact = _compact(line)
    return any(phrase in compact for phrase in _COMPACT_END_PHRASES)


def segment_screen(raw_text: str) -> List[str]:
    """
    Splits screen text into one segment per listing.

    A single-line input (no newline) returns `[raw_text]`. Otherwise blank lines are
    skipped, the lines of each listing are joined with a space, and a listing ends at an
    end-marker line. Lines after the last end marker join the last listing; a screen with
    no end marker is one segment. Whitespace-only multi-line input returns an empty list.
    """
    if "\n" not in raw_text:
        return [raw_text]

    segments: List[str] = []
    current: List[str] = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        current.append(line)
        if is_end_marker(line):
            segments.append(" ".join(current))
            current = []

    if current:
        if segments:
            segments[-1] = segments[-1] + " " + " ".join(current)
        else:
            segments.append(" ".join(current))
    return segments
