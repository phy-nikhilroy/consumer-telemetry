"""Tests for the API request and response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import ValidationError
import pytest

from src.api.schemas import PropensityResponse, ScreenTextPayload


def test_payload_defaults_and_example():
    payload = ScreenTextPayload(raw_text="Apple iPhone 15 ₹68,330", user_id="user_001")
    assert payload.app_name == "unknown"
    assert payload.timestamp is None
    example = ScreenTextPayload.model_config["json_schema_extra"]["examples"][0]
    assert ScreenTextPayload(**example).user_id == "user_042"


@pytest.mark.parametrize(
    "fields",
    [
        {"raw_text": "", "user_id": "u1"},
        {"raw_text": "x" * 2001, "user_id": "u1"},
        {"raw_text": "x", "user_id": "has space"},
        {"raw_text": "x", "user_id": "semi;colon"},
        {"raw_text": "x", "user_id": "u" * 65},
        {"raw_text": "x", "user_id": "u1", "app_name": "UPPER"},
    ],
)
def test_payload_rejects_invalid_fields(fields):
    with pytest.raises(ValidationError):
        ScreenTextPayload(**fields)


def test_probability_bounds():
    common = {"user_id": "u", "risk_segment": "Low", "reference_date": datetime(2026, 1, 1),
              "history_events": 1, "low_history": True}
    PropensityResponse(purchase_probability=0.5, **common)
    with pytest.raises(ValidationError):
        PropensityResponse(purchase_probability=1.5, **common)
