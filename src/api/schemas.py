"""
Request and response models for the consumer telemetry API.

Every endpoint returns one of these Pydantic models, so the OpenAPI document at /docs is
generated from the same definitions that validate the responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.ingestion.pipeline import ExtractedProduct

# Identifiers end up in SQL parameters and in session IDs, so they are restricted to a
# conservative character set rather than accepted as arbitrary strings.
USER_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"
APP_NAME_PATTERN = r"^[a-z0-9_\-]{1,32}$"


class ScreenTextPayload(BaseModel):
    """One raw screen capture from a shopping app."""

    raw_text: str = Field(..., min_length=1, max_length=2000, description="Raw OCR or screen-reader text")
    user_id: str = Field(..., pattern=USER_ID_PATTERN, description="The shopper the capture belongs to")
    app_name: str = Field("unknown", pattern=APP_NAME_PATTERN, description="Source app, e.g. amazon_in")
    timestamp: Optional[datetime] = Field(
        None, description="When the screen was captured; defaults to the serving clock"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "raw_text": "Samsung Galaxy S24 Ultra 256GB Titanium $1,299.99 4.7★ Add to Cart",
                    "user_id": "user_042",
                    "app_name": "amazon_us",
                }
            ]
        }
    }


class IngestResponse(BaseModel):
    """The stored event and the product extracted from the screen text."""

    event_id: str
    user_id: str
    timestamp: datetime
    app_name: str
    stored_action: str = Field(..., description="Funnel step stored in the warehouse: view, cart, or buy")
    session_id: str
    is_new_user: bool
    cross_app_comparisons_detected: int
    product: ExtractedProduct


class HealthResponse(BaseModel):
    """Service status and warehouse size."""

    status: str
    events: int
    users: int
    products: int
    live_events: int
    latest_event_timestamp: Optional[datetime]
    reference_date: datetime
    clock_mode: str
    warehouse: str
    models: Dict[str, str]


class PropensityResponse(BaseModel):
    """Probability that the user buys anything in the next 7 days."""

    user_id: str
    purchase_probability: float = Field(..., ge=0.0, le=1.0)
    risk_segment: str = Field(..., description="High, Medium, or Low")
    reference_date: datetime
    history_events: int
    low_history: bool = Field(..., description="True when the user has too few events for reliable features")


class AppUsage(BaseModel):
    app_name: str
    events: int
    purchases: int
    share: float


class CategoryCount(BaseModel):
    category_l1: str
    category_l2: str
    events: int


class LifetimeValue(BaseModel):
    horizon_days: int
    predicted_clv_inr: float
    predicted_purchases: float
    predicted_avg_order_value_inr: float
    probability_alive: float
    order_value_source: str
    purchase_estimate_source: str


class SegmentAssignment(BaseModel):
    cluster: int
    segment_name: str
    description: Optional[str] = None


class ProfileResponse(BaseModel):
    """Everything known and predicted about one user."""

    user_id: str
    reference_date: datetime
    total_events: int
    total_purchases: int
    total_spend_inr: float
    sessions: int
    first_seen: Optional[datetime]
    last_seen: Optional[datetime]
    purchase_probability: float
    risk_segment: str
    app_switch_probability: float
    likely_to_switch_app: bool
    segment: SegmentAssignment
    lifetime_value: Optional[LifetimeValue] = Field(
        None, description="Null for users who have never purchased"
    )
    top_categories: List[CategoryCount]
    app_usage: List[AppUsage]
    low_history: bool


class SegmentProfile(BaseModel):
    cluster: int
    segment_name: str
    description: Optional[str]
    users: int
    share: float
    mean_purchase_probability: float
    mean_purchases_30d: float
    mean_spend_30d_inr: float
    mean_app_entropy: float
    mean_cross_app_comparisons: float


class SegmentsResponse(BaseModel):
    """All segments, recomputed from current features at the reference date."""

    reference_date: datetime
    total_users: int
    segments: List[SegmentProfile]


class LastPurchase(BaseModel):
    user_id: str
    timestamp: datetime
    app_name: str
    product_title: Optional[str]
    price: Optional[float]
    currency: Optional[str]


class ActivityResponse(BaseModel):
    """Recent activity, used by the terminal monitor and the dashboard."""

    reference_date: datetime
    window_hours: float
    events_in_window: int
    active_users_in_window: int
    purchases_in_window: int
    last_purchase: Optional[LastPurchase]
    events: int
    users: int
    products: int
    live_events: int


class EventRecord(BaseModel):
    event_id: str
    user_id: str
    timestamp: datetime
    app_name: str
    action: str
    price: Optional[float]
    currency: Optional[str]
    session_id: Optional[str]
    canonical_product_id: Optional[str]
    brand: Optional[str]
    product_title: Optional[str]
    category_l2: Optional[str]


class UserSummary(BaseModel):
    user_id: str
    events: int
    last_seen: datetime
