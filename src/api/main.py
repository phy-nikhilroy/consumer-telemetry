"""
FastAPI service: screen-text ingestion and live behavioral predictions.

Endpoints
    GET  /health                     service status and warehouse size
    POST /ingest                     raw screen text -> extracted product -> stored event
    GET  /user/{user_id}/propensity  7-day purchase probability and risk band
    GET  /user/{user_id}/profile     CLV, segment, switching risk, top categories, app usage
    GET  /analytics/segments         every segment with user counts, from current features
    GET  /analytics/activity         trailing-window activity and the latest purchase
    GET  /events/recent              the latest ingested or simulated events
    GET  /users                      users ordered by recent activity

Run locally with
    .venv/bin/uvicorn src.api.main:app --reload

Configuration comes from environment variables, so the same image runs unchanged on a
laptop, in Docker, or on a free hosting tier:
    CT_SERVING_DB      serving warehouse path (default data/processed/serving.duckdb)
    CT_SOURCE_DB       batch warehouse cloned on first start (default data/processed/consumer.duckdb)
    CT_ARTIFACTS_DIR   model artifacts directory (default src/models/artifacts)
    CT_CLOCK           "simulated" (default) or "wall"
    CT_BOOTSTRAP       "1" (default) to create the serving warehouse if it is missing

The endpoints are synchronous functions: extraction and DuckDB calls block, and FastAPI
runs synchronous endpoints in a worker thread pool rather than on the event loop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import threading
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.responses import RedirectResponse
import polars as pl

from src.api.schemas import (
    USER_ID_PATTERN,
    ActivityResponse,
    EventRecord,
    HealthResponse,
    IngestResponse,
    ProfileResponse,
    PropensityResponse,
    ScreenTextPayload,
    SegmentsResponse,
    UserSummary,
)
from src.api.scoring import ModelRegistry, UserScorer, compute_user_features, risk_segment
from src.api.warehouse import CLOCK_SIMULATED, ServingWarehouse, ensure_serving_warehouse
from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_DB_PATH,
    LOW_HISTORY_EVENT_THRESHOLD,
    SERVING_DB_PATH,
)
from src.ingestion.pipeline import process

API_VERSION = "1.0.0"
WARM_UP_TEXT = "Samsung Galaxy S24 Ultra 256GB Titanium $1,299.99 4.7★ Add to Cart"


@dataclass
class Settings:
    """Runtime configuration, read from the environment unless given explicitly."""

    db_path: Path
    source_db_path: Path
    artifacts_dir: Path
    clock_mode: str
    bootstrap: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Builds settings from CT_* environment variables with project defaults."""
        return cls(
            db_path=Path(os.environ.get("CT_SERVING_DB", str(SERVING_DB_PATH))),
            source_db_path=Path(os.environ.get("CT_SOURCE_DB", str(DEFAULT_DB_PATH))),
            artifacts_dir=Path(os.environ.get("CT_ARTIFACTS_DIR", str(ARTIFACTS_DIR))),
            clock_mode=os.environ.get("CT_CLOCK", CLOCK_SIMULATED),
            bootstrap=os.environ.get("CT_BOOTSTRAP", "1") != "0",
        )


class TelemetryService:
    """The API's behavior, independent of HTTP so it can be exercised directly."""

    def __init__(self, warehouse: ServingWarehouse, registry: ModelRegistry, warehouse_origin: str) -> None:
        self.warehouse = warehouse
        self.registry = registry
        self.scorer = UserScorer(registry)
        self.warehouse_origin = warehouse_origin
        # spaCy, the sentence encoder, and FAISS are not documented as thread safe, so
        # extraction is serialized. At demo traffic this costs nothing measurable.
        self._extraction_lock = threading.Lock()
        self._segments_cache: Optional[Tuple[Tuple[int, str], SegmentsResponse]] = None

    # ---------------------------------------------------------------- ingestion

    def ingest(self, payload: ScreenTextPayload) -> IngestResponse:
        """Extracts the product from raw text and stores the interaction."""
        timestamp = payload.timestamp or self.warehouse.clock.now()
        with self._extraction_lock:
            stored = self.warehouse.record_event(
                user_id=payload.user_id,
                app_name=payload.app_name,
                raw_text=payload.raw_text,
                timestamp=timestamp,
            )
        product = stored.pop("product")
        stored.pop("products")
        if product is None:
            # A screen with no text still stores an event; the response reports the empty
            # extraction of the whole text.
            product = process(payload.raw_text, app_source=payload.app_name, timestamp=timestamp)
        return IngestResponse(**stored, product=product)

    # ---------------------------------------------------------------- user scoring

    def _user_features(self, user_id: str, reference: datetime) -> pl.DataFrame:
        if not self.warehouse.user_exists(user_id):
            raise HTTPException(status_code=404, detail=f"Unknown user_id {user_id!r}")
        with self.warehouse.lock:
            features = compute_user_features(self.warehouse.con, reference, user_ids=[user_id])
        if features.height == 0:
            raise HTTPException(status_code=404, detail=f"User {user_id!r} has no events before {reference}")
        return features

    def propensity(self, user_id: str) -> PropensityResponse:
        """Scores one user's 7-day purchase probability."""
        reference = self.warehouse.reference_date()
        features = self._user_features(user_id, reference)
        probability = float(self.scorer.purchase_probability(features)[0])
        history = int(features["total_events_lifetime"][0])
        return PropensityResponse(
            user_id=user_id,
            purchase_probability=round(probability, 4),
            risk_segment=risk_segment(probability),
            reference_date=reference,
            history_events=history,
            low_history=history < LOW_HISTORY_EVENT_THRESHOLD,
        )

    def profile(self, user_id: str) -> ProfileResponse:
        """Builds the full user profile from live features and the user's history."""
        reference = self.warehouse.reference_date()
        features = self._user_features(user_id, reference)
        probability = float(self.scorer.purchase_probability(features)[0])
        switching = float(self.scorer.switching_probability(features)[0])
        segment = self.scorer.segments(features)[0]
        segment["description"] = self.registry.segment_descriptions.get(segment["cluster"])
        with self.warehouse.lock:
            clv = self.scorer.lifetime_value(self.warehouse.con, user_id, reference)
        activity = self.warehouse.user_activity(user_id, reference)

        return ProfileResponse(
            user_id=user_id,
            reference_date=reference,
            purchase_probability=round(probability, 4),
            risk_segment=risk_segment(probability),
            app_switch_probability=round(switching, 4),
            likely_to_switch_app=switching >= self.registry.switching_threshold,
            segment=segment,
            lifetime_value=clv,
            low_history=activity["total_events"] < LOW_HISTORY_EVENT_THRESHOLD,
            **activity,
        )

    # ---------------------------------------------------------------- analytics

    def segments(self) -> SegmentsResponse:
        """
        Assigns every user to a segment from current features and profiles each segment.

        The result is cached until the event count changes or the reference date moves
        into a new hour, because scoring the whole population on every dashboard refresh
        would be wasted work.
        """
        reference = self.warehouse.reference_date()
        key = (self.warehouse.counts()["events"], reference.strftime("%Y-%m-%d %H"))
        if self._segments_cache is not None and self._segments_cache[0] == key:
            return self._segments_cache[1]

        with self.warehouse.lock:
            features = compute_user_features(self.warehouse.con, reference)
        assignments = self.scorer.segments(features)
        scored = features.with_columns(
            pl.Series("cluster", [a["cluster"] for a in assignments]),
            pl.Series("purchase_probability", self.scorer.purchase_probability(features)),
        )
        total = scored.height
        grouped = (
            scored.group_by("cluster")
            .agg(
                pl.len().alias("users"),
                pl.col("purchase_probability").mean().alias("mean_purchase_probability"),
                pl.col("total_purchases_30d").cast(pl.Float64).mean().alias("mean_purchases_30d"),
                pl.col("total_spend_30d").cast(pl.Float64).mean().alias("mean_spend_30d_inr"),
                pl.col("app_entropy").cast(pl.Float64).mean().alias("mean_app_entropy"),
                pl.col("cross_app_comparison_count").cast(pl.Float64).mean().alias("mean_cross_app_comparisons"),
            )
            .sort("users", descending=True)
        )
        segments: List[Dict[str, Any]] = []
        for row in grouped.iter_rows(named=True):
            cluster = int(row["cluster"])
            segments.append(
                {
                    "cluster": cluster,
                    "segment_name": self.registry.segment_names.get(cluster, "Unclassified"),
                    "description": self.registry.segment_descriptions.get(cluster),
                    "users": int(row["users"]),
                    "share": round(row["users"] / total, 4) if total else 0.0,
                    "mean_purchase_probability": round(float(row["mean_purchase_probability"]), 4),
                    "mean_purchases_30d": round(float(row["mean_purchases_30d"] or 0.0), 2),
                    "mean_spend_30d_inr": round(float(row["mean_spend_30d_inr"] or 0.0), 2),
                    "mean_app_entropy": round(float(row["mean_app_entropy"] or 0.0), 4),
                    "mean_cross_app_comparisons": round(float(row["mean_cross_app_comparisons"] or 0.0), 2),
                }
            )
        response = SegmentsResponse(reference_date=reference, total_users=total, segments=segments)
        self._segments_cache = (key, response)
        return response

    def health(self) -> HealthResponse:
        """Reports service status, warehouse size, and the loaded models."""
        return HealthResponse(
            status="ok",
            latest_event_timestamp=self.warehouse.latest_event_timestamp(),
            reference_date=self.warehouse.reference_date(),
            clock_mode=self.warehouse.clock.mode,
            warehouse=self.warehouse_origin,
            models=self.registry.describe(),
            **self.warehouse.counts(),
        )


def create_app(
    db_path: Optional[Union[str, Path]] = None,
    source_db_path: Optional[Union[str, Path]] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    clock_mode: Optional[str] = None,
    bootstrap: Optional[bool] = None,
) -> FastAPI:
    """
    Builds the application. Explicit arguments override the environment, which is how
    tests point the API at a temporary warehouse.
    """
    settings = Settings.from_env()
    if db_path is not None:
        settings.db_path = Path(db_path)
    if source_db_path is not None:
        settings.source_db_path = Path(source_db_path)
    if artifacts_dir is not None:
        settings.artifacts_dir = Path(artifacts_dir)
    if clock_mode is not None:
        settings.clock_mode = clock_mode
    if bootstrap is not None:
        settings.bootstrap = bootstrap

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        origin = "existing"
        if settings.bootstrap:
            origin = ensure_serving_warehouse(
                db_path=settings.db_path, source_db_path=settings.source_db_path
            )
        elif not settings.db_path.exists():
            raise RuntimeError(f"Serving warehouse not found at {settings.db_path} and bootstrap is off")
        warehouse = ServingWarehouse(settings.db_path, clock_mode=settings.clock_mode)
        registry = ModelRegistry(settings.artifacts_dir)
        # The first extraction loads spaCy, the sentence encoder, and the FAISS index.
        # Doing it here keeps that cost out of the first real request.
        process(WARM_UP_TEXT)
        app.state.service = TelemetryService(warehouse, registry, origin)
        try:
            yield
        finally:
            warehouse.close()

    app = FastAPI(
        title="Consumer Telemetry API",
        version=API_VERSION,
        description=(
            "Ingests raw shopping-app screen text and serves live purchase propensity, "
            "app switching, lifetime value, and segment predictions."
        ),
        lifespan=lifespan,
    )

    def service() -> TelemetryService:
        return app.state.service

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health() -> HealthResponse:
        """Service status, warehouse size, and the serving clock."""
        return service().health()

    @app.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
    def ingest_screen_text(payload: ScreenTextPayload) -> IngestResponse:
        """
        Accepts raw screen text, runs the upstream pipeline, stores the extracted event
        in DuckDB, and returns the extracted product with the stored event.
        """
        return service().ingest(payload)

    @app.get("/user/{user_id}/propensity", response_model=PropensityResponse, tags=["predictions"])
    def get_propensity(user_id: str = PathParam(..., pattern=USER_ID_PATTERN)) -> PropensityResponse:
        """Returns the probability that the user purchases in the next 7 days."""
        return service().propensity(user_id)

    @app.get("/user/{user_id}/profile", response_model=ProfileResponse, tags=["predictions"])
    def get_user_profile(user_id: str = PathParam(..., pattern=USER_ID_PATTERN)) -> ProfileResponse:
        """Returns the full user profile: CLV, segment, switching risk, top categories, app usage."""
        return service().profile(user_id)

    @app.get("/analytics/segments", response_model=SegmentsResponse, tags=["analytics"])
    def get_segments() -> SegmentsResponse:
        """Returns all segment profiles with user counts."""
        return service().segments()

    @app.get("/analytics/activity", response_model=ActivityResponse, tags=["analytics"])
    def get_activity(window_hours: float = Query(24.0, gt=0, le=24 * 90)) -> ActivityResponse:
        """Returns activity in the trailing window and the most recent purchase."""
        return ActivityResponse(**service().warehouse.activity_summary(window_hours=window_hours))

    @app.get("/events/recent", response_model=List[EventRecord], tags=["analytics"])
    def get_recent_events(limit: int = Query(20, ge=1, le=200)) -> List[Dict[str, Any]]:
        """Returns the latest events, newest first."""
        return service().warehouse.recent_events(limit=limit)

    @app.get("/users", response_model=List[UserSummary], tags=["analytics"])
    def get_users(limit: int = Query(50, ge=1, le=1000)) -> List[Dict[str, Any]]:
        """Returns users ordered by most recent activity."""
        return service().warehouse.list_users(limit=limit)

    return app


app = create_app()
