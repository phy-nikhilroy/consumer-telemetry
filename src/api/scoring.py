"""
Live scoring for the API: features computed on demand, scored by the committed models.

Features come from the same SQL the models were trained on (`compute_rfm_features` and
`compute_behavioral_features`, restricted to the users being scored), filled with the
same missing-value rules (`fill_missing_features`), and fed to the saved estimators in
the exact column order recorded at training time. Nothing here re-implements a feature,
which is what keeps serving consistent with training.

Scores are computed against a reference date supplied by the caller, normally the
serving clock, so they reflect events ingested a moment ago.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import duckdb
import joblib
import numpy as np
import polars as pl

from src.config import (
    ARTIFACTS_DIR,
    CLV_BGNBD_PATH,
    CLV_GAMMAGAMMA_PATH,
    CLV_HORIZON_DAYS,
    PROPENSITY_CALIBRATED_PATH,
    PROPENSITY_HIGH_THRESHOLD,
    PROPENSITY_LOW_THRESHOLD,
    SEGMENTATION_MODEL_PATH,
    SWITCHING_MODEL_PATH,
    rebase_path,
)
from src.features.behavioral_features import compute_behavioral_features
from src.features.feature_store import fill_missing_features
from src.features.rfm_features import compute_rfm_features
from src.models.clv import build_rfm_summary, load_purchase_transactions, predict_clv

PROPENSITY_METRICS_PATH = ARTIFACTS_DIR / "propensity_metrics.json"
SWITCHING_METRICS_PATH = ARTIFACTS_DIR / "switching_metrics.json"
SEGMENTATION_SCALER_PATH = ARTIFACTS_DIR / "segmentation_scaler.joblib"
SEGMENTATION_SUMMARY_PATH = ARTIFACTS_DIR / "segmentation_summary.json"
PROPENSITY_EVALUATION_PATH = ARTIFACTS_DIR / "propensity_evaluation_summary.json"
CLV_SUMMARY_PATH = ARTIFACTS_DIR / "clv_summary.json"


def format_reference(reference_date: datetime) -> str:
    """Formats a reference date the way the feature SQL expects it."""
    return reference_date.strftime("%Y-%m-%d %H:%M:%S.%f")


def risk_segment(probability: float) -> str:
    """Bands a purchase probability into High, Medium, or Low."""
    if probability >= PROPENSITY_HIGH_THRESHOLD:
        return "High"
    if probability >= PROPENSITY_LOW_THRESHOLD:
        return "Medium"
    return "Low"


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ModelRegistry:
    """Loads every serving artifact once, from the committed artifacts directory."""

    def __init__(self, artifacts_dir: Optional[Union[str, Path]] = None) -> None:
        def path(default: Path) -> Path:
            return rebase_path(default, artifacts_dir)

        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR

        self.propensity_model = joblib.load(path(PROPENSITY_CALIBRATED_PATH))
        self.propensity_features: List[str] = _read_json(path(PROPENSITY_METRICS_PATH))["feature_names"]

        switching_metrics = _read_json(path(SWITCHING_METRICS_PATH))
        self.switching_model = joblib.load(path(SWITCHING_MODEL_PATH))
        self.switching_features: List[str] = switching_metrics["feature_names"]
        self.switching_threshold = float(switching_metrics.get("tuned_threshold", 0.5))

        self.segmentation_model = joblib.load(path(SEGMENTATION_MODEL_PATH))
        self.segmentation_scaler = joblib.load(path(SEGMENTATION_SCALER_PATH))
        segmentation_summary = _read_json(path(SEGMENTATION_SUMMARY_PATH))
        self.segmentation_features: List[str] = segmentation_summary["feature_names"]
        self.segmentation_recency_fill: Dict[str, float] = segmentation_summary.get("recency_fill_values", {})
        self.segment_names: Dict[int, str] = {
            int(s["cluster"]): s["name"] for s in segmentation_summary["segments"]
        }
        self.segment_descriptions: Dict[int, str] = {
            int(s["cluster"]): s["description"] for s in segmentation_summary["segments"]
        }

        self.bgf = joblib.load(path(CLV_BGNBD_PATH))
        self.ggf = joblib.load(path(CLV_GAMMAGAMMA_PATH))

    def describe(self) -> Dict[str, Any]:
        """Returns a short description of the loaded models for the health endpoint."""
        return {
            "propensity": f"{type(self.propensity_model).__name__} over {len(self.propensity_features)} features",
            "switching": f"{type(self.switching_model).__name__}, threshold {self.switching_threshold:.2f}",
            "segmentation": f"KMeans with {len(self.segment_names)} segments",
            "clv": "BG/NBD and Gamma-Gamma",
        }


def compute_user_features(
    con: duckdb.DuckDBPyConnection,
    reference_date: datetime,
    user_ids: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    Builds the unified feature rows for the given users (or everyone) at the reference
    date, joined exactly as `feature_store.build_feature_matrix` joins them.
    """
    ref = format_reference(reference_date)
    rfm = compute_rfm_features(reference_date=ref, con=con, user_ids=user_ids)
    behavioral = compute_behavioral_features(reference_date=ref, con=con, user_ids=user_ids)
    return rfm.join(behavioral, on="user_id", how="inner").sort("user_id")


class UserScorer:
    """Turns feature rows into propensity, switching, segment, and CLV outputs."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def _matrix(self, features: pl.DataFrame, names: List[str]) -> np.ndarray:
        missing = [n for n in names if n not in features.columns]
        if missing:
            raise ValueError(f"Feature rows are missing model inputs: {missing}")
        return fill_missing_features(features, names).to_numpy().astype(np.float32)

    def purchase_probability(self, features: pl.DataFrame) -> np.ndarray:
        """Returns the calibrated 7-day purchase probability per row."""
        X = self._matrix(features, self.registry.propensity_features)
        return self.registry.propensity_model.predict_proba(X)[:, 1]

    def switching_probability(self, features: pl.DataFrame) -> np.ndarray:
        """Returns the probability that the next purchase lands on a non-dominant app."""
        X = self._matrix(features, self.registry.switching_features)
        return self.registry.switching_model.predict_proba(X)[:, 1]

    def segments(self, features: pl.DataFrame) -> List[Dict[str, Any]]:
        """
        Assigns each row to a KMeans segment, using the recency fill values and scaler
        persisted at training time so new users are placed consistently with training.
        """
        names = self.registry.segmentation_features
        raw = fill_missing_features(
            features, names, recency_fill=self.registry.segmentation_recency_fill
        ).to_numpy().astype(np.float64)
        clusters = self.registry.segmentation_model.predict(self.registry.segmentation_scaler.transform(raw))
        return [
            {
                "cluster": int(c),
                "segment_name": self.registry.segment_names.get(int(c), "Unclassified"),
            }
            for c in clusters
        ]

    def lifetime_value(
        self,
        con: duckdb.DuckDBPyConnection,
        user_id: str,
        reference_date: datetime,
        horizon_days: int = CLV_HORIZON_DAYS,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns the user's CLV over the horizon, or None when they have never purchased
        (BG/NBD and Gamma-Gamma describe buyers; there is nothing to extrapolate from).
        """
        transactions = load_purchase_transactions(
            con=con, user_ids=[user_id], reference_date=reference_date
        )
        if transactions.height == 0:
            return None
        summary = build_rfm_summary(transactions, observation_period_end=reference_date)
        row = predict_clv(
            self.registry.bgf, self.registry.ggf, summary, transactions, horizon_days=horizon_days
        ).to_dicts()[0]
        return {
            "horizon_days": horizon_days,
            "predicted_clv_inr": round(float(row["predicted_clv"]), 2),
            "predicted_purchases": round(float(row["predicted_purchases"]), 4),
            "predicted_avg_order_value_inr": round(float(row["predicted_avg_order_value"]), 2),
            "probability_alive": round(float(row["probability_alive"]), 4),
            "order_value_source": str(row["order_value_estimate_source"]),
            "purchase_estimate_source": str(row["purchase_estimate_source"]),
        }
