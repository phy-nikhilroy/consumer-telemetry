"""
Thin HTTP client for the dashboard.

The dashboard reads everything through the API rather than opening the warehouse: DuckDB
allows only one read-write process per database file, and the API is that process. It
also means the dashboard can be hosted separately from the API (for example on a free
Streamlit tier) with nothing but the API URL.

This module depends only on `httpx` and the standard library, so the dashboard's
requirements stay small.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

DEFAULT_API_URL = "http://localhost:8000"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"


class ApiError(RuntimeError):
    """Raised with a readable message when the API cannot answer a request."""


def resolve_api_url(explicit: Optional[str] = None) -> str:
    """Returns the API URL from an explicit value, then CT_API_URL, then the local default."""
    url = explicit or os.environ.get("CT_API_URL") or DEFAULT_API_URL
    return url.rstrip("/")


class ApiClient:
    """Wraps the API endpoints the dashboard uses."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 20.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = resolve_api_url(base_url)
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        """Closes the underlying connection pool."""
        self._client.close()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise ApiError(f"Cannot reach the API at {self.base_url}: {exc}") from exc
        if response.status_code == 404:
            detail = response.json().get("detail", "Not found") if response.content else "Not found"
            raise ApiError(str(detail))
        if response.status_code >= 400:
            raise ApiError(f"API returned {response.status_code} for {path}: {response.text[:200]}")
        return response.json()

    def health(self) -> Dict[str, Any]:
        """Returns `/health`."""
        return self._get("/health")

    def activity(self, window_hours: float = 24.0) -> Dict[str, Any]:
        """Returns `/analytics/activity`."""
        return self._get("/analytics/activity", {"window_hours": window_hours})

    def recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns `/events/recent`."""
        return self._get("/events/recent", {"limit": limit})

    def users(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Returns `/users`."""
        return self._get("/users", {"limit": limit})

    def profile(self, user_id: str) -> Dict[str, Any]:
        """Returns `/user/{user_id}/profile`."""
        return self._get(f"/user/{user_id}/profile")

    def segments(self) -> Dict[str, Any]:
        """Returns `/analytics/segments`."""
        return self._get("/analytics/segments")


def load_model_metrics(artifacts_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Reads the committed model summaries for the performance tab.

    These are the offline evaluation results written by the model orchestrators, so the
    tab shows exactly the numbers in the README and the evaluation report. Missing files
    are skipped rather than treated as errors.
    """
    directory = artifacts_dir or ARTIFACTS_DIR
    out: Dict[str, Any] = {}
    for key, name in (
        ("propensity", "propensity_metrics.json"),
        ("evaluation", "propensity_evaluation_summary.json"),
        ("switching", "switching_metrics.json"),
        ("clv", "clv_summary.json"),
        ("segmentation", "segmentation_summary.json"),
    ):
        path = directory / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                out[key] = json.load(f)
    return out


def headline_metrics(metrics: Dict[str, Any]) -> List[Dict[str, str]]:
    """Condenses the model summaries into the rows shown in the performance table."""
    rows: List[Dict[str, str]] = []
    propensity = metrics.get("propensity", {}).get("test")
    if propensity:
        rows.append({"Model": "Purchase propensity", "Metric": "Test ROC-AUC / PR-AUC",
                     "Value": f"{propensity['roc_auc']:.3f} / {propensity['pr_auc']:.3f}"})
    evaluation = metrics.get("evaluation")
    if evaluation:
        calibrated = evaluation["test_metrics_calibrated"]
        rows.append({"Model": "Purchase propensity", "Metric": "Calibrated Brier score",
                     "Value": f"{calibrated['brier_score']:.3f} ({evaluation['calibration_method']})"})
        rows.append({"Model": "Purchase propensity", "Metric": "Buyers captured in top 10% / 20%",
                     "Value": f"{evaluation['capture_top_10_pct']:.1%} / {evaluation['capture_top_20_pct']:.1%}"})
    switching = metrics.get("switching")
    if switching:
        tuned = switching["binary_test_tuned_threshold"]
        rows.append({"Model": "App switching", "Metric": "Test ROC-AUC / F1 (tuned threshold)",
                     "Value": f"{tuned['roc_auc']:.3f} / {tuned['f1']:.3f}"})
    clv = metrics.get("clv")
    if clv:
        rows.append({"Model": "Lifetime value", "Metric": "Customers / median 30-day CLV",
                     "Value": f"{clv['n_customers']} / INR {clv['median_predicted_clv']:,.0f}"})
    segmentation = metrics.get("segmentation")
    if segmentation:
        rows.append({"Model": "Segmentation", "Metric": "Segments / silhouette",
                     "Value": f"{segmentation['selected_k']} / {segmentation['kmeans_quality']['silhouette']:.3f}"})
    return rows


PERFORMANCE_FIGURES = (
    ("Calibration curve", "propensity_calibration_curve.png"),
    ("Decile lift", "propensity_lift_chart.png"),
    ("Cumulative gains", "propensity_cumulative_gains.png"),
    ("ROC and precision-recall", "propensity_roc_pr_curves.png"),
    ("App switching drivers (SHAP)", "switching_shap_summary.png"),
    ("Segments (PCA projection)", "segmentation_pca_scatter.png"),
)
