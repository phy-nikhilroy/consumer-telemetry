"""
Tests for the Streamlit dashboard and its API client.

The app itself is executed headless with Streamlit's AppTest, once with the API
unreachable (it must degrade to messages, not exceptions) and once with the client
patched to return fixed data (every tab must render it).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

import src.dashboard.client as client_module
from src.dashboard.client import ApiClient, ApiError, headline_metrics, load_model_metrics, resolve_api_url

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "dashboard" / "app.py")

PROFILE = {
    "user_id": "user_001", "reference_date": "2026-10-27T22:00:00", "total_events": 120, "total_purchases": 4,
    "total_spend_inr": 52000.0, "sessions": 40, "first_seen": "2026-09-28T08:00:00",
    "last_seen": "2026-10-27T21:00:00", "purchase_probability": 0.72, "risk_segment": "High",
    "app_switch_probability": 0.41, "likely_to_switch_app": True,
    "segment": {"cluster": 2, "segment_name": "VIP Loyalists", "description": "big spenders"},
    "lifetime_value": {"horizon_days": 30, "predicted_clv_inr": 150000.0, "predicted_purchases": 3.1,
                       "predicted_avg_order_value_inr": 48000.0, "probability_alive": 0.97,
                       "order_value_source": "gamma_gamma", "purchase_estimate_source": "bg_nbd"},
    "top_categories": [{"category_l1": "Electronics", "category_l2": "Smartphones", "events": 60}],
    "app_usage": [{"app_name": "flipkart", "events": 120, "purchases": 4, "share": 1.0}],
    "low_history": False,
}


class FakeClient:
    """Stands in for ApiClient inside the AppTest run."""

    def __init__(self, base_url=None, *args, **kwargs):
        self.base_url = base_url

    def health(self):
        return {"events": 1000, "users": 24, "clock_mode": "simulated", "warehouse": "copied"}

    def activity(self, window_hours=24.0):
        return {"events_in_window": 50, "active_users_in_window": 10, "purchases_in_window": 3, "live_events": 2,
                "reference_date": "2026-10-27T22:00:00",
                "last_purchase": {"user_id": "user_001", "product_title": "Apple iPhone 15", "app_name": "amazon_in",
                                  "timestamp": "2026-10-27T21:00:00"}}

    def recent_events(self, limit=20):
        return [{"timestamp": "2026-10-27T21:00:00", "user_id": "user_001", "app_name": "amazon_in", "action": "buy",
                 "product_title": "Apple iPhone 15", "price": 68330.0, "currency": "INR", "session_id": "user_001_9"}]

    def users(self, limit=200):
        return [{"user_id": "user_001", "events": 120, "last_seen": "2026-10-27T21:00:00"}]

    def profile(self, user_id):
        return PROFILE

    def segments(self):
        return {"reference_date": "2026-10-27T22:00:00", "total_users": 24, "segments": [
            {"cluster": 2, "segment_name": "VIP Loyalists", "description": "big spenders", "users": 24, "share": 1.0,
             "mean_purchase_probability": 0.6, "mean_purchases_30d": 9.0, "mean_spend_30d_inr": 90000.0,
             "mean_app_entropy": 1.1, "mean_cross_app_comparisons": 5.0}]}


def _mock_client(handler) -> ApiClient:
    return ApiClient("http://test", transport=httpx.MockTransport(handler))


def test_resolve_api_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CT_API_URL", raising=False)
    assert resolve_api_url() == "http://localhost:8000"
    monkeypatch.setenv("CT_API_URL", "https://example.test/")
    assert resolve_api_url() == "https://example.test"
    assert resolve_api_url("http://other:9000/") == "http://other:9000"


def test_client_maps_errors():
    client = _mock_client(lambda request: httpx.Response(404, json={"detail": "Unknown user_id 'x'"}))
    with pytest.raises(ApiError, match="Unknown user_id"):
        client.profile("x")
    client = _mock_client(lambda request: httpx.Response(500, text="broken"))
    with pytest.raises(ApiError, match="500"):
        client.health()

    def unreachable(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="Cannot reach"):
        _mock_client(unreachable).segments()


def test_client_passes_parameters():
    seen = {}

    def handler(request):
        seen[request.url.path] = dict(request.url.params)
        return httpx.Response(200, json=[])

    client = _mock_client(handler)
    client.recent_events(limit=7)
    client.users(limit=9)
    assert seen["/events/recent"] == {"limit": "7"}
    assert seen["/users"] == {"limit": "9"}


def test_headline_metrics_from_committed_artifacts():
    metrics = load_model_metrics()
    rows = headline_metrics(metrics)
    models = {row["Model"] for row in rows}
    assert {"Purchase propensity", "App switching", "Lifetime value", "Segmentation"} <= models
    assert headline_metrics({}) == []


def test_dashboard_degrades_when_api_is_down(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CT_API_URL", "http://127.0.0.1:9")
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Cannot reach the API" in e.value for e in at.error)
    assert [t.label for t in at.tabs] == ["Live Feed", "User Lookup", "Segments Overview", "Model Performance"]


def test_dashboard_renders_every_tab(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(client_module, "ApiClient", FakeClient)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    assert not at.error
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Events, last 24h"] == "50"
    assert metrics["Purchase probability (7 days)"] == "72%"
    assert metrics["Predicted 30-day value"] == "INR 150,000"
    assert [h.value for h in at.subheader] == ["user_001: VIP Loyalists"]
    assert any("API online" in s.value for s in at.success)
