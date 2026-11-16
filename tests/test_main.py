"""
Endpoint tests for the FastAPI service, run in process against a private warehouse copy.
"""

from __future__ import annotations

SAMPLE_TEXT = "Samsung Galaxy S24 Ultra 256GB Titanium $1299.99 4.7★ Add to Cart"


def _a_user(client) -> str:
    return client.get("/users", params={"limit": 1}).json()[0]["user_id"]


def test_root_redirects_to_docs(api_client):
    response = api_client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_health(api_client):
    body = api_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["events"] > 0 and body["users"] > 0
    assert body["clock_mode"] == "simulated"
    assert body["warehouse"] == "existing"
    assert set(body["models"]) == {"propensity", "switching", "segmentation", "clv"}


def test_ingest_extracts_and_stores(api_client):
    before = api_client.get("/health").json()
    response = api_client.post(
        "/ingest", json={"raw_text": SAMPLE_TEXT, "user_id": "ingest_user", "app_name": "amazon_us"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["product"]["brand"] == "Samsung"
    assert body["product"]["price"] == 1299.99
    assert body["product"]["currency"] == "USD"
    assert body["stored_action"] == "cart"
    assert body["is_new_user"] is True
    assert body["session_id"] == "ingest_user_1"
    assert body["event_id"].startswith("live_")

    after = api_client.get("/health").json()
    assert after["events"] == before["events"] + 1
    assert after["live_events"] == before["live_events"] + 1
    assert api_client.get("/events/recent", params={"limit": 1}).json()[0]["event_id"] == body["event_id"]


def test_ingest_respects_explicit_timestamp(api_client):
    response = api_client.post(
        "/ingest",
        json={"raw_text": SAMPLE_TEXT, "user_id": "dated_user", "app_name": "walmart",
              "timestamp": "2026-10-27T10:00:00+05:30"},
    )
    assert response.status_code == 200
    assert response.json()["timestamp"].startswith("2026-10-27T04:30:00")


def test_ingest_validation(api_client):
    assert api_client.post("/ingest", json={"raw_text": "", "user_id": "u1"}).status_code == 422
    assert api_client.post("/ingest", json={"raw_text": "x", "user_id": "bad id"}).status_code == 422
    assert api_client.post("/ingest", json={"raw_text": "x", "user_id": "u1", "app_name": "Bad App"}).status_code == 422
    assert api_client.post("/ingest", json={"user_id": "u1"}).status_code == 422


def test_propensity(api_client):
    user_id = _a_user(api_client)
    body = api_client.get(f"/user/{user_id}/propensity").json()
    assert body["user_id"] == user_id
    assert 0.0 <= body["purchase_probability"] <= 1.0
    assert body["risk_segment"] in {"High", "Medium", "Low"}
    assert body["history_events"] > 0


def test_profile(api_client):
    user_id = _a_user(api_client)
    body = api_client.get(f"/user/{user_id}/profile").json()
    assert body["segment"]["segment_name"]
    assert 0.0 <= body["app_switch_probability"] <= 1.0
    assert isinstance(body["likely_to_switch_app"], bool)
    assert body["app_usage"] and body["top_categories"]
    assert body["total_events"] == sum(a["events"] for a in body["app_usage"])


def test_new_user_without_purchases(api_client):
    api_client.post("/ingest", json={"raw_text": SAMPLE_TEXT, "user_id": "fresh_user", "app_name": "flipkart"})
    propensity = api_client.get("/user/fresh_user/propensity").json()
    assert propensity["low_history"] is True
    assert propensity["history_events"] == 1
    profile = api_client.get("/user/fresh_user/profile").json()
    assert profile["lifetime_value"] is None
    assert profile["low_history"] is True


def test_unknown_and_invalid_users(api_client):
    for path in ("/user/nobody_here/propensity", "/user/nobody_here/profile"):
        response = api_client.get(path)
        assert response.status_code == 404
        assert "nobody_here" in response.json()["detail"]
    assert api_client.get("/user/bad%20id/propensity").status_code == 422


def test_segments(api_client):
    body = api_client.get("/analytics/segments").json()
    assert body["total_users"] == sum(s["users"] for s in body["segments"])
    assert sum(s["share"] for s in body["segments"]) == 1.0 or abs(sum(s["share"] for s in body["segments"]) - 1.0) < 1e-3
    for segment in body["segments"]:
        assert 0.0 <= segment["mean_purchase_probability"] <= 1.0

    # Cached until the data changes, recomputed afterwards.
    assert api_client.get("/analytics/segments").json() == body
    api_client.post("/ingest", json={"raw_text": SAMPLE_TEXT, "user_id": "segment_user", "app_name": "flipkart"})
    assert api_client.get("/analytics/segments").json()["total_users"] == body["total_users"] + 1


def test_activity_recent_and_users(api_client):
    activity = api_client.get("/analytics/activity", params={"window_hours": 48}).json()
    assert activity["window_hours"] == 48
    assert activity["events_in_window"] >= activity["purchases_in_window"]
    assert api_client.get("/analytics/activity", params={"window_hours": 0}).status_code == 422

    recent = api_client.get("/events/recent", params={"limit": 5}).json()
    assert len(recent) == 5
    assert api_client.get("/events/recent", params={"limit": 0}).status_code == 422

    users = api_client.get("/users", params={"limit": 3}).json()
    assert len(users) == 3 and {"user_id", "events", "last_seen"} <= set(users[0])
