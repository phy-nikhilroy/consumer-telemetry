"""
Streamlit dashboard for the consumer telemetry API.

Four tabs:
    Live Feed           the latest 20 events and trailing-day activity, auto-refreshing
    User Lookup         one user's propensity, switching risk, CLV, segment, and habits
    Segments Overview   segment sizes and profiles, recomputed from current features
    Model Performance   offline evaluation metrics and charts

Run locally (with the API on port 8000):
    .venv/bin/streamlit run src/dashboard/app.py

The API URL comes from the CT_API_URL environment variable, a `CT_API_URL` Streamlit
secret, or the sidebar field, in that order of precedence after the sidebar.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

# Streamlit puts the script's own folder on sys.path, not the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.dashboard.client import (
    ARTIFACTS_DIR,
    PERFORMANCE_FIGURES,
    ApiClient,
    ApiError,
    headline_metrics,
    load_model_metrics,
    resolve_api_url,
)

REFRESH_SECONDS = 5


def _secret_api_url() -> Optional[str]:
    """Reads CT_API_URL from Streamlit secrets when a secrets file exists."""
    try:
        return st.secrets.get("CT_API_URL")  # type: ignore[no-any-return]
    except Exception:
        return None


def _client(api_url: str) -> ApiClient:
    return ApiClient(api_url)


def render_live_feed(client: ApiClient) -> None:
    """Activity metrics and the latest events; reruns itself every few seconds."""

    @st.fragment(run_every=REFRESH_SECONDS)
    def feed() -> None:
        try:
            activity = client.activity()
            events = client.recent_events(limit=20)
        except ApiError as exc:
            st.error(str(exc))
            return
        cols = st.columns(4)
        cols[0].metric("Events, last 24h", f"{activity['events_in_window']:,}")
        cols[1].metric("Active users, last 24h", f"{activity['active_users_in_window']:,}")
        cols[2].metric("Purchases, last 24h", f"{activity['purchases_in_window']:,}")
        cols[3].metric("Live events ingested", f"{activity['live_events']:,}")
        purchase = activity.get("last_purchase")
        if purchase:
            st.caption(
                f"Last purchase: {purchase['user_id']} bought "
                f"\"{purchase.get('product_title') or 'unknown product'}\" on {purchase['app_name']} "
                f"at {purchase['timestamp']}"
            )
        if events:
            frame = pd.DataFrame(events)[
                ["timestamp", "user_id", "app_name", "action", "product_title", "price", "currency", "session_id"]
            ]
            st.dataframe(frame, hide_index=True, width="stretch")
        st.caption(f"Serving clock: {activity['reference_date']}. Refreshes every {REFRESH_SECONDS} seconds.")

    feed()


def render_user_lookup(client: ApiClient) -> None:
    """Profile of one user, chosen from recent users or typed in."""
    try:
        recent = [u["user_id"] for u in client.users(limit=200)]
    except ApiError as exc:
        st.error(str(exc))
        return
    left, right = st.columns([2, 1])
    picked = left.selectbox("Recently active users", recent, index=0 if recent else None)
    typed = right.text_input("Or enter a user ID")
    user_id = (typed or picked or "").strip()
    if not user_id:
        st.info("Choose or enter a user ID.")
        return

    try:
        profile = client.profile(user_id)
    except ApiError as exc:
        st.warning(str(exc))
        return

    st.subheader(f"{user_id}: {profile['segment']['segment_name']}")
    if profile.get("segment", {}).get("description"):
        st.caption(profile["segment"]["description"])
    if profile["low_history"]:
        st.warning("Fewer than 20 events: features are thin, so treat these scores as rough.")

    cols = st.columns(4)
    cols[0].metric("Purchase probability (7 days)", f"{profile['purchase_probability']:.0%}", profile["risk_segment"])
    cols[1].metric(
        "App switch probability",
        f"{profile['app_switch_probability']:.0%}",
        "likely to switch" if profile["likely_to_switch_app"] else "likely to stay",
    )
    clv = profile.get("lifetime_value")
    if clv:
        cols[2].metric("Predicted 30-day value", f"INR {clv['predicted_clv_inr']:,.0f}")
        cols[3].metric("Probability still active", f"{clv['probability_alive']:.0%}")
    else:
        cols[2].metric("Predicted 30-day value", "n/a", "no purchases yet", delta_color="off")
        cols[3].metric("Purchases so far", f"{profile['total_purchases']}")

    stats = st.columns(4)
    stats[0].metric("Events", f"{profile['total_events']:,}")
    stats[1].metric("Sessions", f"{profile['sessions']:,}")
    stats[2].metric("Purchases", f"{profile['total_purchases']:,}")
    stats[3].metric("Spend (INR)", f"{profile['total_spend_inr']:,.0f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**App usage**")
        apps = pd.DataFrame(profile["app_usage"])
        if not apps.empty:
            st.bar_chart(apps.set_index("app_name")["events"])
    with right:
        st.markdown("**Top categories**")
        st.dataframe(pd.DataFrame(profile["top_categories"]), hide_index=True, width="stretch")


def render_segments(client: ApiClient) -> None:
    """Segment sizes and profiles."""
    try:
        data = client.segments()
    except ApiError as exc:
        st.error(str(exc))
        return
    frame = pd.DataFrame(data["segments"])
    if frame.empty:
        st.info("No segments available.")
        return
    st.caption(
        f"{data['total_users']:,} users assigned from features at {data['reference_date']}. "
        "Segments were learned offline with KMeans; assignment is live."
    )
    st.bar_chart(frame.set_index("segment_name")["users"])
    display = frame.rename(
        columns={
            "segment_name": "Segment",
            "users": "Users",
            "share": "Share",
            "mean_purchase_probability": "Mean purchase probability",
            "mean_purchases_30d": "Purchases (30d)",
            "mean_spend_30d_inr": "Spend 30d (INR)",
            "mean_app_entropy": "App entropy",
            "mean_cross_app_comparisons": "Cross-app comparisons",
            "description": "What sets it apart",
        }
    ).drop(columns=["cluster"])
    st.dataframe(display, hide_index=True, width="stretch")


def render_model_performance(artifacts_dir: Path = ARTIFACTS_DIR) -> None:
    """Offline evaluation metrics and the saved evaluation charts."""
    metrics = load_model_metrics(artifacts_dir)
    rows = headline_metrics(metrics)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Measured on the held-out test period of the simulated dataset.")
    figures = [(title, artifacts_dir / name) for title, name in PERFORMANCE_FIGURES]
    figures = [(title, path) for title, path in figures if path.exists()]
    for start in range(0, len(figures), 2):
        cols = st.columns(2)
        for col, (title, path) in zip(cols, figures[start:start + 2]):
            col.markdown(f"**{title}**")
            col.image(str(path), width="stretch")


def main() -> None:
    """Renders the dashboard."""
    st.set_page_config(page_title="Consumer Telemetry", layout="wide")
    st.title("Consumer Telemetry")
    st.caption("Cross-app shopping behavior: live events, predictions, and segments.")

    default_url = resolve_api_url(_secret_api_url())
    with st.sidebar:
        api_url = st.text_input("API URL", value=default_url).strip() or default_url
        client = _client(api_url)
        try:
            health: Dict[str, Any] = client.health()
            st.success(f"API online: {health['events']:,} events, {health['users']:,} users")
            st.caption(f"Clock mode: {health['clock_mode']}. Warehouse: {health['warehouse']}.")
        except ApiError as exc:
            st.error(str(exc))
            st.caption("Start the API, or set CT_API_URL to a deployed instance.")

    live, lookup, segments, performance = st.tabs(
        ["Live Feed", "User Lookup", "Segments Overview", "Model Performance"]
    )
    with live:
        render_live_feed(client)
    with lookup:
        render_user_lookup(client)
    with segments:
        render_segments(client)
    with performance:
        render_model_performance()


main()
