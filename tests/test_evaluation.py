"""
Unit tests for calibration, decile lift, and cumulative gains analysis.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.evaluation import (
    capture_at_population_fraction,
    compute_calibration_metrics,
    compute_cumulative_gains,
    compute_decile_lift,
    plot_calibration_curves,
    plot_cumulative_gains,
    plot_lift_chart,
)


def _perfectly_ranked(n: int = 100, positives: int = 40):
    """Builds a scoring where every positive outranks every negative."""
    y = np.array([1] * positives + [0] * (n - positives))
    proba = np.linspace(0.99, 0.01, n)
    return y, proba


def test_compute_decile_lift_shape_and_columns():
    """Verifies the decile table structure and that all users are accounted for."""
    y, proba = _perfectly_ranked(200, 80)
    df = compute_decile_lift(y, proba, n_bins=10)

    assert df.height == 10
    for col in (
        "decile",
        "users",
        "mean_predicted_probability",
        "actual_purchasers",
        "actual_purchase_rate",
        "lift",
        "cumulative_users_pct",
        "cumulative_capture_pct",
    ):
        assert col in df.columns

    # Every user lands in exactly one decile, and every positive is counted once.
    assert df["users"].sum() == 200
    assert df["actual_purchasers"].sum() == 80
    # Cumulative capture must reach exactly 100 percent at the final decile.
    assert df["cumulative_capture_pct"].to_list()[-1] == pytest.approx(1.0)
    assert df["cumulative_users_pct"].to_list()[-1] == pytest.approx(1.0)


def test_decile_lift_is_bounded_by_inverse_base_rate():
    """Verifies perfect ranking hits, but cannot exceed, the 1/base_rate lift ceiling."""
    y, proba = _perfectly_ranked(200, 80)
    base_rate = float(np.mean(y))
    df = compute_decile_lift(y, proba, n_bins=10)

    ceiling = 1.0 / base_rate
    assert all(lift <= ceiling + 1e-9 for lift in df["lift"].to_list())
    # With perfect separation the top decile sits exactly at the ceiling.
    assert df["lift"].to_list()[0] == pytest.approx(ceiling)
    # The bottom decile contains no purchasers at all.
    assert df["actual_purchasers"].to_list()[-1] == 0


def test_decile_lift_on_random_scores_stays_near_one():
    """Verifies an uninformative score produces lift close to the random baseline."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=4000)
    proba = rng.random(4000)

    df = compute_decile_lift(y, proba, n_bins=10)
    # No decile should separate meaningfully when the score carries no information.
    assert all(0.75 < lift < 1.25 for lift in df["lift"].to_list())


def test_cumulative_gains_monotonicity_and_endpoints():
    """Verifies the gains curve rises monotonically and terminates at full capture."""
    y, proba = _perfectly_ranked(150, 60)
    gains = compute_cumulative_gains(y, proba)

    assert gains.height == 150
    capture = gains["capture_pct"].to_list()
    population = gains["population_pct"].to_list()

    assert all(b >= a - 1e-12 for a, b in zip(capture, capture[1:]))
    assert all(b > a for a, b in zip(population, population[1:]))
    assert capture[-1] == pytest.approx(1.0)
    assert population[-1] == pytest.approx(1.0)

    # Perfect ranking captures everyone once the positive share is contacted.
    assert capture_at_population_fraction(gains, 0.4) == pytest.approx(1.0)


def test_capture_at_population_fraction_beats_random_for_good_model():
    """Verifies capture exceeds the contacted fraction when ranking carries signal."""
    y, proba = _perfectly_ranked(300, 90)
    gains = compute_cumulative_gains(y, proba)

    for fraction in (0.1, 0.2, 0.3):
        assert capture_at_population_fraction(gains, fraction) > fraction

    # A fraction below the first user still returns a defined value.
    assert capture_at_population_fraction(gains, 0.0) == 0.0


def test_compute_calibration_metrics_on_perfect_and_naive_predictions():
    """Verifies calibration metrics behave as expected at the extremes."""
    y = np.array([0, 0, 1, 1])

    perfect = compute_calibration_metrics(y, np.array([0.0, 0.0, 1.0, 1.0]))
    assert perfect["brier_score"] == pytest.approx(0.0)
    assert perfect["roc_auc"] == pytest.approx(1.0)

    naive = compute_calibration_metrics(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert naive["brier_score"] == pytest.approx(0.25)
    assert naive["mean_predicted"] == pytest.approx(0.5)
    assert naive["observed_rate"] == pytest.approx(0.5)


def test_plot_helpers_write_images(tmp_path):
    """Verifies each chart helper writes a non-trivial PNG to the requested path."""
    y, proba = _perfectly_ranked(200, 90)
    noisy = np.clip(proba + np.random.default_rng(1).normal(0, 0.05, len(proba)), 0.001, 0.999)

    calib = plot_calibration_curves(y, proba, noisy, output_path=tmp_path / "calib.png")
    lift = plot_lift_chart(compute_decile_lift(y, proba), output_path=tmp_path / "lift.png")
    gains = plot_cumulative_gains(
        compute_cumulative_gains(y, proba), output_path=tmp_path / "gains.png"
    )

    for path in (calib, lift, gains):
        assert path.exists()
        assert path.stat().st_size > 1000
