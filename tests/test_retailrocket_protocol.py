"""Tests for the RetailRocket protocol (E10): date resolution, grid shape and cell isolation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from src.experiments import retailrocket_protocol as rr


def test_grid_has_eight_unique_cells():
    cells = rr.grid_cells()
    assert len(cells) == 8
    assert len({c["name"] for c in cells}) == 8
    assert {c["spw_mode"] for c in cells} == {"ratio", "one"}
    assert {c["patience"] for c in cells} == {50, 200}


def test_default_dates_pass_when_data_is_long_enough():
    t, te, w = rr.resolve_dates(rr.RR_TRAIN_REFS, rr.RR_TEST_REF, datetime(2015, 9, 18))
    assert w == 0 and te == rr.RR_TEST_REF and t == list(rr.RR_TRAIN_REFS)


def test_dates_shift_by_whole_weeks():
    t, te, w = rr.resolve_dates(rr.RR_TRAIN_REFS, rr.RR_TEST_REF, datetime(2015, 9, 10))
    assert w == 1 and te == "2015-08-31 00:00:00"
    assert t == ["2015-08-17 00:00:00", "2015-08-24 00:00:00"]


def test_unsatisfiable_split_raises():
    with pytest.raises(ValueError):
        rr.resolve_dates(rr.RR_TRAIN_REFS, rr.RR_TEST_REF, datetime(2000, 1, 1))


@pytest.mark.parametrize("spw", ["ratio", "one"])
def test_fit_cell_spw_modes(spw):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + rng.normal(size=300) > 1).astype(int)
    g = np.arange(300) % 100
    m = rr.fit_cell(X, y, g, spw, "average_precision", 20)
    expected = 1.0 if spw == "one" else (y == 0).sum() / (y == 1).sum()
    assert m.get_params()["scale_pos_weight"] == pytest.approx(expected)
