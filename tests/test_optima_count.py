"""Tests for the iso-elastic profit-function optima counter."""

from __future__ import annotations

import numpy as np

from firm_microsim.dynamic.optima_count import (
    build_schedules,
    count_optima,
    iso_cost,
    local_maxima,
    profit,
)


def test_expected_optima_counts() -> None:
    # n chosen so every regime's interior optimum is realised:
    # flat -> 1, hard notch -> 2, reduced-rate band -> 3, taper -> 2.
    res = count_optima(n=150_000.0, e=0.17)
    expected = {label: exp for label, (_s, exp) in build_schedules().items()}
    for label, exp in expected.items():
        assert res[label]["n_maxima"] == exp, (label, res[label])


def test_flat_rate_single_concave_optimum() -> None:
    # Flat single rate: unique interior maximum at the closed form n(1-tau)**e.
    e, tau, n = 0.17, 0.20, 130_000.0
    res = count_optima(n=n, e=e)["flat single rate"]
    assert res["n_maxima"] == 1
    closed = n * (1 - tau) ** e
    assert abs(res["maxima_y"][0] - closed) / closed < 1e-3


def test_iso_cost_and_profit_shapes() -> None:
    y = np.linspace(40_000, 160_000, 11)
    n = 120_000.0
    c = iso_cost(y, n, 0.17)
    assert np.all(np.diff(c) > 0)  # cost strictly increasing
    flat = build_schedules()["flat single rate"][0]
    p = profit(y, n, flat, 0.17)
    assert p.shape == y.shape


def test_local_maxima_picks_interior_peaks() -> None:
    y = np.linspace(0, 10, 1001)
    p = -((y - 4) ** 2)  # single peak at 4
    peaks = local_maxima(y, p)
    assert peaks.size == 1
    assert abs(peaks[0] - 4) < 0.05
