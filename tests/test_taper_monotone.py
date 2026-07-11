"""Regression tests for the taper schedule (referee finding 2).

The taper is meant to eliminate the dominated region rather than move it. That
requires net revenue R(y) = y * (1 - tau_max * f(y)) to be non-decreasing across
the band, so that keeping an extra pound of turnover never lowers net revenue.
"""

from __future__ import annotations

import numpy as np

from firm_microsim.dynamic.model import (
    TAPER_TOP,
    TAU_MAX,
    T_STAR,
    schedule_taper,
    schedule_taper_average,
)


def _net_revenue(sched, y):
    y = np.asarray(y, dtype=float)
    return y * (1.0 - TAU_MAX * np.asarray(sched(y), dtype=float))


def test_marginal_relief_taper_net_revenue_is_monotone():
    """The shipped taper must not create a dominated interval."""
    y = np.arange(T_STAR - 5_000.0, TAPER_TOP + 25_000.0, 50.0)
    R = _net_revenue(schedule_taper, y)
    assert np.all(np.diff(R) >= -1e-6), "net revenue must be non-decreasing"


def test_legacy_average_taper_is_non_monotone():
    """Guards the documented reason the legacy schedule was replaced."""
    y = np.arange(T_STAR, TAPER_TOP + 1.0, 50.0)
    R = _net_revenue(schedule_taper_average, y)
    assert np.min(np.diff(R)) < 0.0, "legacy average-rate taper should dip"


def test_taper_fraction_is_continuous_and_bounded():
    """f is 0 at the threshold and continuous (no jump) at the band top."""
    assert schedule_taper(np.array([T_STAR]))[0] == 0.0
    lo = schedule_taper(np.array([TAPER_TOP - 1.0]))[0]
    hi = schedule_taper(np.array([TAPER_TOP + 1.0]))[0]
    assert abs(hi - lo) < 1e-3
    y = np.arange(T_STAR, TAPER_TOP + 50_000.0, 100.0)
    f = schedule_taper(y)
    assert np.all((f >= 0.0) & (f <= 1.0))
