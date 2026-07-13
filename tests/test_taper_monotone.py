"""Regression tests for the taper schedule (referee finding 2).

The taper is meant to eliminate the dominated region rather than move it. That
requires net revenue R(y) = y * (1 - tau_max * f(y)) to be non-decreasing across
the band, so that keeping an extra pound of turnover never lowers net revenue.
The shipped design must additionally be BAND-CONFINED: zero relief above the
band top, so the menu row stays comparable to the level and rate rows. The two
constraints are jointly satisfiable only with top = T / (1 - 2 * tau_max)
(£141,666.67 at 20%) — a 20% VAT taper cannot fit in a £20k band.
"""

from __future__ import annotations

import numpy as np

from firm_microsim.dynamic.model import (
    TAPER_TOP,
    TAPER_WIDE_TOP,
    TAU_MAX,
    T_STAR,
    schedule_taper,
    schedule_taper_average,
    schedule_taper_marginal_relief,
)


def _net_revenue(sched, y):
    y = np.asarray(y, dtype=float)
    return y * (1.0 - TAU_MAX * np.asarray(sched(y), dtype=float))


def test_wide_top_is_the_minimum_feasible_band_top():
    assert abs(TAPER_WIDE_TOP - T_STAR / (1.0 - 2.0 * TAU_MAX)) < 1e-9
    assert abs(TAPER_WIDE_TOP - 141_666.6667) < 0.01


def test_shipped_taper_net_revenue_is_monotone():
    """The shipped taper must not create a dominated interval."""
    y = np.arange(T_STAR - 5_000.0, TAPER_WIDE_TOP + 25_000.0, 25.0)
    R = _net_revenue(schedule_taper, y)
    assert np.all(np.diff(R) >= -1e-6), "net revenue must be non-decreasing"


def test_shipped_taper_is_band_confined():
    """Zero relief above the band: liability equals tau_max * y for y > top."""
    y = np.array([TAPER_WIDE_TOP + 1.0, 200_000.0, 1_000_000.0, 1e8])
    f = schedule_taper(y)
    assert np.allclose(f, 1.0), "no relief may leak above the band top"


def test_shipped_taper_marginal_rate_bounded_by_100pct():
    """dL/dy = (y - T)/(top - T) <= 1 in band; net revenue never falls."""
    y = np.arange(T_STAR, TAPER_WIDE_TOP, 10.0)
    L = TAU_MAX * schedule_taper(y) * y
    marginal = np.diff(L) / np.diff(y)
    assert np.all(marginal <= 1.0 + 1e-9)
    assert np.all(marginal >= -1e-9)


def test_shipped_taper_fraction_is_continuous_and_bounded():
    """f is 0 at the threshold and continuous (no jump) at the band top."""
    assert schedule_taper(np.array([T_STAR]))[0] == 0.0
    lo = schedule_taper(np.array([TAPER_WIDE_TOP - 1.0]))[0]
    hi = schedule_taper(np.array([TAPER_WIDE_TOP + 1.0]))[0]
    assert abs(hi - lo) < 1e-4
    y = np.arange(T_STAR, TAPER_WIDE_TOP + 50_000.0, 100.0)
    f = schedule_taper(y)
    assert np.all((f >= 0.0) & (f <= 1.0))


def test_legacy_average_taper_is_non_monotone():
    """Guards the documented reason the legacy schedule was replaced."""
    y = np.arange(T_STAR, TAPER_TOP + 1.0, 50.0)
    R = _net_revenue(schedule_taper_average, y)
    assert np.min(np.diff(R)) < 0.0, "legacy average-rate taper should dip"


def test_marginal_relief_variant_is_monotone_but_leaks_above_band():
    """Guards the documented reason the marginal-relief design is not shipped:
    monotone, but every firm above the nominal band keeps permanent relief of
    tau_max * (top + T)/2 — its incidence is economy-wide, not band-confined."""
    y = np.arange(T_STAR - 5_000.0, TAPER_TOP + 50_000.0, 50.0)
    R = _net_revenue(schedule_taper_marginal_relief, y)
    assert np.all(np.diff(R) >= -1e-6)
    y_big = np.array([1_000_000.0])
    relief = TAU_MAX * (1.0 - schedule_taper_marginal_relief(y_big)) * y_big
    assert abs(relief[0] - TAU_MAX * (TAPER_TOP + T_STAR) / 2.0) < 1e-6
