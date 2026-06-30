"""Fast unit tests for the explicit value-added (deductible) VAT-notch model.

These tests avoid loading the 150MB synthetic population; they exercise only the
closed forms and their equivalence to the existing ``(1 - tau0)*y`` simulator.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from firm_microsim.dynamic import deductible_model as dm
from firm_microsim.dynamic.model import (
    T_STAR,
    dominated_region_width,
    iso_profit,
    marginal_buncher_iso,
)

TAU = dm.TAU


def _numerical_argmax(n, e, delta):
    res = minimize_scalar(
        lambda y: -float(dm.deductible_profit(y, n, e, delta, TAU, registered=True)),
        bounds=(1e-6, 2.0 * n),
        method="bounded",
        options={"xatol": 1e-8},
    )
    return float(res.x)


@pytest.mark.parametrize("n", [95_000.0, 150_000.0, 220_000.0])
@pytest.mark.parametrize("delta", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("e", [0.05, 0.17, 0.32])
def test_closed_form_matches_numerical(n, e, delta):
    """y* = n*(1 - tau*(1-delta))**e matches the numerical argmax."""
    y_cf = float(dm.optimal_turnover(n, e, delta, TAU))
    y_num = _numerical_argmax(n, e, delta)
    assert abs(y_num - y_cf) / y_cf < 1e-4


@pytest.mark.parametrize("delta", [0.0, 0.2, 0.5, 0.7, 0.9])
def test_equivalence_with_tau0_form(delta):
    """Explicit (delta,tau) model == (1 - tau0)*y model with tau0 = tau*(1-delta)."""
    e, n = 0.17, 150_000.0
    tau0 = TAU * (1.0 - delta)

    # Optimum.
    y_expl = float(dm.optimal_turnover(n, e, delta, TAU))
    y_tau0 = float(n * (1.0 - tau0) ** e)
    assert abs(y_expl - y_tau0) < 1e-9

    # Dominated-region width.
    a_expl = dm.dominated_width(delta, TAU)
    a_tau0 = float(dominated_region_width(T=T_STAR, tau=tau0))
    assert abs(a_expl - a_tau0) < 1e-9

    # Registered profit at a test turnover.
    y_test = max(y_expl, T_STAR + 1.0)
    pi_expl = float(dm.deductible_profit(y_test, n, e, delta, TAU, registered=True))
    pi_tau0 = float(iso_profit(y_test, n, e, net=1.0 - tau0))
    assert abs(pi_expl - pi_tau0) < 1e-6

    # Marginal buncher (only where there is a notch).
    if dm.effective_wedge(delta, TAU) >= dm.WEDGE_FLOOR:
        nH_expl, _ = dm.marginal_buncher_deductible(e, delta, TAU)
        nH_tau0, _ = marginal_buncher_iso(e, T=T_STAR, tau=tau0)
        assert abs(nH_expl - nH_tau0) < 1e-6


def test_limit_delta_to_one():
    """delta -> 1: wedge -> 0 and y* -> n (no distortion)."""
    n, e = 150_000.0, 0.17
    for delta in (0.9, 0.99, 0.999):
        assert dm.effective_wedge(delta, TAU) == pytest.approx(TAU * (1 - delta))
    delta = dm.DELTA_MAX
    assert dm.effective_wedge(delta, TAU) < 1e-6
    assert float(dm.optimal_turnover(n, e, delta, TAU)) == pytest.approx(n, rel=1e-6)


def test_limit_delta_to_zero():
    """delta -> 0: wedge -> tau (harsh whole-turnover notch)."""
    n, e = 150_000.0, 0.17
    assert dm.effective_wedge(0.0, TAU) == pytest.approx(TAU)
    y_expl = float(dm.optimal_turnover(n, e, 0.0, TAU))
    y_whole = n * (1.0 - TAU) ** e
    assert y_expl == pytest.approx(y_whole, rel=1e-12)


@pytest.mark.parametrize("delta", [0.0, 0.25, 0.6, 0.85])
def test_dominated_width_formula(delta):
    """dominated_width == T* * tau_eff/(1 - tau_eff)."""
    te = TAU * (1.0 - delta)
    expected = T_STAR * te / (1.0 - te)
    assert dm.dominated_width(delta, TAU) == pytest.approx(expected, rel=1e-12)


def test_effective_wedge_and_notch_jump():
    """tau_eff = tau*(1-delta); notch jump = tau_eff*T*."""
    delta = 0.4
    te = dm.effective_wedge(delta, TAU)
    assert te == pytest.approx(0.20 * 0.6)
    assert dm.notch_jump(delta, TAU) == pytest.approx(te * T_STAR)


def test_implied_delta_inverts_ratio():
    """implied_delta(tau_eff) inverts the wedge and clips to [0, 1)."""
    # A mid-range ratio inverts exactly.
    assert dm.implied_delta(0.05, TAU) == pytest.approx(1.0 - 0.05 / TAU)
    # Ratio >= tau clips delta to 0; zero ratio clips delta below 1.
    assert dm.implied_delta(0.25, TAU) == pytest.approx(0.0)
    assert dm.implied_delta(0.0, TAU) == pytest.approx(dm.DELTA_MAX)


def test_no_notch_buncher_is_nan():
    """A near-zero wedge (all-deductible sector) has no marginal buncher."""
    nH, dy = dm.marginal_buncher_deductible(0.17, dm.DELTA_MAX, TAU)
    assert np.isnan(nH) and np.isnan(dy)
