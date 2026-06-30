"""Fast unit tests for the value-added VAT-notch model A (inputs as a real cost).

These tests avoid loading the 150MB synthetic population; they exercise only the
closed forms of model A and its relation to model B
(:mod:`firm_microsim.dynamic.deductible_model`).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from firm_microsim.dynamic import value_added_model as va
from firm_microsim.dynamic.model import T_STAR

TAU = va.TAU


def _numerical_argmax(n, e, delta):
    res = minimize_scalar(
        lambda y: -float(va.va_profit(y, n, e, delta, TAU, registered=True)),
        bounds=(1e-6, 2.0 * n),
        method="bounded",
        options={"xatol": 1e-8},
    )
    return float(res.x)


@pytest.mark.parametrize("n", [95_000.0, 150_000.0, 220_000.0])
@pytest.mark.parametrize("delta", [0.0, 0.3, 0.7])
@pytest.mark.parametrize("e", [0.05, 0.17, 0.32])
def test_closed_form_matches_numerical(n, e, delta):
    """y*_A = n*[(1-delta)(1-tau)]**e matches the numerical argmax."""
    y_cf = float(va.optimal_turnover_A(n, e, delta, TAU))
    y_num = _numerical_argmax(n, e, delta)
    assert abs(y_num - y_cf) / y_cf < 1e-4


@pytest.mark.parametrize("delta", [0.0, 0.3, 0.5, 0.7, 0.9])
def test_dominated_width_delta_independent(delta):
    """a_A == T* * tau/(1-tau) == £21,250 for every delta (delta-independence)."""
    expected = T_STAR * TAU / (1.0 - TAU)
    assert expected == pytest.approx(21_250.0)
    assert va.dominated_width_A(delta, TAU) == pytest.approx(21_250.0, abs=1e-9)


@pytest.mark.parametrize("delta", [0.0, 0.2, 0.5, 0.7, 0.9])
def test_wedge_A_and_gap_to_B(delta):
    """w_A == (1-delta)(1-tau) and w_B - w_A == delta exactly."""
    assert va.wedge_A(delta, TAU) == pytest.approx((1.0 - delta) * (1.0 - TAU))
    # w_B built from deductible_model.effective_wedge.
    assert va.wedge_B(delta, TAU) - va.wedge_A(delta, TAU) == pytest.approx(delta)


def test_A_and_B_coincide_at_delta_zero():
    """A and B optima coincide at delta=0 and diverge for delta>0."""
    n, e = 150_000.0, 0.17
    assert va.optimal_turnover_A(n, e, 0.0, TAU) == pytest.approx(
        va.optimal_turnover_B(n, e, 0.0, TAU), rel=1e-12
    )
    for delta in (0.3, 0.6, 0.9):
        yA = float(va.optimal_turnover_A(n, e, delta, TAU))
        yB = float(va.optimal_turnover_B(n, e, delta, TAU))
        assert yA < yB  # A's smaller wedge -> smaller registered optimum
        assert abs(yA - yB) > 1.0


@pytest.mark.parametrize("delta", [0.0, 0.25, 0.6, 0.85])
def test_notch_jump_A_equals_B(delta):
    """notch_jump_A == notch_jump_B == tau*(1-delta)*T*."""
    expected = TAU * (1.0 - delta) * T_STAR
    assert va.notch_jump_A(delta, TAU) == pytest.approx(expected)
    assert va.notch_jump_A(delta, TAU) == pytest.approx(va.notch_jump_B(delta, TAU))


def test_recover_ability_A_branches():
    """Ability recovery uses the unregistered branch below T*, registered above."""
    e, delta = 0.17, 0.4
    # Unregistered: n = y_obs/(1-delta)**e.
    y_lo = 60_000.0
    assert va.recover_ability_A(y_lo, e, delta, TAU) == pytest.approx(
        y_lo / (1.0 - delta) ** e
    )
    # Registered: n = y_obs/[(1-delta)(1-tau)]**e.
    y_hi = 120_000.0
    assert va.recover_ability_A(y_hi, e, delta, TAU) == pytest.approx(
        y_hi / ((1.0 - delta) * (1.0 - TAU)) ** e
    )


def test_no_notch_buncher_is_nan():
    """A near-zero effective notch (all-deductible sector) has no marginal buncher."""
    nH, dy = va.marginal_buncher_A(0.17, va.DELTA_MAX, TAU)
    assert np.isnan(nH) and np.isnan(dy)
