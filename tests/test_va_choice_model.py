"""Fast unit tests for the value-added-choice model (A2). No 150MB data load."""

from __future__ import annotations

import numpy as np
import pytest

from firm_microsim.dynamic import va_choice_model as vc
from firm_microsim.dynamic.model import T_STAR, iso_cost

TAU = vc.TAU


def _numeric_argmax_registered(n, e, tau=TAU):
    """Fine-grid argmax of A2's registered profit pi=(1-tau)z - C(z;n,e)."""
    grid = np.linspace(0.30 * n, 1.05 * n, 600_001)
    prof = (1.0 - tau) * grid - iso_cost(grid, n, e)
    return float(grid[int(np.argmax(prof))])


@pytest.mark.parametrize("n", [90_000.0, 150_000.0, 220_000.0])
@pytest.mark.parametrize("e", [0.05, 0.17, 0.32])
def test_registered_optimum_matches_numeric_argmax(n, e):
    z_cf = vc.optimal_va(n, e, TAU, registered=True)
    z_num = _numeric_argmax_registered(n, e)
    assert z_cf == pytest.approx(n * (1.0 - TAU) ** e)
    assert abs(z_cf - z_num) / z_cf < 1e-4


@pytest.mark.parametrize("n", [90_000.0, 150_000.0])
@pytest.mark.parametrize("e", [0.05, 0.17, 0.32])
def test_unregistered_optimum_is_n(n, e):
    assert float(vc.optimal_va(n, e, TAU, registered=False)) == pytest.approx(n)


@pytest.mark.parametrize("delta", [0.0, 0.3, 0.5, 0.7, 0.9])
def test_dominated_width_turnover_is_21250(delta):
    assert vc.dominated_width_turnover(delta, TAU) == pytest.approx(21_250.0)


@pytest.mark.parametrize("delta", [0.0, 0.3, 0.5, 0.7, 0.9])
def test_dominated_width_va(delta):
    assert vc.dominated_width_va(delta, TAU) == pytest.approx((1.0 - delta) * 21_250.0)


@pytest.mark.parametrize("delta", [0.0, 0.2, 0.5, 0.85])
def test_notch_jump(delta):
    assert vc.notch_jump(delta, TAU) == pytest.approx(TAU * (1.0 - delta) * T_STAR)


def test_va_threshold():
    for delta in (0.0, 0.3, 0.6):
        assert vc.va_threshold(delta) == pytest.approx((1.0 - delta) * T_STAR)


@pytest.mark.parametrize("delta", [0.01, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
def test_ability_recovery_finite(delta):
    n = vc.recover_ability(120_000.0, delta, 0.17, TAU)
    assert np.isfinite(n)
    assert n > 0.0


def test_ability_recovery_goes_to_zero():
    y_obs = 120_000.0
    n_mid = float(vc.recover_ability(y_obs, 0.5, 0.17, TAU))
    n_hi = float(vc.recover_ability(y_obs, 0.99, 0.17, TAU))
    assert n_hi < n_mid
    assert n_hi < 0.05 * n_mid  # collapses toward zero


def test_registered_recovery_uses_full_rate_on_value_added():
    y_obs, delta, e = 120_000.0, 0.4, 0.17
    n = float(vc.recover_ability(y_obs, delta, e, TAU))
    expected = (1.0 - delta) * y_obs / (1.0 - TAU) ** e
    assert n == pytest.approx(expected)


def test_unregistered_recovery_is_observed_value_added():
    y_obs, delta, e = 60_000.0, 0.4, 0.17  # below T*
    n = float(vc.recover_ability(y_obs, delta, e, TAU))
    assert n == pytest.approx((1.0 - delta) * y_obs)


@pytest.mark.parametrize("e", [0.05, 0.17, 0.32])
def test_elasticity_identity(e):
    n, tau = 150_000.0, TAU
    s0 = 1.0 - tau
    h = 1e-5
    z_p = float(vc.optimal_va(n, e, 1.0 - s0 * np.exp(h), registered=True))
    z_m = float(vc.optimal_va(n, e, 1.0 - s0 * np.exp(-h), registered=True))
    slope = (np.log(z_p) - np.log(z_m)) / (2.0 * h)
    assert slope == pytest.approx(e, abs=1e-5)
