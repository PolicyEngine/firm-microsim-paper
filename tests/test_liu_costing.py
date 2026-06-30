"""Fast unit tests for the parametric extensive-margin Liu costing.

These use a tiny synthetic firm population (no 150MB data load).
"""

from __future__ import annotations

import pandas as pd
import pytest

from firm_microsim.dynamic.liu_costing import (
    T_RAISED,
    gov_revenue,
    is_registered,
    reform_cost,
)
from firm_microsim.dynamic.model import TAU_MAX, T_STAR

T = TAU_MAX


def tiny_df():
    """3 firms straddling the [85k, 100k) flip band, unit weights."""
    return pd.DataFrame(
        {
            "turnover": [50_000.0, 90_000.0, 120_000.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )


# --------------------------------------------------------------------------
# gov_revenue
# --------------------------------------------------------------------------
def test_gov_revenue_registered_is_value_added():
    omega, y = 0.3, 90_000.0
    got = gov_revenue(y, True, omega, T)
    assert got == pytest.approx(T * (1 - omega) * y)


def test_gov_revenue_unregistered_is_irrecoverable_input_vat():
    omega, y = 0.3, 90_000.0
    got = gov_revenue(y, False, omega, T)
    assert got == pytest.approx(T * omega * y)


def test_gov_revenue_band_rate_applies_only_to_registered():
    omega, y, r = 0.3, 90_000.0, 0.10
    assert gov_revenue(y, True, omega, T, rate=r) == pytest.approx(r * (1 - omega) * y)
    # unregistered ignores the band rate (irrecoverable input VAT at statutory t).
    assert gov_revenue(y, False, omega, T, rate=r) == pytest.approx(T * omega * y)


# --------------------------------------------------------------------------
# is_registered
# --------------------------------------------------------------------------
def test_is_registered_mandatory_above_threshold():
    # Consumer-facing firm (no voluntary reg) above T must be registered.
    assert bool(is_registered(120_000.0, T_STAR, omega=0.2, beta=0.9))


def test_is_registered_voluntary_below_threshold_when_input_heavy():
    # omega - beta*(1-omega) = 0.7 - 0.2*0.3 = 0.64 > 0 -> voluntary.
    assert bool(is_registered(50_000.0, T_STAR, omega=0.7, beta=0.2))


def test_is_registered_not_below_threshold_when_consumer_facing():
    # omega - beta*(1-omega) = 0.2 - 0.9*0.8 = -0.52 < 0 -> not registered.
    assert not bool(is_registered(50_000.0, T_STAR, omega=0.2, beta=0.9))


# --------------------------------------------------------------------------
# reform_cost
# --------------------------------------------------------------------------
def test_reform_cost_raise100k_hand_computed():
    df = tiny_df()
    omega, beta = 0.2, 0.9  # consumer-facing: no voluntary registration.
    out = reform_cost(df, omega, beta, "raise100k")

    # Baseline (T*=85k): firm1 (50k) unregistered, firm2 (90k) registered,
    # firm3 (120k) registered.
    base = (
        T * omega * 50_000.0
        + T * (1 - omega) * 90_000.0
        + T * (1 - omega) * 120_000.0
    )
    # Reform (T=100k): firm2 (90k) flips to unregistered; firm3 stays registered.
    reform = (
        T * omega * 50_000.0
        + T * omega * 90_000.0
        + T * (1 - omega) * 120_000.0
    )
    assert out["baseline"] == pytest.approx(base)
    assert out["reform"] == pytest.approx(reform)
    assert out["cost"] == pytest.approx(reform - base)
    # The flipped firm only changes by t*(1-2*omega)*y.
    assert out["cost"] == pytest.approx(-T * (1 - 2 * omega) * 90_000.0)


def test_reform_cost_raise100k_sign_is_loss_for_consumer_facing():
    # omega < 0.5, consumer-facing: raising the threshold loses revenue.
    df = tiny_df()
    out = reform_cost(df, omega=0.2, beta=0.9, reform="raise100k")
    assert out["cost"] < 0


def test_raise100k_no_effect_when_all_voluntarily_register():
    # Input-heavy B2B (omega=0.7,beta=0.2) all voluntarily register, so the
    # threshold is irrelevant -> zero cost.
    df = tiny_df()
    out = reform_cost(df, omega=0.7, beta=0.2, reform="raise100k")
    assert out["cost"] == pytest.approx(0.0)


def test_reform_cost_unknown_reform_raises():
    with pytest.raises(ValueError):
        reform_cost(tiny_df(), 0.3, 0.5, "nonsense")


def test_t_raised_constant():
    assert T_RAISED == 100_000.0
