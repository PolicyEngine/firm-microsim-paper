"""Tests for the stylised Liu et al. (2021) VAT-notch mechanism illustration."""

from __future__ import annotations

import numpy as np

from firm_microsim.dynamic.liu_mechanism import (
    profit_registered,
    profit_unregistered,
    registration_effect,
)
from firm_microsim.dynamic.model import schedule_taper


def test_registration_effect_sign_splits_by_firm_type() -> None:
    y = 100_000.0
    # consumer-facing, low input share -> registering is a net cost (bunching)
    assert registration_effect(y, omega=0.2, beta=0.9) < 0
    # input-heavy, low B2C -> registering is a net gain (voluntary registration)
    assert registration_effect(y, omega=0.7, beta=0.2) > 0


def test_registration_effect_matches_closed_form() -> None:
    y, omega, beta, t = 90_000.0, 0.45, 0.5, 0.20
    expected = t * y * (omega - beta * (1 - omega))
    assert np.isclose(registration_effect(y, omega, beta, t=t), expected)


def test_profit_difference_equals_registration_effect_full_rate() -> None:
    # At the full output rate (schedule factor == 1) the gap between registered
    # and unregistered profit must equal the registration effect.
    y = np.linspace(60_000, 150_000, 50)
    omega, beta = 0.3, 0.6
    pr = profit_registered(y, omega, beta, lambda yy: np.ones_like(yy))
    pu = profit_unregistered(y, omega)
    assert np.allclose(pr - pu, registration_effect(y, omega, beta))


def test_taper_schedule_reduces_output_burden_in_band() -> None:
    # In the taper band the output rate is below the full rate, so a registered
    # firm's profit is at least as high as under the full rate.
    y = np.array([95_000.0])
    omega, beta = 0.2, 0.9
    full = profit_registered(y, omega, beta, lambda yy: np.ones_like(yy))
    tapered = profit_registered(y, omega, beta, schedule_taper)
    assert tapered[0] >= full[0]
