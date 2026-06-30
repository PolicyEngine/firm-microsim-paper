"""Fast unit tests for the iso-elastic optimum verifier.

Small grids only — no full-population load, no figure rendering.
"""

from __future__ import annotations

import numpy as np
import pytest

from firm_microsim.dynamic.verify_optimum import (
    check_e_limit,
    check_foc_identity,
    closed_form_ystar,
    grid_argmax,
    numerical_second_derivative,
    scalar_argmax,
)

CASES = [
    (120_000.0, 0.05, 0.20),
    (160_000.0, 0.17, 0.20),
    (80_000.0, 0.32, 0.03),
]


@pytest.mark.parametrize("n,e,tau", CASES)
def test_closed_form_matches_numerical(n, e, tau):
    """y* = n(1-tau)**e matches grid argmax and minimize_scalar."""
    y_cf = closed_form_ystar(n, e, tau)
    y_grid = grid_argmax(n, e, tau, npts=200_001)
    y_scal = scalar_argmax(n, e, tau)
    assert abs(y_grid - y_cf) / y_cf < 1e-4
    assert abs(y_scal - y_cf) / y_cf < 1e-4


@pytest.mark.parametrize("e", [0.05, 0.17, 0.32])
def test_foc_elasticity_identity(e):
    """d ln y* / d ln(1-tau) ~ e."""
    out = check_foc_identity(150_000.0, elasticities=(e,))
    assert out[e]["err"] < 1e-3


@pytest.mark.parametrize("n,e,tau", CASES)
def test_soc_strictly_concave(n, e, tau):
    """pi''(y*) < 0 at the interior optimum."""
    y_cf = closed_form_ystar(n, e, tau)
    pp = numerical_second_derivative(n, e, tau, y_cf)
    assert pp < 0.0


def test_e_limit_to_ability():
    """y* -> n as e -> 0."""
    n_grid = np.linspace(40_000.0, 200_000.0, 5)
    out = check_e_limit(n_grid, es=(0.1, 1e-3, 1e-5))
    assert out["limit_rel_gap"] < 1e-3
    # Monotone shrink of the gap toward the frictionless optimum.
    gaps = [r["max_rel_gap"] for r in out["rows"]]
    assert gaps[0] > gaps[-1]
