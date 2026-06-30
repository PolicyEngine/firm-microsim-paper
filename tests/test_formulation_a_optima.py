"""Fast unit tests for the formulation-A optima figure module.

No big-data load: everything is analytic on a turnover grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from firm_microsim.dynamic.formulation_a_optima import (
    DELTAS,
    E_DEFAULT,
    N_ABILITY,
    local_maxima,
    pi_a,
    schedule_optima,
)
from firm_microsim.dynamic.model import TAU_MAX, schedule_notch


def _flat_schedule(y):
    """Fully-registered flat schedule: f(y) = 1 everywhere."""
    return np.ones_like(np.asarray(y, dtype=float))


@pytest.mark.parametrize("delta", DELTAS)
def test_flat_optimum_matches_closed_form(delta):
    """Argmax of pi_A under a flat rate is n[(1-delta)(1-tau)]**e."""
    y = np.arange(20_000.0, 200_000.0, 5.0)
    pi = pi_a(y, N_ABILITY, E_DEFAULT, delta, _flat_schedule)
    y_star_grid = y[int(np.argmax(pi))]
    closed = N_ABILITY * ((1.0 - delta) * (1.0 - TAU_MAX)) ** E_DEFAULT
    assert abs(y_star_grid - closed) / closed < 1e-3


@pytest.mark.parametrize("delta", DELTAS)
def test_flat_schedule_has_single_optimum(delta):
    """A smooth flat rate yields exactly one (interior) local maximum."""
    y, _pi, maxima_idx, _g = schedule_optima(
        N_ABILITY, E_DEFAULT, delta, _flat_schedule
    )
    assert maxima_idx.size == 1


@pytest.mark.parametrize("delta", DELTAS)
def test_notch_has_two_optima(delta):
    """The hard notch adds a boundary bunching peak: two local maxima."""
    y, _pi, maxima_idx, global_idx = schedule_optima(
        N_ABILITY, E_DEFAULT, delta, schedule_notch
    )
    assert maxima_idx.size == 2
    assert global_idx in maxima_idx


def test_value_added_factor_scales_curve():
    """(1-delta) multiplies the revenue term: pi_A + cost scales linearly."""
    from firm_microsim.dynamic.model import iso_cost

    y = np.array([90_000.0, 120_000.0])
    cost = iso_cost(y, N_ABILITY, E_DEFAULT)
    rev_full = pi_a(y, N_ABILITY, E_DEFAULT, 0.0, _flat_schedule) + cost
    rev_half = pi_a(y, N_ABILITY, E_DEFAULT, 0.5, _flat_schedule) + cost
    assert np.allclose(rev_half, 0.5 * rev_full)


def test_higher_delta_pulls_registered_optimum_toward_threshold():
    """Registered optimum n[(1-delta)(1-tau)]**e decreases in delta."""
    opts = [
        N_ABILITY * ((1.0 - d) * (1.0 - TAU_MAX)) ** E_DEFAULT for d in (0.0, 0.4, 0.8)
    ]
    assert opts[0] > opts[1] > opts[2]


def test_local_maxima_simple():
    """local_maxima finds the single peak of a concave sequence."""
    pi = np.array([0.0, 1.0, 3.0, 2.0, 1.0])
    idx = local_maxima(pi)
    assert idx.tolist() == [2]
