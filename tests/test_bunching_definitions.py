"""Bunching estimator definitions (issue #38): gross vs net E, censoring flag,
and recovery of a constructed signal."""

from __future__ import annotations

import numpy as np

from firm_microsim.bunching.model import (
    bin_density,
    fit_counterfactual,
    locate_marginal_buncher_full,
)


def _smooth_density(centres: np.ndarray) -> np.ndarray:
    return 12_000.0 * np.exp(-(centres - 20.0) / 60.0)


def test_gross_and_net_excess_differ_when_gap_changes_sign() -> None:
    centres = np.arange(20.0, 141.0, 1.0)
    f_cf = _smooth_density(centres)
    f_obs = f_cf.copy()
    f_obs[(centres >= 80) & (centres < 85)] += 1_000.0   # +5,000 excess
    f_obs[(centres >= 70) & (centres < 75)] -= 400.0     # -2,000 deficit inside window
    r = locate_marginal_buncher_full(centres, f_obs, f_cf, 85.0, 15.0, 15.0)
    assert r["E"] == 5_000.0
    assert r["E_net"] == 3_000.0


def test_y_r_is_flagged_censored_when_missing_mass_is_insufficient() -> None:
    centres = np.arange(20.0, 141.0, 1.0)
    f_cf = _smooth_density(centres)
    f_obs = f_cf.copy()
    f_obs[(centres >= 80) & (centres < 85)] += 1_000.0   # excess 5,000, no missing mass above
    r = locate_marginal_buncher_full(centres, f_obs, f_cf, 85.0, 15.0, 15.0)
    assert r["y_R_censored"] is True
    assert r["Delta_R"] == 0.0
    assert r["y_R"] == 100.0  # search cap = T* + window_hi


def test_constructed_relocation_is_recovered_with_uncensored_y_r() -> None:
    """Move 20,000 firms of mass from (85, 95] to [80, 85): the estimator
    recovers a substantial share of the injected excess, mass conservation
    binds (Delta_R = E) and y_R lands inside the window (not censored). The
    global-area rescaling in ``fit_counterfactual`` attenuates E, which is
    why recovery is well below 100% -- the paper's recovery exercise
    documents the same attenuation."""
    rng = np.random.default_rng(0)
    turnover = rng.uniform(20.0, 140.0, 400_000)
    weight = np.full_like(turnover, 2_000_000.0 / len(turnover))
    donor = (turnover > 85.0) & (turnover <= 95.0)
    idx = np.where(donor)[0][: int(20_000 / weight[0])]
    turnover[idx] = rng.uniform(80.0, 85.0, len(idx))
    centres, f_obs = bin_density(turnover, weight)
    f_cf = fit_counterfactual(centres, f_obs, 85.0, 7, 15.0, 15.0)
    r = locate_marginal_buncher_full(centres, f_obs, f_cf, 85.0, 15.0, 15.0)
    assert 0.4 < r["E"] / 20_000.0 < 1.0
    assert r["y_R_censored"] is False
    assert 85.0 < r["y_R"] <= 100.0
    assert r["Delta_R"] >= r["E"]  # closing bin counted in full; y_R interpolated inside it
