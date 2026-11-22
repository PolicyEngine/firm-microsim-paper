"""Bunching estimator definitions (issue #38, review C): threshold-aligned
bins, full end bins, gross vs net E, censoring, closing-bin interpolation, the
zero-excess case, the LLAT eq. (20) normalisation, and recovery of a
constructed signal."""

from __future__ import annotations

import numpy as np
import pytest

from firm_microsim.bunching.model import (
    bin_density,
    bin_edges,
    excess_bunching_llat,
    fit_counterfactual,
    in_range_mask,
    locate_marginal_buncher_full,
)

T = 85.0
W = 15.0
# Threshold-aligned £1k bins (e, e+1] on (20, 140]: centres 20.5 ... 139.5.
CENTRES = np.arange(20.5, 140.0, 1.0)


def _smooth_density(centres: np.ndarray) -> np.ndarray:
    return 12_000.0 * np.exp(-(centres - 20.0) / 60.0)


# ---------------------------------------------------------------------------
# Binning (C1, N1)
# ---------------------------------------------------------------------------

def test_edges_are_threshold_aligned_and_cover_the_full_range() -> None:
    for t_star in (85.0, 90.0):
        edges = bin_edges(t_star)
        assert edges[0] == 20.0 and edges[-1] == 140.0
        assert len(edges) - 1 == 120
        assert t_star in edges
    centres, _ = bin_density(np.array([50.0]), np.array([1.0]), T)
    np.testing.assert_allclose(centres, CENTRES)


def test_firm_at_exactly_threshold_lands_in_lower_bin() -> None:
    centres, f = bin_density(np.array([85.0]), np.array([7.0]), T)
    assert f[centres == 84.5][0] == 7.0
    assert f.sum() == 7.0


def test_cross_threshold_relocation_changes_f_obs_and_E() -> None:
    """Moving w firms from T*+0.1 to T*-0.1 must change the histogram (the
    old centre-on-threshold bin hid it), raise E_net by exactly w (f_cf is
    unchanged because the move stays inside the excluded window) and E by
    about w, leaving total mass unchanged."""
    grid = np.arange(20.05, 140.0, 0.1)             # 10 firms per £1k bin
    base_w = np.full(grid.shape, 1_000.0)
    extra_n, w_each = 50, 100.0                      # w = 5,000
    w_total = extra_n * w_each

    def run(at: float) -> tuple[np.ndarray, dict]:
        t = np.concatenate([grid, np.full(extra_n, at)])
        wt = np.concatenate([base_w, np.full(extra_n, w_each)])
        c, f = bin_density(t, wt, T)
        cf = fit_counterfactual(c, f, T, 7, W, W)
        return f, locate_marginal_buncher_full(c, f, cf, T, W, W), cf

    f_above, r_above, cf_above = run(T + 0.1)
    f_below, r_below, cf_below = run(T - 0.1)
    assert not np.allclose(f_above, f_below)
    assert f_above.sum() == pytest.approx(f_below.sum())
    np.testing.assert_allclose(cf_above, cf_below)
    assert r_below["E_net"] - r_above["E_net"] == pytest.approx(w_total)
    assert r_below["E"] - r_above["E"] == pytest.approx(w_total, rel=0.05)


def test_uniform_data_gives_flat_f_obs_including_end_bins() -> None:
    grid = np.arange(20.05, 140.0, 0.1)              # 10 points per bin
    mask = in_range_mask(grid, T)
    assert mask.all()
    centres, f = bin_density(grid[mask], np.ones(mask.sum()), T)
    np.testing.assert_allclose(f, 10.0)
    assert f[0] == f[-1] == 10.0
    # Points on the range edges: 20 is excluded, 140 is included.
    assert not in_range_mask(np.array([20.0]), T)[0]
    assert in_range_mask(np.array([140.0]), T)[0]


# ---------------------------------------------------------------------------
# Marginal buncher (C2)
# ---------------------------------------------------------------------------

def test_gross_and_net_excess_differ_when_gap_changes_sign() -> None:
    f_cf = _smooth_density(CENTRES)
    f_obs = f_cf.copy()
    f_obs[(CENTRES > 80) & (CENTRES < 85)] += 1_000.0   # +5,000 excess
    f_obs[(CENTRES > 70) & (CENTRES < 75)] -= 400.0     # -2,000 deficit inside window
    r = locate_marginal_buncher_full(CENTRES, f_obs, f_cf, T, W, W)
    assert r["E"] == pytest.approx(5_000.0)
    assert r["E_net"] == pytest.approx(3_000.0)


def test_y_r_is_flagged_censored_when_missing_mass_is_insufficient() -> None:
    f_cf = _smooth_density(CENTRES)
    f_obs = f_cf.copy()
    f_obs[(CENTRES > 80) & (CENTRES < 85)] += 1_000.0   # excess 5,000, no missing mass above
    r = locate_marginal_buncher_full(CENTRES, f_obs, f_cf, T, W, W)
    assert r["y_R_censored"] is True
    assert r["Delta_R"] == 0.0
    assert r["y_R"] == 100.0  # search cap = T* + window_hi


def test_closing_bin_interpolates_from_lower_edge() -> None:
    """E = 150 with deficits of 100 in each of the first two bins above T*:
    the first bin (85, 86] closes 100, the second (86, 87] closes the last
    50, half of it, so y_R = 86.5 = T* + 1.5 and Delta_R = 150."""
    f_cf = np.full(CENTRES.shape, 1_000.0)
    f_obs = f_cf.copy()
    f_obs[CENTRES == 84.5] += 150.0
    f_obs[CENTRES == 85.5] -= 100.0
    f_obs[CENTRES == 86.5] -= 100.0
    r = locate_marginal_buncher_full(CENTRES, f_obs, f_cf, T, W, W)
    assert r["E"] == pytest.approx(150.0)
    assert r["y_R"] == pytest.approx(T + 1.5)
    assert r["Delta_R"] == pytest.approx(150.0)
    assert r["y_R_censored"] is False


def test_zero_excess_gives_y_r_at_threshold_and_no_missing_mass() -> None:
    f_cf = np.full(CENTRES.shape, 1_000.0)
    f_obs = f_cf.copy()
    f_obs[CENTRES == 85.5] -= 100.0      # missing mass above, but nothing to balance
    r = locate_marginal_buncher_full(CENTRES, f_obs, f_cf, T, W, W)
    assert r["E"] == 0.0
    assert r["y_R"] == T
    assert r["Delta_R"] == 0.0
    assert r["dyR"] == 0.0
    assert r["y_R_censored"] is False


# ---------------------------------------------------------------------------
# b_llat, LLAT (2021) eq. 20 (N2)
# ---------------------------------------------------------------------------

def test_b_llat_ignores_positive_noise_above_threshold() -> None:
    f_cf = np.full(CENTRES.shape, 10_000.0)
    f_obs = f_cf.copy()
    f_obs[(CENTRES > T) & (CENTRES < T + W)] += 3_000.0
    assert excess_bunching_llat(CENTRES, f_obs, f_cf, T, W, W) == 0.0


def test_b_llat_is_signed_excess_below_over_mean_cf() -> None:
    f_cf = np.full(CENTRES.shape, 10_000.0)
    f_obs = f_cf.copy()
    f_obs[(CENTRES > 80) & (CENTRES < T)] += 1_000.0     # +5,000 below
    assert excess_bunching_llat(CENTRES, f_obs, f_cf, T, W, W) == pytest.approx(0.5)
    f_obs = f_cf.copy()
    f_obs[(CENTRES > 80) & (CENTRES < T)] -= 1_000.0     # -5,000 below
    assert excess_bunching_llat(CENTRES, f_obs, f_cf, T, W, W) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_constructed_relocation_is_recovered_with_uncensored_y_r() -> None:
    """Move 20,000 firms of mass from (85, 95] to (80, 85] on a flat density:
    with threshold-aligned bins the estimator recovers the injected excess to
    within sampling noise (E is gross, so it slightly exceeds the injection),
    mass conservation binds (Delta_R = E) and y_R lands near the true
    marginal buncher at 95, not censored. (On the old half-integer bins the
    slice (84.5, 85] was counted above T*, which attenuated E.)"""
    rng = np.random.default_rng(0)
    turnover = rng.uniform(20.0, 140.0, 400_000)
    weight = np.full_like(turnover, 2_000_000.0 / len(turnover))
    donor = (turnover > 85.0) & (turnover <= 95.0)
    idx = np.where(donor)[0][: int(20_000 / weight[0])]
    turnover[idx] = rng.uniform(80.0, 85.0, len(idx))
    centres, f_obs = bin_density(turnover, weight, T)
    f_cf = fit_counterfactual(centres, f_obs, T, 7, W, W)
    r = locate_marginal_buncher_full(centres, f_obs, f_cf, T, W, W)
    assert 0.95 < r["E"] / 20_000.0 < 1.1
    assert 0.95 < r["E_net"] / 20_000.0 < 1.05
    assert r["y_R_censored"] is False
    assert abs(r["y_R"] - 95.0) < 1.0
    assert r["Delta_R"] == pytest.approx(r["E"])
