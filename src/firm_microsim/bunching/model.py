"""Reduced-form bunching estimator at the VAT registration threshold.

Ports the authoritative, mass-conserving £85k bunching estimator
(``analysis/bunching_inference.py`` in the uk-vatlab repo) into a clean,
package-style API that mirrors :class:`static.model.StaticVATModel`.

The estimator implements the Kleven--Waseem / Chetty bunching design with the
mass-conservation constraint imposed:

  * the counterfactual density is an iterated polynomial in ``(y - t*)`` (with a
    scaled Vandermonde basis for high-degree numerical stability) that is forced
    to integrate to the observed total mass over the estimation range, the
    excess mass below the threshold being reabsorbed into the above-threshold
    counterfactual; and
  * the upper edge of the manipulation window -- the *marginal buncher* ``y_R``
    -- is located endogenously by the mass-conservation condition ``E = Delta_R``
    (cumulative excess mass below the threshold equals cumulative missing mass
    above), rather than fixed at an arbitrary window.

From these objects it computes the bunching ratio ``b``, excess mass ``E``, the
CES/logit substitution elasticity ``sigma``, the local (notch-width) turnover
elasticity, and bootstrap standard errors.

Normalisation note (literature comparability)
---------------------------------------------
The headline bunching ratio ``b`` reported here is a *width-style* ratio: excess
mass in the below-threshold window expressed relative to the counterfactual mass
**in that same window** (``b = (q_N_obs - q_N_cf) / q_N_cf``). This is convenient
internally but is NOT the normalisation used by Liu, Lockwood, Almunia & Tam
(2021, "VAT Notches, Voluntary Registration, and Bunching"), who report excess
bunching ``b_LLAT = 1.361`` defined as the total excess mass just below the
threshold divided by the **average counterfactual density** (the mean height of
``f_cf``) over the excluded range. To compare like-for-like with their 1.361,
:func:`excess_bunching_llat` (also returned as ``"b_llat"`` from
:meth:`BunchingEstimator.estimate`) divides the excess mass by the average
counterfactual height rather than by the integrated counterfactual mass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm_microsim.config import SYNTHETIC_DATA_DIR, VINTAGES

# ---------------------------------------------------------------------------
# Constants (ported from analysis/bunching_inference.py)
# ---------------------------------------------------------------------------

TAU_MAX = 0.20          # statutory VAT rate
BIN_WIDTH = 1.0         # £1,000 histogram bins
RANGE_LO = 20.0         # estimation range lower edge (£1,000)
RANGE_HI = 140.0        # estimation range upper edge (£1,000)

DEFAULT_DEGREE = 7      # counterfactual polynomial degree
DEFAULT_WINDOW = 15.0   # excluded window half-width either side of t* (£1,000)
TAU_E = 0.05            # effective wedge for the CES substitution elasticity

N_BOOT = 200            # bootstrap replications
RNG_SEED = 20240617


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_density(
    turnover: np.ndarray,
    weight: np.ndarray,
    lo: float = RANGE_LO,
    hi: float = RANGE_HI,
    bin_width: float = BIN_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted £1k histogram of turnover -> ``(centres, density)``.

    ``density`` is counts per unit of turnover (counts / ``bin_width``), so the
    integral ``sum(density) * bin_width`` recovers total weighted mass.
    """
    edges = np.arange(lo - 0.5, hi + 0.5 + bin_width, bin_width)
    centres = (edges[:-1] + edges[1:]) / 2.0
    counts, _ = np.histogram(turnover, bins=edges, weights=weight)
    return centres, counts / bin_width


# ---------------------------------------------------------------------------
# Mass-conserving counterfactual
# ---------------------------------------------------------------------------

def fit_counterfactual(
    centres: np.ndarray,
    f_obs: np.ndarray,
    t_star: float,
    degree: int = DEFAULT_DEGREE,
    window_lo: float = DEFAULT_WINDOW,
    window_hi: float = DEFAULT_WINDOW,
    bin_width: float = BIN_WIDTH,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Mass-conserving Kleven--Waseem / Chetty counterfactual density.

    Fit a polynomial in ``(y - t_star)`` to the observed density on bins OUTSIDE
    the excluded (manipulation) region ``[t_star - window_lo, t_star + window_hi]``,
    then iterate: scale the bins above the excluded region up by an absorption
    factor so the fitted counterfactual integrates to the same total mass as the
    observed density over the estimation range (the excess bunching mass is
    reabsorbed into the counterfactual above the threshold). Re-fit until the
    integration constraint converges.

    The regressor ``(y - t_star)`` is rescaled to ``[-1, 1]`` so the Vandermonde
    basis stays well conditioned at high degree.

    Returns ``f_cf`` (same length as ``centres``), non-negative and
    mass-conserving.
    """
    y_raw = centres - t_star
    scale = max(np.abs(y_raw).max(), 1.0)
    y = y_raw / scale

    excl = (centres >= t_star - window_lo) & (centres <= t_star + window_hi)
    above_excl = centres > t_star + window_hi
    inc = ~excl

    total_obs = np.sum(f_obs) * bin_width
    xfull = np.vander(y, N=degree + 1, increasing=True)

    adj = 1.0
    f_cf = np.zeros_like(f_obs)
    for _ in range(max_iter):
        f_target = f_obs.copy()
        f_target[above_excl] = f_obs[above_excl] * adj

        coef, *_ = np.linalg.lstsq(xfull[inc], f_target[inc], rcond=None)
        f_cf = np.maximum(xfull @ coef, 0.0)

        total_cf = np.sum(f_cf) * bin_width
        if total_cf <= 0:
            break
        ratio = total_obs / total_cf
        if abs(ratio - 1.0) < tol:
            break
        adj *= ratio

    return f_cf


# ---------------------------------------------------------------------------
# Marginal buncher (endogenous, by mass conservation E = Delta_R)
# ---------------------------------------------------------------------------

def locate_marginal_buncher(
    centres: np.ndarray,
    f_obs: np.ndarray,
    f_cf: np.ndarray,
    t_star: float,
    window_lo: float = DEFAULT_WINDOW,
    window_hi: float = DEFAULT_WINDOW,
    bin_width: float = BIN_WIDTH,
) -> tuple[float, float, float, float]:
    """Locate the marginal buncher endogenously via ``E = Delta_R``.

    Integrate cumulative excess mass below ``t_star`` (``E``) and walk up from
    ``t_star`` accumulating missing mass; the marginal buncher ``y_R`` is the
    turnover at which cumulative missing mass above first equals ``E``
    (with linear interpolation inside the closing bin).

    Returns ``(E, Delta_R, y_R, dyR)`` where ``dyR = y_R - t_star`` is the
    excess turnover span of the marginal buncher (Method B input).
    """
    below = (centres >= t_star - window_lo) & (centres < t_star)
    excess_below = np.maximum(f_obs[below] - f_cf[below], 0.0)
    E = float(np.sum(excess_below) * bin_width)

    above_idx = np.where(centres >= t_star)[0]
    cum_missing = 0.0
    y_R = t_star
    for j in above_idx:
        if centres[j] > t_star + window_hi:
            break
        deficit = max(f_cf[j] - f_obs[j], 0.0) * bin_width
        cum_missing += deficit
        y_R = centres[j]
        if cum_missing >= E:
            over = cum_missing - E
            if deficit > 0:
                y_R = centres[j] - over / deficit * bin_width
            break
    Delta_R = float(cum_missing)
    dyR = max(y_R - t_star, 0.0)
    return E, Delta_R, float(y_R), float(dyR)


# ---------------------------------------------------------------------------
# Bunching statistics
# ---------------------------------------------------------------------------

def bunching_stats(
    centres: np.ndarray,
    f_obs: np.ndarray,
    f_cf: np.ndarray,
    t_star: float,
    window_lo: float,
    y_R: float,
    bin_width: float = BIN_WIDTH,
) -> tuple[float, float, float, float, float]:
    """Bunching ratio ``b`` and window masses.

    ``b = (q_N_obs - q_N_cf) / q_N_cf`` over the below-threshold window
    ``[t_star - window_lo, t_star)``; the above-threshold window runs up to the
    endogenous marginal buncher ``y_R``.

    Returns ``(b, q_N_obs, q_R_obs, q_N_cf, q_R_cf)``.
    """
    mask_N = (centres >= t_star - window_lo) & (centres < t_star)
    mask_R = (centres >= t_star) & (centres <= y_R)

    q_N_obs = float(np.sum(f_obs[mask_N]) * bin_width)
    q_R_obs = float(np.sum(f_obs[mask_R]) * bin_width)
    q_N_cf = float(np.sum(f_cf[mask_N]) * bin_width)
    q_R_cf = float(np.sum(f_cf[mask_R]) * bin_width)

    b = (q_N_obs - q_N_cf) / q_N_cf if q_N_cf > 0 else np.nan
    return b, q_N_obs, q_R_obs, q_N_cf, q_R_cf


def excess_bunching_llat(
    centres: np.ndarray,
    f_obs: np.ndarray,
    f_cf: np.ndarray,
    t_star: float,
    window_lo: float,
    y_R: float,
    bin_width: float = BIN_WIDTH,
) -> float:
    """Literature-comparable excess bunching, LLAT (2021) normalisation.

    Liu, Lockwood, Almunia & Tam (2021) report excess bunching
    ``b_LLAT = 1.361``, defined as the total excess mass just below the
    threshold divided by the **average counterfactual density** (the mean height
    of ``f_cf``) over the excluded range. This differs from the width-style
    ``b`` of :func:`bunching_stats`, which divides by the integrated
    counterfactual *mass* in the window.

    Here the excess mass is taken over the manipulation region
    ``[t_star - window_lo, y_R]`` (below-threshold excess plus the displaced mass
    just above), and the average counterfactual height is the mean of ``f_cf``
    over that same range, so the result is directly comparable to LLAT's 1.361.
    """
    region = (centres >= t_star - window_lo) & (centres <= y_R)
    excess_mass = float(np.sum(np.maximum(f_obs[region] - f_cf[region], 0.0)) * bin_width)
    avg_cf_height = float(np.mean(f_cf[region])) if np.any(region) else np.nan
    if not avg_cf_height or avg_cf_height <= 0:
        return np.nan
    return excess_mass / avg_cf_height


# ---------------------------------------------------------------------------
# Elasticities
# ---------------------------------------------------------------------------

def substitution_elasticity(
    q_N_obs: float,
    q_R_obs: float,
    q_N_cf: float,
    q_R_cf: float,
    tau_e: float = TAU_E,
) -> float:
    """CES/logit share equation: ``sigma = ln(RR) / ln(1 + tau_e)``.

    ``RR = (q_R_cf / q_N_cf) / (q_R_obs / q_N_obs)`` is the relative-risk of the
    above/below share between counterfactual and observed worlds.
    """
    if min(q_N_obs, q_R_obs, q_N_cf, q_R_cf) <= 0:
        return np.nan
    rr = (q_R_cf / q_N_cf) / (q_R_obs / q_N_obs)
    if rr <= 0:
        return np.nan
    return float(np.log(rr) / np.log(1 + tau_e))


def local_turnover_elasticity(
    centres: np.ndarray,
    f_obs: np.ndarray,
    f_cf: np.ndarray,
    t_star: float,
    window_lo: float,
    y_R: float,
) -> tuple[float, float]:
    """Local (notch-width, Method B) turnover elasticity.

    Following the notch logic of Kleven--Waseem, the excess turnover span of the
    marginal buncher identifies an elasticity relative to the effective wedge:

        eps = (dy / t_star) / tau_eff,   tau_eff = TAU_MAX / 2

    (``tau(t_star) = tau_max / 2`` holds at the threshold for any sigmoid
    steepness, so ``tau_eff`` is steepness-invariant.)

    Returns ``(eps_median, eps_marginal)``: the bunching-mass-weighted median of
    bin-level implied elasticities across the manipulation window, and the
    headline marginal-buncher elasticity built from ``dyR = y_R - t_star``.
    """
    tau_eff = TAU_MAX / 2.0
    mask = (centres >= t_star - window_lo) & (centres <= y_R)
    excess = np.maximum(f_obs[mask] - f_cf[mask], 0.0)
    dy = np.abs(centres[mask] - t_star)
    eps_bins = (dy / t_star) / tau_eff

    if excess.sum() <= 0:
        return np.nan, np.nan

    order = np.argsort(eps_bins)
    e_sorted = eps_bins[order]
    w_sorted = excess[order]
    cw = np.cumsum(w_sorted) / np.sum(w_sorted)
    eps_median = float(np.interp(0.5, cw, e_sorted))

    dyR = max(y_R - t_star, 0.0)
    eps_marginal = float((dyR / t_star) / tau_eff)
    return eps_median, eps_marginal


# ---------------------------------------------------------------------------
# Single-pass estimator
# ---------------------------------------------------------------------------

def _run_estimator(
    turnover: np.ndarray,
    weight: np.ndarray,
    t_star: float,
    degree: int = DEFAULT_DEGREE,
    window_lo: float = DEFAULT_WINDOW,
    window_hi: float = DEFAULT_WINDOW,
    tau_e: float = TAU_E,
) -> dict:
    """Run the full mass-conserving estimator once and return all statistics."""
    centres, f_obs = bin_density(turnover, weight)
    f_cf = fit_counterfactual(centres, f_obs, t_star, degree, window_lo, window_hi)
    E, Delta_R, y_R, dyR = locate_marginal_buncher(
        centres, f_obs, f_cf, t_star, window_lo, window_hi
    )
    b, q_N_obs, q_R_obs, q_N_cf, q_R_cf = bunching_stats(
        centres, f_obs, f_cf, t_star, window_lo, y_R
    )
    sigma = substitution_elasticity(q_N_obs, q_R_obs, q_N_cf, q_R_cf, tau_e)
    eps_median, eps_marginal = local_turnover_elasticity(
        centres, f_obs, f_cf, t_star, window_lo, y_R
    )
    b_llat = excess_bunching_llat(centres, f_obs, f_cf, t_star, window_lo, y_R)
    Pi = 1 - (1 + tau_e) ** (-sigma) if np.isfinite(sigma) else np.nan
    return {
        "b": b,
        "b_llat": b_llat,
        "E": E,
        "Delta_R": Delta_R,
        "y_R": y_R,
        "dyR": dyR,
        "sigma": sigma,
        "Pi": Pi,
        "eps_local_median": eps_median,
        "eps_marginal": eps_marginal,
        "centres": centres,
        "f_obs": f_obs,
        "f_cf": f_cf,
    }


# ---------------------------------------------------------------------------
# Estimator class
# ---------------------------------------------------------------------------

class BunchingEstimator:
    """Mass-conserving reduced-form bunching estimator on the synthetic firms.

    Mirrors the shape of :class:`static.model.StaticVATModel`: construct on a
    ``vintage`` (which fixes the registration threshold ``t_star``), then call
    :meth:`estimate`, :meth:`bootstrap`, :meth:`summary`, or :meth:`sensitivity`.
    """

    def __init__(self, vintage: str = "2023-24") -> None:
        """Load the synthetic population for ``vintage``.

        The threshold ``t_star`` is taken from
        ``firm_microsim.config.VINTAGES[vintage]["threshold"]`` -- never
        hardcoded. The population is filtered to ``[RANGE_LO, RANGE_HI]``.
        """
        self.vintage = vintage
        if vintage not in VINTAGES:
            choices = ", ".join(VINTAGES)
            raise ValueError(f"Unknown vintage {vintage!r}; choose from {choices}")
        self.t_star = float(VINTAGES[vintage]["threshold"])

        path = SYNTHETIC_DATA_DIR / f"synthetic_firms_{vintage}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — generate it first with:\n"
                f"  python -m firm_microsim --vintage {vintage} "
                f"--output synthetic_firms_{vintage}.csv"
            )
        firms = pd.read_csv(path, usecols=["annual_turnover_k", "weight"])
        firms = firms[
            (firms["annual_turnover_k"] >= RANGE_LO)
            & (firms["annual_turnover_k"] <= RANGE_HI)
        ].reset_index(drop=True)
        self.firms = firms

    # -- public results ----------------------------------------------------
    def estimate(
        self,
        *,
        degree: int = DEFAULT_DEGREE,
        window_lo: float = DEFAULT_WINDOW,
        window_hi: float = DEFAULT_WINDOW,
        tau_e: float = TAU_E,
    ) -> dict:
        """Point estimates over the full weighted sample.

        Returns a dict with ``b``, ``b_llat``, ``E``, ``Delta_R``, ``y_R``,
        ``dyR``, ``sigma``, ``Pi``, ``eps_local_median``, ``eps_marginal``, and
        the binned ``centres``, ``f_obs``, ``f_cf`` arrays.
        """
        return _run_estimator(
            self.firms["annual_turnover_k"].to_numpy(),
            self.firms["weight"].to_numpy(),
            self.t_star,
            degree=degree,
            window_lo=window_lo,
            window_hi=window_hi,
            tau_e=tau_e,
        )

    def bootstrap(
        self,
        n_boot: int = N_BOOT,
        seed: int = RNG_SEED,
        **kw,
    ) -> pd.DataFrame:
        """Weighted resample-and-re-estimate bootstrap.

        Draw firms with replacement with probability proportional to survey
        weight, re-bin, and re-run the estimator on each replicate. Returns a
        DataFrame of replicate estimates for the scalar parameters.
        """
        rng = np.random.default_rng(seed)
        turnover = self.firms["annual_turnover_k"].to_numpy()
        w = self.firms["weight"].to_numpy()
        n = len(turnover)
        p = w / w.sum()

        keys = ["b", "b_llat", "E", "sigma", "Pi",
                "eps_local_median", "eps_marginal", "dyR"]
        rows = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True, p=p)
            res = _run_estimator(
                turnover[idx], np.ones(n), self.t_star, **kw
            )
            rows.append({k: res[k] for k in keys})
        return pd.DataFrame(rows)

    def summary(self, n_boot: int = N_BOOT, seed: int = RNG_SEED, **kw) -> pd.DataFrame:
        """Point estimate, bootstrap SE, and 95% percentile CI per parameter."""
        point = self.estimate(**kw)
        boot = self.bootstrap(n_boot=n_boot, seed=seed, **kw)
        rows = []
        for param in boot.columns:
            s = boot[param].dropna()
            lo, hi = (np.percentile(s, [2.5, 97.5]) if len(s) else (np.nan, np.nan))
            rows.append(
                {
                    "parameter": param,
                    "point": point[param],
                    "se": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "n_boot": int(len(s)),
                }
            )
        return pd.DataFrame(rows).set_index("parameter")

    def sensitivity(
        self,
        degrees: tuple[int, ...] = (5, 6, 7, 8),
        windows: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0),
        tau_es: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10),
    ) -> dict[str, pd.DataFrame]:
        """Point-estimate sensitivity grids.

        Returns ``{"degree_window": df, "tau_e": df}``: the first sweeps the
        counterfactual polynomial degree against the exclusion window
        (symmetric), the second sweeps the effective wedge ``tau_e``.
        """
        turnover = self.firms["annual_turnover_k"].to_numpy()
        w = self.firms["weight"].to_numpy()

        dw_rows = []
        for deg in degrees:
            for win in windows:
                r = _run_estimator(
                    turnover, w, self.t_star,
                    degree=deg, window_lo=win, window_hi=win,
                )
                dw_rows.append(
                    {
                        "degree": deg,
                        "window": win,
                        "b": r["b"],
                        "b_llat": r["b_llat"],
                        "E": r["E"],
                        "sigma": r["sigma"],
                        "eps_local_median": r["eps_local_median"],
                    }
                )

        te_rows = []
        for te in tau_es:
            r = _run_estimator(turnover, w, self.t_star, tau_e=te)
            te_rows.append({"tau_e": te, "sigma": r["sigma"], "Pi": r["Pi"]})

        return {
            "degree_window": pd.DataFrame(dw_rows),
            "tau_e": pd.DataFrame(te_rows),
        }
