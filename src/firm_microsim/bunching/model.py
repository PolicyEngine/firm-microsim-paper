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
mass-conservation geometry (``Delta_R``, ``y_R``), and bootstrap standard
errors. It deliberately reports NO elasticities: with the notch geometry these
data produce, no elasticity is identified from the density alone (see the
paper's placebo), and earlier wedge-normalised "elasticity" outputs inherited
an ad hoc ``tau/2`` normalisation from a deleted smoothed-schedule model.

Binning convention
------------------
Bins are £1k wide with edges aligned to the threshold ``t_star`` and closed on
the right, ``(e, e + bw]``, so ``t_star`` is a bin edge and a firm at exactly
``t_star`` (not required to register) falls in the bin below it. The
estimation range is ``(RANGE_LO, RANGE_HI]`` taken from the same edges
(:func:`bin_edges`); every caller filters firms with :func:`in_range_mask`, so
both end bins are full. The excluded window and the search cap are defined on
bin centres: below ``(t_star - W, t_star)``, above ``(t_star, t_star + W)``.

Normalisation note (literature comparability)
---------------------------------------------
The headline bunching ratio ``b`` reported here is a *width-style* ratio: excess
mass in the below-threshold window expressed relative to the counterfactual mass
**in that same window** (``b = (q_N_obs - q_N_cf) / q_N_cf``).

``b_llat`` (:func:`excess_bunching_llat`) follows equation (20) of Liu,
Lockwood, Almunia & Tam (2021): the SIGNED excess of observed over
counterfactual bin counts summed over the excluded bins below the threshold,
divided by the mean counterfactual bin count over the excluded window. It
differs from theirs in one respect: the upper edge of the excluded range is the
symmetric window ``t_star + W`` here, whereas LLAT estimate it. Because the
synthetic population is calibrated to coarse administrative bands, ``b_llat``
is not a replication of LLAT's 1.361 and should not be read as one.
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

N_BOOT = 200            # bootstrap replications
RNG_SEED = 20240617


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def bin_edges(
    t_star: float,
    lo: float = RANGE_LO,
    hi: float = RANGE_HI,
    bin_width: float = BIN_WIDTH,
) -> np.ndarray:
    """Threshold-aligned bin edges covering ``(lo, hi]``.

    Edges are ``t_star + k * bin_width`` for every integer ``k`` with the edge
    inside ``[lo, hi]``, so ``t_star`` is always an edge. Bins are ``(e, e+bw]``.
    This is the single source of the estimation range: filter firms with
    :func:`in_range_mask` so the end bins are never partially populated.
    """
    eps = 1e-9
    k_lo = int(np.ceil((lo - t_star) / bin_width - eps))
    k_hi = int(np.floor((hi - t_star) / bin_width + eps))
    return t_star + bin_width * np.arange(k_lo, k_hi + 1, dtype=float)


def bin_index(turnover: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Index of the right-closed bin ``(edges[i], edges[i+1]]`` holding each value.

    Values outside ``(edges[0], edges[-1]]`` get ``-1`` or ``len(edges) - 1``;
    callers mask them out with :func:`in_range_mask`.
    """
    return np.searchsorted(edges, np.asarray(turnover, dtype=float), side="left") - 1


def in_range_mask(
    turnover: np.ndarray,
    t_star: float,
    lo: float = RANGE_LO,
    hi: float = RANGE_HI,
    bin_width: float = BIN_WIDTH,
) -> np.ndarray:
    """Boolean mask of firms inside the estimation range ``(edges[0], edges[-1]]``."""
    edges = bin_edges(t_star, lo, hi, bin_width)
    t = np.asarray(turnover, dtype=float)
    return (t > edges[0]) & (t <= edges[-1])


def bin_density(
    turnover: np.ndarray,
    weight: np.ndarray,
    t_star: float,
    lo: float = RANGE_LO,
    hi: float = RANGE_HI,
    bin_width: float = BIN_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted threshold-aligned histogram of turnover -> ``(centres, density)``.

    Bins are ``(e, e + bin_width]`` on :func:`bin_edges`, so a firm at exactly
    ``t_star`` is counted in the bin below the threshold. Firms outside
    ``(edges[0], edges[-1]]`` are dropped. ``density`` is counts per unit of
    turnover (counts / ``bin_width``), so ``sum(density) * bin_width``
    recovers total weighted mass in range.
    """
    edges = bin_edges(t_star, lo, hi, bin_width)
    centres = (edges[:-1] + edges[1:]) / 2.0
    t = np.asarray(turnover, dtype=float)
    w = np.asarray(weight, dtype=float)
    idx = bin_index(t, edges)
    keep = (idx >= 0) & (idx < len(centres))
    counts = np.bincount(idx[keep], weights=w[keep], minlength=len(centres))
    return centres, counts.astype(float) / bin_width


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
    the excluded (manipulation) region, the bins whose centres lie in
    ``(t_star - window_lo, t_star + window_hi)``,
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

    excl = (centres > t_star - window_lo) & (centres < t_star + window_hi)
    above_excl = centres >= t_star + window_hi
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
    turnover at which cumulative missing mass above first equals ``E``, with
    linear interpolation from the closing bin's lower edge.

    ``E`` is the GROSS positive-part excess below the threshold. The search
    for ``y_R`` stops at ``t_star + window_hi``: if cumulative missing mass
    never reaches ``E`` inside the window, ``y_R`` is CENSORED at
    ``t_star + window_hi`` and ``Delta_R < E``. If ``E <= 0`` there is nothing
    to balance: ``y_R = t_star`` and ``Delta_R = 0``, uncensored. Use
    :func:`locate_marginal_buncher_full` for the censoring flag and the signed
    net excess.

    Returns ``(E, Delta_R, y_R, dyR)`` where ``dyR = y_R - t_star`` is the
    excess turnover span of the marginal buncher (Method B input).
    """
    r = locate_marginal_buncher_full(
        centres, f_obs, f_cf, t_star, window_lo, window_hi, bin_width
    )
    return r["E"], r["Delta_R"], r["y_R"], r["dyR"]


def locate_marginal_buncher_full(
    centres: np.ndarray,
    f_obs: np.ndarray,
    f_cf: np.ndarray,
    t_star: float,
    window_lo: float = DEFAULT_WINDOW,
    window_hi: float = DEFAULT_WINDOW,
    bin_width: float = BIN_WIDTH,
) -> dict:
    """Marginal-buncher search returning every diagnostic (issue #38).

    Keys: ``E`` (gross positive-part excess below ``t_star`` over the bins with
    centres in ``(t_star - window_lo, t_star)``), ``E_net`` (signed excess over
    the same bins), ``Delta_R`` (cumulative missing mass above ``t_star`` up to
    ``y_R``, equal to ``min(cumulative missing mass, E)``), ``y_R``, ``dyR``,
    and ``y_R_censored`` (True when the missing mass in the bins with centres
    in ``(t_star, t_star + window_hi)`` is insufficient to balance ``E``, so
    ``y_R`` is set to ``t_star + window_hi`` and mass conservation does NOT
    bind).

    In the closing bin ``j`` (the first where cumulative missing mass reaches
    ``E``), ``y_R = lower_edge_j + (E - cum_before) / deficit_j * bin_width``.
    With ``E <= 0`` the search is skipped: ``y_R = t_star``, ``Delta_R = 0``,
    not censored.
    """
    below = (centres > t_star - window_lo) & (centres < t_star)
    gap_below = f_obs[below] - f_cf[below]
    E = float(np.sum(np.maximum(gap_below, 0.0)) * bin_width)
    E_net = float(np.sum(gap_below) * bin_width)

    if E <= 0.0:
        return {
            "E": E,
            "E_net": E_net,
            "Delta_R": 0.0,
            "y_R": float(t_star),
            "dyR": 0.0,
            "y_R_censored": False,
        }

    search = np.where((centres > t_star) & (centres < t_star + window_hi))[0]
    cum_missing = 0.0
    y_R = float(t_star + window_hi)
    censored = True
    for j in search:
        deficit = max(f_cf[j] - f_obs[j], 0.0) * bin_width
        if deficit > 0.0 and cum_missing + deficit >= E:
            lower_edge = centres[j] - bin_width / 2.0
            y_R = float(lower_edge + (E - cum_missing) / deficit * bin_width)
            cum_missing += deficit
            censored = False
            break
        cum_missing += deficit
    return {
        "E": E,
        "E_net": E_net,
        "Delta_R": float(min(cum_missing, E)),
        "y_R": float(y_R),
        "dyR": float(max(y_R - t_star, 0.0)),
        "y_R_censored": bool(censored),
    }


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

    ``b = (q_N_obs - q_N_cf) / q_N_cf`` over the below-threshold bins (centres
    in ``(t_star - window_lo, t_star)``); the above-threshold window covers the
    bins with centres in ``(t_star, y_R]``.

    Returns ``(b, q_N_obs, q_R_obs, q_N_cf, q_R_cf)``.
    """
    mask_N = (centres > t_star - window_lo) & (centres < t_star)
    mask_R = (centres > t_star) & (centres <= y_R)

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
    window_lo: float = DEFAULT_WINDOW,
    window_hi: float = DEFAULT_WINDOW,
    bin_width: float = BIN_WIDTH,
) -> float:
    """Normalised excess bunching following LLAT (2021), equation (20).

    Liu, Lockwood, Almunia & Tam (2021, eq. 20) define
    ``b = sum_{j below the notch} (c_j - c_hat_j) / (mean c_hat over the
    excluded range)``, with ``c`` and ``c_hat`` bin counts. Here:

    * numerator: the SIGNED excess ``E_net`` of observed over counterfactual
      counts summed over the excluded bins BELOW the threshold (centres in
      ``(t_star - window_lo, t_star)``); it can be negative;
    * denominator: the mean counterfactual bin count over the whole excluded
      window (centres in ``(t_star - window_lo, t_star + window_hi)``).

    The one difference from LLAT is the upper edge of the excluded range: they
    estimate it, while this uses the symmetric window ``t_star + window_hi``.
    The statistic is not a replication of LLAT's 1.361 on administrative data.
    """
    below = (centres > t_star - window_lo) & (centres < t_star)
    excl = (centres > t_star - window_lo) & (centres < t_star + window_hi)
    if not np.any(excl):
        return np.nan
    e_net = float(np.sum(f_obs[below] - f_cf[below]) * bin_width)
    mean_cf_count = float(np.mean(f_cf[excl]) * bin_width)
    if mean_cf_count <= 0:
        return np.nan
    return e_net / mean_cf_count


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
) -> dict:
    """Run the full mass-conserving estimator once and return all statistics.

    Reports density geometry only (``b``, ``b_llat``, ``E``, ``Delta_R``,
    ``y_R``, ``dyR``); no elasticity is identified from these data.
    """
    centres, f_obs = bin_density(turnover, weight, t_star)
    f_cf = fit_counterfactual(centres, f_obs, t_star, degree, window_lo, window_hi)
    mb = locate_marginal_buncher_full(
        centres, f_obs, f_cf, t_star, window_lo, window_hi
    )
    E, Delta_R, y_R, dyR = mb["E"], mb["Delta_R"], mb["y_R"], mb["dyR"]
    b, q_N_obs, q_R_obs, q_N_cf, q_R_cf = bunching_stats(
        centres, f_obs, f_cf, t_star, window_lo, y_R
    )
    b_llat = excess_bunching_llat(
        centres, f_obs, f_cf, t_star, window_lo, window_hi
    )
    return {
        "b": b,
        "b_llat": b_llat,
        "E": E,
        "E_net": mb["E_net"],
        "Delta_R": Delta_R,
        "y_R": y_R,
        "y_R_censored": mb["y_R_censored"],
        "dyR": dyR,
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

    def __init__(self, vintage: str = "2023-24", *, scope_only: bool = False) -> None:
        """Load the synthetic population for ``vintage``.

        The threshold ``t_star`` is taken from
        ``firm_microsim.config.VINTAGES[vintage]["threshold"]`` -- never
        hardcoded. The population is filtered to the estimation range
        ``(RANGE_LO, RANGE_HI]`` on threshold-aligned edges
        (:func:`in_range_mask`).
        With ``scope_only`` the density is that of in-scope VAT firms
        (``vat_scope``); by default it is the whole ONS-frame density, the
        universe on which the near-threshold shape targets are applied.
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
        cols = ["annual_turnover_k", "weight"] + (["vat_scope"] if scope_only else [])
        firms = pd.read_csv(path, usecols=cols)
        if scope_only:
            firms = firms[firms["vat_scope"].astype(bool)]
        self.scope_only = scope_only
        firms = firms[
            in_range_mask(firms["annual_turnover_k"].to_numpy(), self.t_star)
        ].reset_index(drop=True)
        self.firms = firms

    # -- public results ----------------------------------------------------
    def estimate(
        self,
        *,
        degree: int = DEFAULT_DEGREE,
        window_lo: float = DEFAULT_WINDOW,
        window_hi: float = DEFAULT_WINDOW,
    ) -> dict:
        """Point estimates over the full weighted sample.

        Returns a dict with ``b``, ``b_llat``, ``E``, ``Delta_R``, ``y_R``,
        ``dyR``, and the binned ``centres``, ``f_obs``, ``f_cf`` arrays.
        """
        return _run_estimator(
            self.firms["annual_turnover_k"].to_numpy(),
            self.firms["weight"].to_numpy(),
            self.t_star,
            degree=degree,
            window_lo=window_lo,
            window_hi=window_hi,
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
        # Each replicate carries uniform weights summing to the ORIGINAL
        # weighted mass W (draws are already probability-proportional-to-w);
        # unit weights would rescale mass statistics such as E by n/W.
        rep_w = np.full(n, w.sum() / n)

        keys = ["b", "b_llat", "E", "dyR"]
        rows = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True, p=p)
            res = _run_estimator(
                turnover[idx], rep_w, self.t_star, **kw
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
    ) -> dict[str, pd.DataFrame]:
        """Point-estimate sensitivity grid.

        Returns ``{"degree_window": df}``: the counterfactual polynomial degree
        swept against the (symmetric) exclusion window.
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
                        "E_net": r["E_net"],
                        "y_R": r["y_R"],
                        "censored": r["y_R_censored"],
                    }
                )

        return {"degree_window": pd.DataFrame(dw_rows)}
