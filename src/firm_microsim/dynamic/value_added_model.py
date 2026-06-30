"""Alternative value-added VAT-notch model (A): inputs subtracted as a REAL cost.

The paper's structural simulator (and its explicit reformulation in
:mod:`firm_microsim.dynamic.deductible_model`) writes a registered firm's profit
as a wedge on the WHOLE turnover. Letting ``tau`` be the statutory rate (0.20)
and ``delta in [0, 1)`` the sector deductible-input share of turnover (so
``d = delta*y`` is deductible input spend and ``1 - delta`` is the value-added
share), the two readings of the same iso-elastic problem are:

MODEL B  (existing paper / ``deductible_model.py``)
    The tax base is adjusted to value added but the input spend ``d`` is NOT
    separately subtracted as a real cost; ``C`` is interpreted as the firm's
    TOTAL turnover cost::

        Unregistered:  pi_B = y                       - C(y; n, e)
        Registered:    pi_B = (1 - tau*(1 - delta))*y - C(y; n, e)

MODEL A  (this module)
    VAT is levied on value added AND the deductible input spend ``d = delta*y``
    is subtracted as a real cost; ``C`` is now interpreted as the firm's
    OWN-FACTOR / NON-INPUT cost::

        Unregistered:  pi_A = (1 - delta)*y          - C(y; n, e)   (tau = 0)
        Registered:    pi_A = (1 - delta)*(1 - tau)*y - C(y; n, e)

The iso-elastic cost ``C(y; n, e) = (n/(1+1/e))*(y/n)**(1+1/e)`` is IDENTICAL in
form in both models; the SAME function carries a DIFFERENT economic meaning
(total cost in B, own-factor cost in A). The two models coincide only at
``delta = 0``; in general the registered net-of-tax retention wedges differ by
exactly ``delta``::

    w_A(delta) = (1 - delta)*(1 - tau)        (A: marginal £ retained, registered)
    w_B(delta) = 1 - tau*(1 - delta)          (B: marginal £ retained, registered)
    w_B - w_A  = delta                        (exactly)

CLOSED FORMS FOR A (derived, implemented and verified numerically below)
------------------------------------------------------------------------
* Registered optimum     y*_A  = n*[(1 - delta)*(1 - tau)]**e
* Unregistered optimum    y*_A0 = n*(1 - delta)**e            (tau = 0)
* Registered wedge        w_A   = (1 - delta)*(1 - tau)
* Ability recovery        registered   n = y_obs/[(1-delta)(1-tau)]**e
                          unregistered n = y_obs/(1-delta)**e
* Notch jump at T*        Delta_A = (1-delta)*T* - (1-delta)(1-tau)*T*
                                  = tau*(1-delta)*T*            (== B's jump)
* Dominated-region width  a_A solves (1-delta)(1-tau)(T*+a) = (1-delta)T*
                                  => a_A = T* * tau/(1 - tau) = £21,250,
                          INDEPENDENT of delta (the (1-delta) cancels). This is
                          the key result; B's width T*tau(1-delta)/(1-tau(1-delta))
                          shrinks with delta.

Run ``python -m firm_microsim.dynamic.value_added_model`` (or call :func:`main`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import brentq, minimize_scalar

from firm_microsim.config import RESULTS_DIR
from firm_microsim.dynamic import deductible_model as dm
from firm_microsim.dynamic import sector_vat
from firm_microsim.dynamic.model import (
    E_HEADLINE,
    ELASTICITIES,
    T_STAR,
    TAU_MAX,
    iso_cost,
    iso_profit,
)

# Statutory VAT rate (the rate levied on value added once registered).
TAU = TAU_MAX  # 0.20

# delta is a share in [0, 1); clip just below 1 so the value-added share is > 0.
DELTA_MAX = dm.DELTA_MAX

# Effective notch (= tau*(1-delta)) below this is treated as "no notch".
WEDGE_FLOOR = dm.WEDGE_FLOOR

# Same representative SIC divisions deductible_model uses.
REPRESENTATIVE_SECTORS = dm.REPRESENTATIVE_SECTORS

# House style (matches deductible_model / dynamic figures).
PALETTE = ["#326b77", "#122740", "#1b485e", "#568b87", "#80ae9a", "#b5d1ae"]
PRIMARY = "#326b77"
ACCENT = "#d62728"
SECONDARY = "#122740"
LABEL_SIZE = 15
TICK_SIZE = 13


# ---------------------------------------------------------------------------
# Core closed forms of MODEL A (inputs subtracted as a real cost)
# ---------------------------------------------------------------------------
def value_added_share(delta):
    """Value-added share ``1 - delta`` (the unregistered marginal retained £)."""
    return 1.0 - np.asarray(delta, dtype=float)


def wedge_A(delta, tau=TAU):
    """Registered net-of-tax retention wedge under A: ``(1 - delta)*(1 - tau)``.

    This is the marginal £ of turnover a REGISTERED firm keeps once both the
    VAT on value added and the deductible input spend are netted out.
    """
    return (1.0 - np.asarray(delta, dtype=float)) * (1.0 - tau)


def wedge_B(delta, tau=TAU):
    """Registered retention wedge under B: ``1 - tau*(1 - delta)``.

    Built on :func:`deductible_model.effective_wedge` (= ``tau*(1 - delta)``), so
    ``w_B = 1 - effective_wedge``. Satisfies ``w_B - w_A == delta`` exactly.
    """
    return 1.0 - dm.effective_wedge(delta, tau)


def va_profit(y, n, e, delta, tau=TAU, registered=None, T=T_STAR):
    """Model-A profit ``pi_A(y; n, e, delta)`` (£).

    Deductible inputs ``d = delta*y`` are a REAL cost, so an UNREGISTERED firm
    keeps ``(1 - delta)*y`` and a REGISTERED firm keeps ``(1 - delta)*(1 - tau)*y``
    of turnover, net of the iso-elastic OWN-FACTOR cost ``C(y; n, e)``.

    ``registered`` forces the regime; if ``None`` it is inferred from ``y >= T``
    elementwise (the hard-notch convention).
    """
    y = np.asarray(y, dtype=float)
    net_unreg = 1.0 - np.asarray(delta, dtype=float)
    net_reg = net_unreg * (1.0 - tau)
    if registered is None:
        reg = y >= T
    else:
        reg = np.broadcast_to(np.asarray(registered, dtype=bool), y.shape)
    net = np.where(reg, net_reg, net_unreg)
    return net * y - iso_cost(y, n, e)


def optimal_turnover_A(n, e, delta, tau=TAU):
    """Interior REGISTERED optimum ``y*_A = n*[(1 - delta)*(1 - tau)]**e`` (£)."""
    return np.asarray(n, dtype=float) * wedge_A(delta, tau) ** e


def unregistered_optimum_A(n, e, delta):
    """Interior UNREGISTERED optimum ``y*_A0 = n*(1 - delta)**e`` (£; tau = 0)."""
    return np.asarray(n, dtype=float) * value_added_share(delta) ** e


def recover_ability_A(y_obs, e, delta, tau=TAU, T=T_STAR):
    """Recover frictionless ability ``n`` from observed turnover under A.

    ``y_obs <  T*`` -> ``n = y_obs/(1-delta)**e``         (unregistered);
    ``y_obs >= T*`` -> ``n = y_obs/[(1-delta)(1-tau)]**e`` (registered).
    """
    y_obs = np.asarray(y_obs, dtype=float)
    n_unreg = y_obs / value_added_share(delta) ** e
    n_reg = y_obs / wedge_A(delta, tau) ** e
    return np.where(y_obs < T, n_unreg, n_reg)


def notch_jump_A(delta, tau=TAU, T=T_STAR):
    """Drop in net revenue crossing into registration: ``tau*(1 - delta)*T*`` (£).

    ``Delta_A = (1-delta)*T* - (1-delta)(1-tau)*T* = tau*(1-delta)*T*`` -- the SAME
    as B's notch jump.
    """
    return tau * (1.0 - np.asarray(delta, dtype=float)) * float(T)


def dominated_width_A(delta, tau=TAU, T=T_STAR):
    """Kleven-Waseem dominated-region width under A: ``T* * tau/(1 - tau)`` (£).

    Solving ``(1-delta)(1-tau)(T*+a) = (1-delta)T*`` cancels the common
    ``(1-delta)`` factor, so ``a_A = T* * tau/(1-tau)`` is INDEPENDENT of delta
    (== £21,250 for T*=£85k, tau=0.20). ``delta`` is accepted for signature
    symmetry with B but does not enter.
    """
    _ = delta  # intentionally unused: a_A is delta-independent.
    return float(T) * tau / (1.0 - tau)


def marginal_buncher_A(e, delta, tau=TAU, T=T_STAR):
    """Marginal buncher ``n_H`` under the A notch (£).

    Solves the indifference between bunching just below ``T*`` (unregistered,
    net ``1 - delta``) and registering at the post-notch interior optimum
    ``y1 = n*[(1-delta)(1-tau)]**e`` (net ``(1-delta)(1-tau)``). Mirrors
    :func:`deductible_model.marginal_buncher_deductible` /
    :func:`model.marginal_buncher_iso`. Returns ``(n_H, dy_star)`` with
    ``dy_star = n_H - T*``.

    When the effective notch ``tau*(1-delta)`` is below ``WEDGE_FLOOR`` (a
    near-zero-VAT, all-deductible sector) there is effectively no notch and no
    buncher: returns ``(nan, nan)``.
    """
    eff_notch = tau * (1.0 - float(delta))
    if eff_notch < WEDGE_FLOOR:
        return float("nan"), float("nan")

    net_unreg = 1.0 - float(delta)
    net_reg = net_unreg * (1.0 - tau)

    def gap(n):
        # Unregistered firm produces its interior optimum n*(1-delta)**e but is
        # CAPPED at T* (it bunches there if it would otherwise exceed T*).
        y_bunch = min(n * net_unreg**e, T)
        u_bunch = iso_profit(y_bunch, n, e, net=net_unreg)
        # Registered firm produces its taxed interior optimum.
        y1 = n * net_reg**e
        u_tax = iso_profit(y1, n, e, net=net_reg)
        return float(u_bunch - u_tax)

    # Bunching can only dominate once the cap binds (n*(1-delta)**e > T*); below
    # that the unregistered optimum is interior and strictly beats registering.
    lo, hi = T * (1.0 + 1e-6), T * 5.0
    while gap(lo) * gap(hi) > 0 and hi < T * 200:
        hi *= 1.5
    if gap(lo) * gap(hi) > 0:
        # No sign change: the cap never bites hard enough -> effectively no notch.
        return float("nan"), float("nan")
    n_H = brentq(gap, lo, hi)
    return float(n_H), float(n_H - T)


# ---------------------------------------------------------------------------
# B analogues (imported / wrapped from deductible_model, for side-by-side use)
# ---------------------------------------------------------------------------
def optimal_turnover_B(n, e, delta, tau=TAU):
    """Interior registered optimum under B: ``y*_B = n*[1 - tau*(1-delta)]**e``."""
    return dm.optimal_turnover(n, e, delta, tau)


def recover_ability_B(y_obs, e, delta, tau=TAU, T=T_STAR):
    """Recover ability under B (whole turnover kept when unregistered).

    ``y_obs <  T*`` -> ``n = y_obs`` (unregistered, net 1);
    ``y_obs >= T*`` -> ``n = y_obs/[1 - tau*(1-delta)]**e`` (registered).
    """
    y_obs = np.asarray(y_obs, dtype=float)
    n_reg = y_obs / wedge_B(delta, tau) ** e
    return np.where(y_obs < T, y_obs, n_reg)


def notch_jump_B(delta, tau=TAU, T=T_STAR):
    """B's notch jump ``tau*(1-delta)*T*`` (== A's)."""
    return dm.notch_jump(delta, tau, T)


def dominated_width_B(delta, tau=TAU, T=T_STAR):
    """B's dominated-region width ``T* * tau(1-delta)/(1 - tau(1-delta))`` (£)."""
    return dm.dominated_width(delta, tau, T)


def marginal_buncher_B(e, delta, tau=TAU, T=T_STAR):
    """B's marginal buncher (delegates to deductible_model)."""
    return dm.marginal_buncher_deductible(e, delta, tau, T)


# ---------------------------------------------------------------------------
# Verification: A's closed-form optimum vs numerical argmax (grid + scipy)
# ---------------------------------------------------------------------------
def _grid_argmax(n, e, delta, tau=TAU, *, npts=400_001, lo_frac=0.30, hi_frac=1.03):
    """Maximise A's registered profit over a fine y-grid in [lo*n, hi*n]."""
    grid = np.linspace(lo_frac * n, hi_frac * n, npts)
    prof = va_profit(grid, n, e, delta, tau, registered=True)
    return float(grid[int(np.argmax(prof))])


def _scalar_argmax(n, e, delta, tau=TAU):
    """Maximise A's registered profit with bounded ``minimize_scalar``."""
    res = minimize_scalar(
        lambda y: -float(va_profit(y, n, e, delta, tau, registered=True)),
        bounds=(1e-6, 2.0 * n),
        method="bounded",
        options={"xatol": 1e-8},
    )
    return float(res.x)


def verify_optimum(
    *,
    n_grid=None,
    deltas=(0.0, 0.3, 0.5, 0.7, 0.9),
    elasticities=ELASTICITIES,
    tau=TAU,
):
    """A's closed-form optimum vs fine-grid AND scipy argmax over (n, delta, e).

    Confirms strict concavity (pi'' < 0 at the optimum) and the delta-INDEPENDENCE
    of the dominated-region width (a_A == T*tau/(1-tau) == £21,250 for every
    delta). Returns a results dict.
    """
    if n_grid is None:
        n_grid = np.linspace(90_000.0, 250_000.0, 9)

    worst_rel = 0.0
    worst_abs = 0.0
    worst_pp = -np.inf  # least-negative second derivative seen
    rows = []
    for e in elasticities:
        for delta in deltas:
            rel_g = rel_s = abs_g = abs_s = 0.0
            for n in n_grid:
                y_cf = float(optimal_turnover_A(n, e, delta, tau))
                y_g = _grid_argmax(n, e, delta, tau)
                y_s = _scalar_argmax(n, e, delta, tau)
                abs_g = max(abs_g, abs(y_g - y_cf))
                abs_s = max(abs_s, abs(y_s - y_cf))
                rel_g = max(rel_g, abs(y_g - y_cf) / y_cf)
                rel_s = max(rel_s, abs(y_s - y_cf) / y_cf)
                h = 1e-4 * y_cf
                f_p = float(va_profit(y_cf + h, n, e, delta, tau, registered=True))
                f_0 = float(va_profit(y_cf, n, e, delta, tau, registered=True))
                f_m = float(va_profit(y_cf - h, n, e, delta, tau, registered=True))
                worst_pp = max(worst_pp, (f_p - 2.0 * f_0 + f_m) / (h * h))
            worst_rel = max(worst_rel, rel_g, rel_s)
            worst_abs = max(worst_abs, abs_g, abs_s)
            rows.append({"e": e, "delta": delta, "rel_grid": rel_g,
                         "rel_scalar": rel_s, "abs_grid": abs_g, "abs_scalar": abs_s})

    # delta-independence of a_A: compute across deltas; all must equal T*tau/(1-tau).
    a_target = float(T_STAR) * tau / (1.0 - tau)
    a_check = [(d, float(dominated_width_A(d, tau))) for d in (0.0, 0.3, 0.5, 0.7, 0.9)]
    a_max_dev = max(abs(a - a_target) for _, a in a_check)

    return {
        "rows": rows,
        "worst_rel": worst_rel,
        "worst_abs": worst_abs,
        "worst_second_derivative": float(worst_pp),
        "all_concave": bool(worst_pp < 0),
        "a_target": a_target,
        "a_check": a_check,
        "a_max_dev": float(a_max_dev),
        "a_delta_independent": bool(a_max_dev < 1e-9),
    }


# ---------------------------------------------------------------------------
# A-vs-B sector comparison
# ---------------------------------------------------------------------------
def compare_sectors(
    vintage="2023-24",
    *,
    e=E_HEADLINE,
    n_sample=150_000.0,
    y_obs=120_000.0,
    tau=TAU,
):
    """Side-by-side A vs B geometry for the representative sectors (real delta).

    Treats each sector's empirical net-VAT-to-turnover ratio as B's ``tau_eff``
    and backs out the implied delta; tabulates, per sector, the registered wedge
    (A vs B), the registered optimum at ``n_sample`` (A vs B), the ability
    recovered from a common observed ``y_obs`` (A vs B), the notch jump (shared),
    the dominated-region width (A flat vs B sector-specific) and the marginal
    buncher (A vs B).
    """
    ratios = sector_vat.sector_vat_ratios(vintage)
    pop_mean_ratio = float(ratios.attrs["population_mean"])

    rows = []
    for sic, name in REPRESENTATIVE_SECTORS:
        ratio = float(ratios.loc[sic]) if sic in ratios.index else pop_mean_ratio
        delta = float(dm.implied_delta(ratio, tau))
        nH_A, _ = marginal_buncher_A(e, delta, tau)
        nH_B, _ = marginal_buncher_B(e, delta, tau)
        rows.append({
            "sic": sic, "name": name, "ratio": ratio, "delta": delta,
            "w_A": float(wedge_A(delta, tau)), "w_B": float(wedge_B(delta, tau)),
            "y_A": float(optimal_turnover_A(n_sample, e, delta, tau)),
            "y_B": float(optimal_turnover_B(n_sample, e, delta, tau)),
            "n_A": float(recover_ability_A(y_obs, e, delta, tau)),
            "n_B": float(recover_ability_B(y_obs, e, delta, tau)),
            "jump_A": float(notch_jump_A(delta, tau)),
            "jump_B": float(notch_jump_B(delta, tau)),
            "a_A": float(dominated_width_A(delta, tau)),
            "a_B": float(dominated_width_B(delta, tau)),
            "nH_A": nH_A, "nH_B": nH_B,
        })
    return {"e": e, "n_sample": n_sample, "y_obs": y_obs, "rows": rows}


# ---------------------------------------------------------------------------
# Population comparison using real data (150MB CSV loaded once)
# ---------------------------------------------------------------------------
def compare_population(vintage="2023-24", *, e=E_HEADLINE, tau=TAU):
    """Population-weighted A-vs-B means using the real synthetic population.

    Loads :func:`model.load_reform_data`, attaches each firm's sector delta from
    :mod:`sector_vat`, and reports population-weighted mean registered wedge
    (A vs B), mean dominated-region width (A flat ~£21,250 vs B smaller), and how
    much A's recovered abilities exceed B's on average (A has the smaller wedge,
    so it attributes MORE distortion and recovers HIGHER abilities).
    """
    ratios = sector_vat.sector_vat_ratios(vintage)
    df = sector_vat._load_population_with_sic(vintage)
    tau_eff = sector_vat.attach_sector_tau0(df, ratios)  # per-firm tau*(1-delta)
    delta = dm.implied_delta(tau_eff, tau)               # per-firm delta

    w = df["weight"].to_numpy(dtype=float)
    t = df["turnover"].to_numpy(dtype=float)
    pos = t > 0

    wA = wedge_A(delta, tau)
    wB = wedge_B(delta, tau)
    aA = np.full_like(delta, dominated_width_A(0.0, tau))  # flat
    aB = dominated_width_B(delta, tau)

    def wmean(x, mask):
        return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))

    mean_wA = wmean(wA, pos)
    mean_wB = wmean(wB, pos)
    mean_aA = wmean(aA, pos)
    mean_aB = wmean(aB, pos)

    # Recovered abilities (registered firms only: turnover >= T*).
    reg = pos & (t >= T_STAR)
    nA = recover_ability_A(t, e, delta, tau)
    nB = recover_ability_B(t, e, delta, tau)
    ratio_n = np.where(nB > 0, nA / nB, 1.0)
    mean_nA = wmean(nA, reg)
    mean_nB = wmean(nB, reg)
    mean_ratio_n = wmean(ratio_n, reg)
    excess_pct = 100.0 * (mean_ratio_n - 1.0)

    return {
        "e": e,
        "n_firms": int(np.sum(reg)),
        "mean_wedge_A": mean_wA,
        "mean_wedge_B": mean_wB,
        "mean_wedge_gap": mean_wB - mean_wA,  # == population-weighted mean delta
        "mean_a_A": mean_aA,
        "mean_a_B": mean_aB,
        "mean_recovered_n_A": mean_nA,
        "mean_recovered_n_B": mean_nB,
        "mean_ability_ratio": mean_ratio_n,
        "ability_excess_pct": excess_pct,
    }


# ---------------------------------------------------------------------------
# Illustrative behavioural-response direction (NOT a full re-costing)
# ---------------------------------------------------------------------------
def reform_response_illustration(
    vintage="2023-24", *, e=E_HEADLINE, tau=TAU, tau_band=0.15
):
    """Illustrative intensive response of a band firm to a reduced-rate reform.

    A reduced-rate reform drops the value-added rate in the band from the
    statutory ``tau`` (0.20) to ``tau_band`` (e.g. 0.15). For a REGISTERED firm
    the intensive response is ``y*(tau_band)/y*(tau)``:

        A:  [(1 - tau_band)/(1 - tau)]**e         (delta cancels)
        B:  [(1 - tau_band(1-delta))/(1 - tau(1-delta))]**e

    A's smaller wedge means a given proportional rate change moves a more-distorted
    firm MORE, so A produces a LARGER base-broadening response -> reforms are
    relatively CHEAPER under A than B. This is ILLUSTRATIVE (single representative
    firm), NOT a re-costed population table.
    """
    ratios = sector_vat.sector_vat_ratios(vintage)
    pop_mean_ratio = float(ratios.attrs["population_mean"])

    rows = []
    for sic, name in REPRESENTATIVE_SECTORS:
        ratio = float(ratios.loc[sic]) if sic in ratios.index else pop_mean_ratio
        delta = float(dm.implied_delta(ratio, tau))
        scale_A = (wedge_A(delta, tau_band) / wedge_A(delta, tau)) ** e
        scale_B = (wedge_B(delta, tau_band) / wedge_B(delta, tau)) ** e
        rows.append({
            "sic": sic, "name": name, "delta": delta,
            "scale_A": float(scale_A), "scale_B": float(scale_B),
            "extra_pp": float(100.0 * (scale_A - scale_B)),
        })
    a_more = all(r["scale_A"] >= r["scale_B"] for r in rows)
    return {"e": e, "tau": tau, "tau_band": tau_band, "rows": rows,
            "A_response_larger": bool(a_more)}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _style_ax(ax):
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def make_figure(sector, path=None, *, tau=TAU):
    """Two-panel figure: wedge w_A vs w_B (A) and dominated width a_A vs a_B (B)."""
    if path is None:
        path = RESULTS_DIR / "value_added_comparison.png"
    path = Path(path)

    dd = np.linspace(0.0, 0.98, 400)
    wA = wedge_A(dd, tau)
    wB = wedge_B(dd, tau)
    aA = np.array([dominated_width_A(d, tau) for d in dd])
    aB = np.array([dominated_width_B(d, tau) for d in dd])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel A: registered retention wedge w_A vs w_B as functions of delta.
    axL.plot(dd, wA, color=PRIMARY, lw=2.4,
             label="A: $w_A=(1-\\delta)(1-\\tau)$  (inputs a real cost)")
    axL.plot(dd, wB, color=SECONDARY, lw=2.4, ls="--",
             label="B: $w_B=1-\\tau(1-\\delta)$  (paper)")
    for s in sector["rows"]:
        axL.plot(s["delta"], s["w_A"], "o", color=ACCENT, ms=8, zorder=6)
        axL.plot(s["delta"], s["w_B"], "s", color=ACCENT, ms=7, zorder=6,
                 markerfacecolor="white")
        axL.annotate(s["name"].split("(")[0].strip(), (s["delta"], s["w_A"]),
                     textcoords="offset points", xytext=(6, -12),
                     fontsize=TICK_SIZE - 4, color="black")
    axL.set_xlabel("Deductible share $\\delta$ (value-added share $=1-\\delta$)",
                   fontsize=LABEL_SIZE)
    axL.set_ylabel("Registered retention wedge", fontsize=LABEL_SIZE)
    axL.set_title("Registered net-of-tax wedge: A vs B  ($w_B-w_A=\\delta$)",
                  fontsize=LABEL_SIZE)
    _style_ax(axL)
    axL.legend(frameon=False, fontsize=TICK_SIZE - 1, loc="lower left")

    # --- Panel B: dominated-region width a_A (flat) vs a_B (declining).
    axR.plot(dd, aA / 1000.0, color=PRIMARY, lw=2.6,
             label="A: $a_A=T^*\\tau/(1-\\tau)=£21{,}250$ (flat)")
    axR.plot(dd, aB / 1000.0, color=SECONDARY, lw=2.4, ls="--",
             label="B: $a_B=T^*\\tau(1-\\delta)/(1-\\tau(1-\\delta))$")
    for s in sector["rows"]:
        axR.plot(s["delta"], s["a_A"] / 1000.0, "o", color=ACCENT, ms=8, zorder=6)
        axR.plot(s["delta"], s["a_B"] / 1000.0, "s", color=ACCENT, ms=7, zorder=6,
                 markerfacecolor="white")
        axR.annotate(s["name"].split("(")[0].strip(), (s["delta"], s["a_B"] / 1000.0),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=TICK_SIZE - 4, color="black")
    axR.set_xlabel("Deductible share $\\delta$ (value-added share $=1-\\delta$)",
                   fontsize=LABEL_SIZE)
    axR.set_ylabel("Dominated-region width (£k)", fontsize=LABEL_SIZE)
    axR.set_title("Dominated region: A flat £21,250 vs B sector-shrinking",
                  fontsize=LABEL_SIZE)
    _style_ax(axR)
    axR.legend(frameon=False, fontsize=TICK_SIZE - 1, loc="upper right")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def build_report(opt, sector, pop, reform):
    """Assemble the human-readable A-vs-B report (list of lines)."""
    L = []
    L.append("Value-added VAT notch: model A (inputs as a real cost) vs model B")
    L.append("=" * 72)
    L.append("TWO PROFIT FUNCTIONS (same iso-elastic C, different meaning of C)")
    L.append("-" * 72)
    L.append("  tau = 0.20 statutory; delta = sector deductible-input share of")
    L.append("  turnover; value-added share = (1 - delta); d = delta*y inputs.")
    L.append("")
    L.append("  MODEL B (existing paper / deductible_model.py):")
    L.append("    Unregistered:  pi_B = y                       - C(y;n,e)")
    L.append("    Registered:    pi_B = (1 - tau*(1-delta))*y   - C(y;n,e)")
    L.append("    -> C is TOTAL turnover cost; inputs are NOT separately netted.")
    L.append("")
    L.append("  MODEL A (this module):")
    L.append("    Unregistered:  pi_A = (1-delta)*y             - C(y;n,e)")
    L.append("    Registered:    pi_A = (1-delta)*(1-tau)*y     - C(y;n,e)")
    L.append("    -> C is OWN-FACTOR (NON-INPUT) cost; inputs d=delta*y are a")
    L.append("       real cost subtracted from turnover.")
    L.append("")
    L.append("  C(y;n,e) = (n/(1+1/e))*(y/n)**(1+1/e) is the SAME FUNCTION in both;")
    L.append("  only its economic INTERPRETATION differs. A and B coincide only at")
    L.append("  delta=0; in general w_B - w_A = delta exactly.")
    L.append("")
    L.append("  Registered wedge   w_A=(1-delta)(1-tau)   w_B=1-tau(1-delta)")
    L.append("  Registered optimum y*_A=n[(1-delta)(1-tau)]**e   "
             "y*_B=n[1-tau(1-delta)]**e")
    L.append("  Notch jump at T*   Delta_A = Delta_B = tau*(1-delta)*T*  (IDENTICAL)")
    L.append("  Dominated width    a_A = T*tau/(1-tau) (delta-INDEPENDENT)")
    L.append("                     a_B = T*tau(1-delta)/(1-tau(1-delta)) (shrinks)")
    L.append("")

    # 1. A optimum verification.
    L.append("1. MODEL-A OPTIMUM VERIFICATION (closed form vs grid AND scipy)")
    L.append("-" * 72)
    L.append(f"   {'e':>5} {'delta':>7} {'rel_grid':>11} {'rel_scalar':>11}"
             f" {'abs_grid(£)':>13} {'abs_scalar(£)':>14}")
    for r in opt["rows"]:
        L.append(f"   {r['e']:>5} {r['delta']:>7.3f} {r['rel_grid']:>11.2e}"
                 f" {r['rel_scalar']:>11.2e} {r['abs_grid']:>13.4f}"
                 f" {r['abs_scalar']:>14.4f}")
    L.append(f"   worst relative error: {opt['worst_rel']:.2e}"
             f"  (target < 1e-4: {'PASS' if opt['worst_rel'] < 1e-4 else 'FAIL'})")
    L.append(f"   worst (least-negative) pi''(y*): {opt['worst_second_derivative']:.4e}"
             f"  strictly concave: {'PASS' if opt['all_concave'] else 'FAIL'}")
    L.append("")
    L.append("   DOMINATED-REGION delta-INDEPENDENCE (a_A across delta):")
    for d, a in opt["a_check"]:
        L.append(f"     delta={d:<5g}  a_A = £{a:,.2f}")
    L.append(f"   all equal £{opt['a_target']:,.2f}  (max dev {opt['a_max_dev']:.2e}): "
             f"{'PASS' if opt['a_delta_independent'] else 'FAIL'}")
    L.append("")

    # 2. Sector comparison.
    L.append("2. A-vs-B SECTOR COMPARISON (real delta, "
             f"e={sector['e']}, n=£{sector['n_sample']:,.0f}, "
             f"y_obs=£{sector['y_obs']:,.0f})")
    L.append("-" * 72)
    L.append(f"   {'sector':<26}{'delta':>6}{'w_A':>7}{'w_B':>7}"
             f"{'y*_A':>9}{'y*_B':>9}{'n_A':>9}{'n_B':>9}{'jump':>8}"
             f"{'a_A':>8}{'a_B':>8}")
    for s in sector["rows"]:
        nm = f"{s['sic']:>2} {s['name'].split('(')[0].strip()}"[:25]
        L.append(f"   {nm:<26}{s['delta']:>6.3f}{s['w_A']:>7.3f}{s['w_B']:>7.3f}"
                 f"{s['y_A']:>9,.0f}{s['y_B']:>9,.0f}{s['n_A']:>9,.0f}"
                 f"{s['n_B']:>9,.0f}{s['jump_A']:>8,.0f}"
                 f"{s['a_A']:>8,.0f}{s['a_B']:>8,.0f}")
    L.append("   marginal buncher n_H (A vs B):")
    for s in sector["rows"]:
        nm = s["name"].split("(")[0].strip()[:25]
        nHa = "no notch" if np.isnan(s["nH_A"]) else f"£{s['nH_A']:,.0f}"
        nHb = "no notch" if np.isnan(s["nH_B"]) else f"£{s['nH_B']:,.0f}"
        L.append(f"     {nm:<26} n_H,A={nHa:>12}   n_H,B={nHb:>12}")
    L.append("   (jump_A == jump_B by construction; a_A flat £21,250, a_B shrinks)")
    L.append("")

    # 3. Population comparison.
    L.append("3. POPULATION COMPARISON (real synthetic population, weighted means)")
    L.append("-" * 72)
    L.append(f"   registered firms used (turnover >= T*): {pop['n_firms']:,}")
    L.append(f"   mean registered wedge   w_A = {pop['mean_wedge_A']:.4f}   "
             f"w_B = {pop['mean_wedge_B']:.4f}   "
             f"(w_B - w_A = {pop['mean_wedge_gap']:.4f} = mean delta)")
    L.append(f"   mean dominated width    a_A = £{pop['mean_a_A']:,.0f} (flat)   "
             f"a_B = £{pop['mean_a_B']:,.0f} (smaller)")
    L.append(f"   mean recovered ability  n_A = £{pop['mean_recovered_n_A']:,.0f}   "
             f"n_B = £{pop['mean_recovered_n_B']:,.0f}")
    L.append(f"   A recovers HIGHER abilities: mean n_A/n_B = "
             f"{pop['mean_ability_ratio']:.4f} "
             f"(+{pop['ability_excess_pct']:.2f}% on average)")
    L.append("   A's smaller wedge attributes MORE of observed turnover to the tax")
    L.append("   distortion, so it recovers larger underlying abilities than B.")
    L.append("")

    # 4. Reform direction illustration.
    L.append("4. ILLUSTRATIVE REFORM-RESPONSE DIRECTION (NOT a re-costing)")
    L.append("-" * 72)
    L.append(f"   Reduced-rate band reform: value-added rate {reform['tau']:.2f} -> "
             f"{reform['tau_band']:.2f}. Intensive scaling y*(new)/y*(old):")
    L.append(f"   {'sector':<28}{'delta':>7}{'scale_A':>10}{'scale_B':>10}"
             f"{'A-B (pp)':>10}")
    for r in reform["rows"]:
        nm = r["name"].split("(")[0].strip()[:27]
        L.append(f"   {nm:<28}{r['delta']:>7.3f}{r['scale_A']:>10.4f}"
                 f"{r['scale_B']:>10.4f}{r['extra_pp']:>10.3f}")
    L.append("   A's wedge is smaller, so a given proportional rate cut moves a")
    L.append("   MORE-distorted firm more: A's intensive response exceeds B's "
             f"({'confirmed' if reform['A_response_larger'] else 'NOT confirmed'}).")
    L.append("   DIRECTION: larger base-broadening response under A => reforms are")
    L.append("   relatively CHEAPER under A than under B. (Illustrative single-firm")
    L.append("   scaling; the full population reform-cost engine is out of scope.)")
    L.append("")

    # Verdict.
    L.append("VERDICT (defensibility -- neither is identified by the data)")
    L.append("=" * 72)
    L.append("Both readings use the IDENTICAL iso-elastic cost form; they differ")
    L.append("only in whether deductible input spend d=delta*y is subtracted as a")
    L.append("real cost (A) or folded into total cost C (B). The data cannot tell")
    L.append("them apart -- it is a MODELLING CONVENTION, not an identified fact.")
    L.append("")
    L.append("* Model B is INTERNALLY CONSISTENT with the rest of the paper: its")
    L.append("  turnover-based ability recovery and calibration to net-VAT-to-")
    L.append("  turnover data both presume the 'C = total turnover cost' reading,")
    L.append("  and its sector-shrinking dominated region follows from the same")
    L.append("  effective wedge tau*(1-delta) used everywhere else.")
    L.append("* Model A is the more LITERAL 'VAT on value added with inputs as a")
    L.append("  real cost' statement. Notably it REPRODUCES the paper's headline")
    L.append("  £21,250 Kleven-Waseem dominated region for EVERY sector (the")
    L.append("  (1-delta) cancels), whereas B makes that region sector-specific.")
    L.append("* Both agree EXACTLY on the notch jump tau*(1-delta)*T*. They differ")
    L.append("  on the retained wedge (by exactly delta), hence on recovered")
    L.append("  abilities (A higher) and on reform responsiveness (A larger,")
    L.append("  reforms relatively cheaper).")
    L.append("")
    L.append("Recommendation: state the convention explicitly. B is defensible and")
    L.append("coherent with the paper as written; A is an equally defensible")
    L.append("alternative that happens to restore the canonical £21,250 figure.")
    return L


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(vintage="2023-24"):
    """Run A verification, A-vs-B sector/population/reform comparison; write outputs."""
    opt = verify_optimum()
    sector = compare_sectors(vintage)
    pop = compare_population(vintage)
    reform = reform_response_illustration(vintage)
    lines = build_report(opt, sector, pop, reform)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt = RESULTS_DIR / "value_added_comparison.txt"
    Path(txt).write_text("\n".join(lines) + "\n")
    fig = make_figure(sector)

    print("\n".join(lines))
    print(f"\nWrote {txt}")
    print(f"Wrote {fig}")
    return {"opt": opt, "sector": sector, "pop": pop, "reform": reform}


def cli(argv=None):
    """argparse entry point."""
    ap = argparse.ArgumentParser(
        prog="firm-microsim-value-added-model",
        description="Alternative value-added VAT-notch model A (deductible inputs "
                    "subtracted as a real cost) and a rigorous comparison to the "
                    "paper's model B (deductible_model.py). Verifies A's optimum, "
                    "shows the delta-independent £21,250 dominated region, and "
                    "tabulates sector + population A-vs-B differences. Writes "
                    "results/value_added_comparison.{png,txt}.",
    )
    ap.add_argument("--vintage", default="2023-24",
                    help="data vintage (default: 2023-24)")
    args = ap.parse_args(argv)
    return main(args.vintage)


if __name__ == "__main__":
    cli()
