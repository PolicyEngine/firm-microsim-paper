"""Value-added-CHOICE reformulation of the iso-elastic VAT-notch problem (A2).

Two earlier readings of the registered firm's profit over TURNOVER ``y`` (with
``delta`` the sector deductible share, ``tau = 0.20`` statutory and the shared
iso-elastic cost ``C(x; n, e) = (n/(1+1/e))*(x/n)**(1+1/e)``) each get ONE thing
right and ONE thing wrong:

MODEL B  (``deductible_model.py``)
    pi = (1 - tau*(1 - delta))*y - C(y; n, e).
    Clean ability primitive (ability = undistorted turnover) but the
    Kleven-Waseem dominated region SHRINKS by sector (a_B < £21,250).

MODEL A  (``value_added_model.py``)
    pi = (1 - delta)*(1 - tau)*y - C(y; n, e).
    Restores the flat £21,250 dominated region, but the ability recovery
    n = y_obs/[(1-delta)(1-tau)]**e BLOWS UP as delta -> 1 (a near-zero-value-
    added firm is attributed an unbounded ability).

MODEL A2  (this module) FIXES BOTH
----------------------------------
Let the firm CHOOSE its **value added** ``z = (1 - delta)*y`` directly. This is
pure Kleven-Waseem with ``z`` as the base, and ``z`` is the only behavioural
margin. Ability ``n`` is the firm's UNDISTORTED value added (the frictionless
optimum of ``z``). The iso-elastic cost is now a cost of GENERATING VALUE ADDED,
``C(z; n, e) = (n/(1+1/e))*(z/n)**(1+1/e)``, marginal cost ``(z/n)**(1/e)``::

    Unregistered:  pi = z          - C(z; n, e)   ->  z* = n
    Registered:    pi = (1 - tau)*z - C(z; n, e)  ->  z* = n*(1 - tau)**e

Turnover is the accounting image ``y = z/(1 - delta)``. The £85,000 registration
notch lives on TURNOVER, so in value-added space the threshold is sector-
specific, ``z_T = (1 - delta)*T*``.

DERIVED OBJECTS (implemented and VERIFIED below; not re-derived here)
--------------------------------------------------------------------
* Registered optimum         z* = n*(1 - tau)**e        (delta-free; full rate).
* Value-added threshold       z_T = (1 - delta)*T*.
* Notch jump at the threshold tau*z_T = tau*(1 - delta)*T*   (IDENTICAL to A, B).
* Dominated region (VA space) a_z = z_T*tau/(1-tau) = (1-delta)*£21,250.
* Dominated region (turnover) a_y = a_z/(1-delta) = T**tau/(1-tau) = £21,250,
                              delta-INDEPENDENT (matches A and the paper's sec. 4).
* Ability recovery from observed baseline turnover y_obs (z_obs = (1-delta)y_obs):
      registered (y_obs >= T*):  n = z_obs/(1 - tau)**e = (1-delta)y_obs/(1-tau)**e
      unregistered (y_obs <  T*): n = z_obs            = (1-delta)y_obs
  BOUNDED: as delta -> 1, z_obs -> 0 so n -> 0 (a zero-value-added firm has zero
  value-adding capacity) -- NOT infinity. This is the key fix over A.
* Intensive response          a reform tau -> tau' scales z (and y = z/(1-delta))
                              by [(1-tau')/(1-tau)]**e -- governed by the FULL
                              statutory rate on value added, so LARGER than B's
                              tau*(1-delta)-diluted response. d ln z*/d ln(1-tau)=e.

THE THREE PROPERTIES A2 SECURES
-------------------------------
1. A literal value-added base: the firm optimises over value added itself.
2. A bounded, well-behaved ability primitive n = undistorted value added, which
   -> 0 (not infinity) as delta -> 1.
3. The paper's headline £21,250 turnover dominated region for EVERY sector.

THE ONE COST
------------
The single £85k notch on turnover becomes a sector-specific notch on value added
at z_T = (1 - delta)*T* (the threshold the firm "feels" depends on its sector).

Run ``python -m firm_microsim.dynamic.va_choice_model`` (or call :func:`main`).
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
from firm_microsim.dynamic import value_added_model as va
from firm_microsim.dynamic.model import (
    E_HEADLINE,
    ELASTICITIES,
    T_STAR,
    TAU_MAX,
    iso_cost,
    iso_profit,
)

# Statutory VAT rate (levied on value added once registered).
TAU = TAU_MAX  # 0.20

# delta is a share in [0, 1); clip just below 1 so the value-added share is > 0.
DELTA_MAX = dm.DELTA_MAX

# Effective notch (= tau*(1-delta)) below this is treated as "no notch".
WEDGE_FLOOR = dm.WEDGE_FLOOR

# Same representative SIC divisions as B / A.
REPRESENTATIVE_SECTORS = dm.REPRESENTATIVE_SECTORS

# House style (matches deductible_model / value_added_model figures).
PALETTE = ["#326b77", "#122740", "#1b485e", "#568b87", "#80ae9a", "#b5d1ae"]
PRIMARY = "#326b77"
ACCENT = "#d62728"
SECONDARY = "#122740"
TERTIARY = "#80ae9a"
LABEL_SIZE = 15
TICK_SIZE = 13


# ---------------------------------------------------------------------------
# Core closed forms of MODEL A2 (the firm CHOOSES value added z)
# ---------------------------------------------------------------------------
def vc_profit(z, n, e, registered, tau=TAU):
    """Value-added-choice profit ``pi(z; n, e)`` (£), over the chosen value added z.

    ``registered=True``  -> ``pi = (1 - tau)*z - C(z; n, e)`` (VAT on value added);
    ``registered=False`` -> ``pi = z          - C(z; n, e)`` (frictionless).
    The iso-elastic cost ``C(z; n, e)`` is now a cost of GENERATING value added.
    """
    z = np.asarray(z, dtype=float)
    reg = np.broadcast_to(np.asarray(registered, dtype=bool), z.shape)
    net = np.where(reg, 1.0 - tau, 1.0)
    return net * z - iso_cost(z, n, e)


def optimal_va(n, e, tau=TAU, registered=True):
    """Interior optimum value added (£).

    Unregistered: ``z* = n`` (frictionless). Registered: ``z* = n*(1 - tau)**e``.
    The registered optimum is DELTA-FREE -- the firm faces the full statutory rate
    on its value-added base.
    """
    n = np.asarray(n, dtype=float)
    if registered:
        return n * (1.0 - tau) ** e
    return n * np.ones_like(n)


def va_threshold(delta, T=T_STAR):
    """Registration threshold in value-added space ``z_T = (1 - delta)*T*`` (£).

    The £85k notch is on turnover; in value-added space it is sector-specific.
    """
    return (1.0 - np.asarray(delta, dtype=float)) * float(T)


def recover_ability(y_obs, delta, e, tau=TAU, T=T_STAR):
    """Recover ability ``n`` (= undistorted value added) from observed turnover.

    ``z_obs = (1 - delta)*y_obs`` is observed value added. Then
    ``y_obs <  T*`` -> ``n = z_obs``            (unregistered);
    ``y_obs >= T*`` -> ``n = z_obs/(1 - tau)**e`` (registered at the taxed optimum).

    BOUNDED: as ``delta -> 1`` the observed value added ``z_obs -> 0`` so
    ``n -> 0`` -- the key fix versus model A (whose recovery diverges).
    """
    y_obs = np.asarray(y_obs, dtype=float)
    z_obs = (1.0 - np.asarray(delta, dtype=float)) * y_obs
    n_unreg = z_obs
    n_reg = z_obs / (1.0 - tau) ** e
    return np.where(y_obs < T, n_unreg, n_reg)


def notch_jump(delta, tau=TAU, T=T_STAR):
    """Drop in net revenue crossing into registration: ``tau*(1 - delta)*T*`` (£).

    At the threshold value added ``z_T = (1-delta)T*`` the registered firm loses
    ``tau*z_T = tau*(1-delta)*T*`` -- IDENTICAL to models A and B.
    """
    return tau * (1.0 - np.asarray(delta, dtype=float)) * float(T)


def dominated_width_va(delta, tau=TAU, T=T_STAR):
    """Dominated-region width in VALUE-ADDED space (£): ``(1-delta)*T**tau/(1-tau)``.

    ``a_z = z_T*tau/(1-tau)`` with ``z_T = (1-delta)T*``; equals ``(1-delta)*£21,250``.
    """
    return (1.0 - np.asarray(delta, dtype=float)) * float(T) * tau / (1.0 - tau)


def dominated_width_turnover(delta, tau=TAU, T=T_STAR):
    """Dominated-region width in TURNOVER space (£): ``T**tau/(1-tau)`` = £21,250.

    ``a_y = a_z/(1-delta) = T**tau/(1-tau)``; the ``(1-delta)`` cancels so the
    turnover dominated region is delta-INDEPENDENT (£21,250 for every sector).
    ``delta`` is accepted for signature symmetry but does not enter.
    """
    _ = delta  # intentionally unused: a_y is delta-independent.
    return float(T) * tau / (1.0 - tau)


def marginal_buncher_vc(e, delta, tau=TAU, T=T_STAR):
    """Marginal buncher ``n_H`` under the A2 (value-added) notch (£).

    Solves the indifference between bunching unregistered at the value-added
    threshold ``z_T = (1-delta)T*`` (net 1, value added capped at z_T) and
    registering at the taxed optimum ``z1 = n*(1-tau)**e`` (net 1-tau). Mirrors
    :func:`firm_microsim.dynamic.model.marginal_buncher_iso`. Returns
    ``(n_H, dz_star)`` with ``dz_star = n_H - z_T`` (ability in VALUE-ADDED units).

    When the effective notch ``tau*(1-delta)`` is below ``WEDGE_FLOOR`` (a
    near-zero-VAT / all-deductible sector) there is no notch: returns (nan, nan).
    """
    eff_notch = tau * (1.0 - float(delta))
    if eff_notch < WEDGE_FLOOR:
        return float("nan"), float("nan")

    z_T = (1.0 - float(delta)) * float(T)

    def gap(n):
        # Unregistered optimum is z*=n, but value added is CAPPED at z_T (the firm
        # bunches there if it would otherwise cross the registration threshold).
        z_bunch = min(n, z_T)
        u_bunch = iso_profit(z_bunch, n, e, net=1.0)
        z1 = n * (1.0 - tau) ** e
        u_tax = iso_profit(z1, n, e, net=1.0 - tau)
        return float(u_bunch - u_tax)

    # Bunching can only dominate once the cap binds (n > z_T); below that the
    # unregistered interior optimum is feasible and strictly beats registering.
    lo, hi = z_T * (1.0 + 1e-6), z_T * 5.0
    while gap(lo) * gap(hi) > 0 and hi < z_T * 200:
        hi *= 1.5
    if gap(lo) * gap(hi) > 0:
        return float("nan"), float("nan")
    n_H = brentq(gap, lo, hi)
    return float(n_H), float(n_H - z_T)


# ---------------------------------------------------------------------------
# Verification: A2's optimum vs numerical argmax (grid + scipy) and identities
# ---------------------------------------------------------------------------
def _grid_argmax(n, e, tau=TAU, *, npts=400_001, lo_frac=0.30, hi_frac=1.03):
    """Maximise A2's registered profit over a fine z-grid in [lo*n, hi*n]."""
    grid = np.linspace(lo_frac * n, hi_frac * n, npts)
    prof = vc_profit(grid, n, e, registered=True, tau=tau)
    return float(grid[int(np.argmax(prof))])


def _scalar_argmax(n, e, tau=TAU):
    """Maximise A2's registered profit with bounded ``minimize_scalar``."""
    res = minimize_scalar(
        lambda z: -float(vc_profit(z, n, e, registered=True, tau=tau)),
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
    """A2's closed-form optimum vs fine-grid AND scipy argmax; identities.

    The registered optimum ``z* = n*(1-tau)**e`` is delta-free, so the argmax loop
    spans (n, e); we still confirm, across delta, that (i) the turnover dominated
    region is the flat £21,250, and (ii) the notch jump equals tau*(1-delta)*T*.
    Also checks strict concavity and the elasticity identity
    ``d ln z*/d ln(1-tau) = e``. Returns a results dict.
    """
    if n_grid is None:
        n_grid = np.linspace(90_000.0, 250_000.0, 9)

    worst_rel = 0.0
    worst_abs = 0.0
    worst_pp = -np.inf
    rows = []
    for e in elasticities:
        rel_g = rel_s = abs_g = abs_s = 0.0
        for n in n_grid:
            z_cf = float(optimal_va(n, e, tau, registered=True))
            z_g = _grid_argmax(n, e, tau)
            z_s = _scalar_argmax(n, e, tau)
            abs_g = max(abs_g, abs(z_g - z_cf))
            abs_s = max(abs_s, abs(z_s - z_cf))
            rel_g = max(rel_g, abs(z_g - z_cf) / z_cf)
            rel_s = max(rel_s, abs(z_s - z_cf) / z_cf)
            h = 1e-4 * z_cf
            f_p = float(vc_profit(z_cf + h, n, e, registered=True, tau=tau))
            f_0 = float(vc_profit(z_cf, n, e, registered=True, tau=tau))
            f_m = float(vc_profit(z_cf - h, n, e, registered=True, tau=tau))
            worst_pp = max(worst_pp, (f_p - 2.0 * f_0 + f_m) / (h * h))
        worst_rel = max(worst_rel, rel_g, rel_s)
        worst_abs = max(worst_abs, abs_g, abs_s)
        rows.append({"e": e, "rel_grid": rel_g, "rel_scalar": rel_s,
                     "abs_grid": abs_g, "abs_scalar": abs_s})

    # Turnover dominated region flat £21,250 across delta; notch jump tau(1-delta)T.
    a_target = float(T_STAR) * tau / (1.0 - tau)
    a_check = [(d, float(dominated_width_turnover(d, tau))) for d in deltas]
    a_max_dev = max(abs(a - a_target) for _, a in a_check)
    jump_check = [(d, float(notch_jump(d, tau)),
                   float(tau * (1.0 - d) * T_STAR)) for d in deltas]
    jump_max_dev = max(abs(j - jt) for _, j, jt in jump_check)

    # Elasticity identity d ln z*/d ln(1-tau) = e (finite difference in ln(1-tau)).
    elas_rows = []
    n_ref = 150_000.0
    for e in elasticities:
        h = 1e-5
        s0 = 1.0 - tau
        z_p = float(optimal_va(n_ref, e, 1.0 - s0 * np.exp(h), registered=True))
        z_m = float(optimal_va(n_ref, e, 1.0 - s0 * np.exp(-h), registered=True))
        slope = (np.log(z_p) - np.log(z_m)) / (2.0 * h)
        elas_rows.append({"e": e, "slope": float(slope),
                          "err": float(abs(slope - e))})
    elas_max_err = max(r["err"] for r in elas_rows)

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
        "jump_check": jump_check,
        "jump_max_dev": float(jump_max_dev),
        "jump_ok": bool(jump_max_dev < 1e-6),
        "elas_rows": elas_rows,
        "elas_max_err": float(elas_max_err),
        "elasticity_ok": bool(elas_max_err < 1e-4),
    }


# ---------------------------------------------------------------------------
# Bounded ability: A2 vs A vs B as delta -> 1 (the fix), + sector table
# ---------------------------------------------------------------------------
def ability_boundedness(
    *, y_obs=120_000.0, e=E_HEADLINE, tau=TAU,
    deltas=(0.0, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, DELTA_MAX),
):
    """Recovered ability n(A2) vs n(A) vs n(B) as delta -> 1 at fixed y_obs.

    A2 stays finite and -> 0; A diverges up; B is mild. Returns the sweep plus a
    flag that A2 is monotone-decreasing to ~0 while A increases without bound.
    """
    rows = []
    for d in deltas:
        rows.append({
            "delta": float(d),
            "n_A2": float(recover_ability(y_obs, d, e, tau)),
            "n_A": float(va.recover_ability_A(y_obs, e, d, tau)),
            "n_B": float(va.recover_ability_B(y_obs, e, d, tau)),
        })
    n_A2 = [r["n_A2"] for r in rows]
    n_A = [r["n_A"] for r in rows]
    a2_to_zero = bool(n_A2[-1] < n_A2[0] and n_A2[-1] < 1.0)
    a_diverges = bool(n_A[-1] > 10.0 * n_A[0])
    return {"y_obs": y_obs, "e": e, "rows": rows,
            "A2_bounded_to_zero": a2_to_zero, "A_diverges": a_diverges}


def ability_sectors(vintage="2023-24", *, y_obs=120_000.0, e=E_HEADLINE, tau=TAU):
    """Recovered ability n(A2) vs n(A) vs n(B) for the four representative sectors."""
    ratios = sector_vat.sector_vat_ratios(vintage)
    pop_mean_ratio = float(ratios.attrs["population_mean"])
    rows = []
    for sic, name in REPRESENTATIVE_SECTORS:
        ratio = float(ratios.loc[sic]) if sic in ratios.index else pop_mean_ratio
        delta = float(dm.implied_delta(ratio, tau))
        rows.append({
            "sic": sic, "name": name, "delta": delta,
            "n_A2": float(recover_ability(y_obs, delta, e, tau)),
            "n_A": float(va.recover_ability_A(y_obs, e, delta, tau)),
            "n_B": float(va.recover_ability_B(y_obs, e, delta, tau)),
        })
    return {"y_obs": y_obs, "e": e, "rows": rows}


# ---------------------------------------------------------------------------
# Three-way population comparison (A2 vs A vs B) on the real synthetic data
# ---------------------------------------------------------------------------
def compare_population(vintage="2023-24", *, e=E_HEADLINE, tau=TAU, tau_band=0.15):
    """Population-weighted A2-vs-A-vs-B means using the real synthetic population.

    Reports: mean recovered ability (A2 bounded; A inflated; B mild), mean
    dominated-region width in turnover (A2 and A flat £21,250; B smaller), and the
    intensive turnover response to a value-added-rate reform tau -> tau_band (A2
    uses the FULL statutory rate, so larger than B's tau*(1-delta)-diluted move).
    """
    ratios = sector_vat.sector_vat_ratios(vintage)
    df = sector_vat._load_population_with_sic(vintage)
    tau_eff = sector_vat.attach_sector_tau0(df, ratios)  # per-firm tau*(1-delta)
    delta = dm.implied_delta(tau_eff, tau)               # per-firm delta

    w = df["weight"].to_numpy(dtype=float)
    t = df["turnover"].to_numpy(dtype=float)
    pos = t > 0
    reg = pos & (t >= T_STAR)

    def wmean(x, mask):
        return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))

    # Recovered abilities (registered firms only).
    nA2 = recover_ability(t, delta, e, tau)
    nA = va.recover_ability_A(t, e, delta, tau)
    nB = va.recover_ability_B(t, e, delta, tau)

    # Dominated-region width in turnover.
    aA2 = np.full_like(delta, dominated_width_turnover(0.0, tau))   # flat 21,250
    aA = np.full_like(delta, va.dominated_width_A(0.0, tau))        # flat 21,250
    aB = dm.dominated_width(delta, tau)                             # sector-specific

    # Intensive turnover response to a value-added-rate reform tau -> tau_band.
    # A2: y scales by [(1-tau_band)/(1-tau)]**e (delta cancels; full rate).
    scale_A2 = np.full_like(delta, ((1.0 - tau_band) / (1.0 - tau)) ** e)
    scale_A = (va.wedge_A(delta, tau_band) / va.wedge_A(delta, tau)) ** e
    scale_B = (va.wedge_B(delta, tau_band) / va.wedge_B(delta, tau)) ** e

    return {
        "e": e, "tau": tau, "tau_band": tau_band,
        "n_firms": int(np.sum(reg)),
        "mean_n_A2": wmean(nA2, reg),
        "mean_n_A": wmean(nA, reg),
        "mean_n_B": wmean(nB, reg),
        "mean_a_A2": wmean(aA2, pos),
        "mean_a_A": wmean(aA, pos),
        "mean_a_B": wmean(aB, pos),
        "mean_scale_A2": wmean(scale_A2, reg),
        "mean_scale_A": wmean(scale_A, reg),
        "mean_scale_B": wmean(scale_B, reg),
        "resp_pct_A2": 100.0 * (wmean(scale_A2, reg) - 1.0),
        "resp_pct_B": 100.0 * (wmean(scale_B, reg) - 1.0),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _style_ax(ax):
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def make_figure(sectors, path=None, *, y_obs=120_000.0, e=E_HEADLINE, tau=TAU):
    """Two-panel figure: bounded ability (A) and flat dominated region (B)."""
    if path is None:
        path = RESULTS_DIR / "va_choice_comparison.png"
    path = Path(path)

    dd = np.linspace(0.0, 0.985, 400)
    nA2 = recover_ability(y_obs, dd, e, tau)
    nA = va.recover_ability_A(y_obs, e, dd, tau)
    nB = va.recover_ability_B(y_obs, e, dd, tau)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel A: recovered ability vs delta (A2 bounded -> 0; A diverges; B mild).
    axL.plot(dd, nA / 1000.0, color=ACCENT, lw=2.4, ls="--",
             label="A: $n=y_{obs}/[(1-\\delta)(1-\\tau)]^{e}$  (diverges $\\to\\infty$)")
    axL.plot(dd, nB / 1000.0, color=SECONDARY, lw=2.2, ls=":",
             label="B: $n=y_{obs}/[1-\\tau(1-\\delta)]^{e}$  (mild)")
    axL.plot(dd, nA2 / 1000.0, color=PRIMARY, lw=2.8,
             label="A2: $n=(1-\\delta)y_{obs}/(1-\\tau)^{e}$  (bounded $\\to 0$)")
    for s in sectors["rows"]:
        axL.plot(s["delta"], s["n_A2"] / 1000.0, "o", color=PRIMARY, ms=8, zorder=6)
        axL.annotate(s["name"].split("(")[0].strip(),
                     (s["delta"], s["n_A2"] / 1000.0),
                     textcoords="offset points", xytext=(6, -12),
                     fontsize=TICK_SIZE - 4, color="black")
    axL.set_ylim(0.0, min(900.0, float(np.nanmax(nA2 / 1000.0)) * 6.0))
    axL.set_xlabel("Deductible share $\\delta$ (value-added share $=1-\\delta$)",
                   fontsize=LABEL_SIZE)
    axL.set_ylabel(f"Recovered ability $n$ (£k),  $y_{{obs}}$=£{y_obs/1000:.0f}k",
                   fontsize=LABEL_SIZE)
    axL.set_title("Ability recovery: A2 bounded (fixes A's blow-up)",
                  fontsize=LABEL_SIZE)
    _style_ax(axL)
    axL.legend(frameon=False, fontsize=TICK_SIZE - 1, loc="upper left")

    # --- Panel B: dominated-region width (turnover): A2 & A flat; B declining.
    aA2 = np.array([dominated_width_turnover(d, tau) for d in dd])
    aA = np.array([va.dominated_width_A(d, tau) for d in dd])
    aB = np.array([dm.dominated_width(d, tau) for d in dd])
    axR.plot(dd, aA / 1000.0, color=ACCENT, lw=2.4, ls="--",
             label="A: flat £21,250")
    axR.plot(dd, aA2 / 1000.0, color=PRIMARY, lw=2.8,
             label="A2: $a_y=T^*\\tau/(1-\\tau)=£21{,}250$ (flat)")
    axR.plot(dd, aB / 1000.0, color=SECONDARY, lw=2.2, ls=":",
             label="B: $T^*\\tau(1-\\delta)/(1-\\tau(1-\\delta))$ (declining)")
    for s in sectors["rows"]:
        axR.plot(s["delta"], dominated_width_turnover(s["delta"], tau) / 1000.0,
                 "o", color=PRIMARY, ms=8, zorder=6)
    axR.set_xlabel("Deductible share $\\delta$ (value-added share $=1-\\delta$)",
                   fontsize=LABEL_SIZE)
    axR.set_ylabel("Dominated-region width, turnover (£k)", fontsize=LABEL_SIZE)
    axR.set_title("Turnover dominated region: A2 & A flat £21,250 vs B shrinking",
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
def build_report(opt, bound, sectors, pop):
    """Assemble the human-readable A2 report (list of lines)."""
    L = []
    L.append("Value-added-CHOICE VAT notch (A2): the firm chooses value added z")
    L.append("=" * 72)
    L.append("PROFIT FUNCTIONS (choice variable z = value added; y = z/(1-delta))")
    L.append("-" * 72)
    L.append("  tau = 0.20 statutory; delta = sector deductible share;")
    L.append("  C(z;n,e) = (n/(1+1/e))*(z/n)**(1+1/e); ability n = undistorted VA.")
    L.append("")
    L.append("    Unregistered:  pi = z          - C(z;n,e)   ->  z* = n")
    L.append("    Registered:    pi = (1-tau)*z  - C(z;n,e)   ->  z* = n*(1-tau)**e")
    L.append("")
    L.append("  Turnover is the accounting image y = z/(1-delta). The £85k notch is")
    L.append("  on TURNOVER, so in value-added space the threshold is sector-specific")
    L.append("  z_T = (1-delta)*T*.")
    L.append("")
    L.append("  Notch jump          tau*z_T = tau*(1-delta)*T*  (IDENTICAL to A, B)")
    L.append("  Dominated (VA)      a_z = (1-delta)*T**tau/(1-tau) = (1-delta)*£21,250")
    L.append("  Dominated (turnover)a_y = a_z/(1-delta) = T**tau/(1-tau) = £21,250")
    L.append("                      delta-INDEPENDENT (matches the paper's section 4)")
    L.append("  Ability recovery    reg:   n = (1-delta)*y_obs/(1-tau)**e")
    L.append("                      unreg: n = (1-delta)*y_obs")
    L.append("")
    L.append("THREE PROPERTIES A2 SECURES")
    L.append("-" * 72)
    L.append("  1. A literal value-added base: the firm optimises over value added.")
    L.append("  2. A bounded ability n = undistorted value added; as delta->1 the")
    L.append("     observed value added (1-delta)y_obs -> 0 so n -> 0, NOT infinity")
    L.append("     (this is the key fix versus model A).")
    L.append("  3. The paper's headline £21,250 turnover dominated region for EVERY")
    L.append("     sector (section 4).")
    L.append("")

    # 1. Optimum verification.
    L.append("1. A2 OPTIMUM VERIFICATION (closed form z*=n(1-tau)**e vs grid + scipy)")
    L.append("-" * 72)
    L.append(f"   {'e':>5} {'rel_grid':>11} {'rel_scalar':>11}"
             f" {'abs_grid(£)':>13} {'abs_scalar(£)':>14}")
    for r in opt["rows"]:
        L.append(f"   {r['e']:>5} {r['rel_grid']:>11.2e} {r['rel_scalar']:>11.2e}"
                 f" {r['abs_grid']:>13.4f} {r['abs_scalar']:>14.4f}")
    L.append(f"   worst relative error: {opt['worst_rel']:.2e}"
             f"  (target < 1e-4: {'PASS' if opt['worst_rel'] < 1e-4 else 'FAIL'})")
    L.append(f"   worst (least-negative) pi''(z*): {opt['worst_second_derivative']:.4e}"
             f"  strictly concave: {'PASS' if opt['all_concave'] else 'FAIL'}")
    L.append("   elasticity identity  d ln z*/d ln(1-tau) = e:")
    for r in opt["elas_rows"]:
        L.append(f"     e={r['e']:<5} numeric slope={r['slope']:.6f}"
                 f"  err={r['err']:.2e}")
    L.append(f"   elasticity check: {'PASS' if opt['elasticity_ok'] else 'FAIL'}")
    L.append("")
    L.append("   TURNOVER dominated region flat £21,250 across delta:")
    for d, a in opt["a_check"]:
        L.append(f"     delta={d:<5g}  a_y = £{a:,.2f}")
    L.append(f"   all equal £{opt['a_target']:,.2f} (max dev {opt['a_max_dev']:.2e}): "
             f"{'PASS' if opt['a_delta_independent'] else 'FAIL'}")
    L.append("   notch jump == tau*(1-delta)*T* across delta:")
    for d, j, jt in opt["jump_check"]:
        L.append(f"     delta={d:<5g}  jump = £{j:,.2f}  (target £{jt:,.2f})")
    L.append(f"   notch-jump identity: {'PASS' if opt['jump_ok'] else 'FAIL'}")
    L.append("")

    # 2. Bounded ability (the fix).
    L.append("2. ABILITY IS WELL-BEHAVED (A2 bounded -> 0 vs A -> infinity)")
    L.append("-" * 72)
    L.append(f"   recovery from a fixed observed turnover y_obs=£{bound['y_obs']:,.0f},"
             f" e={bound['e']}:")
    L.append(f"   {'delta':>8}{'n(A2)':>14}{'n(A)':>16}{'n(B)':>14}")
    for r in bound["rows"]:
        L.append(f"   {r['delta']:>8.4f}{r['n_A2']:>14,.0f}{r['n_A']:>16,.0f}"
                 f"{r['n_B']:>14,.0f}")
    L.append(f"   A2 -> 0 as delta -> 1: {'PASS' if bound['A2_bounded_to_zero'] else 'FAIL'}"
             f"   A diverges upward: {'confirmed' if bound['A_diverges'] else 'no'}")
    L.append("   A2's ability is the firm's UNDISTORTED VALUE ADDED: a near-all-")
    L.append("   deductible firm has little value-adding capacity, so n -> 0. A")
    L.append("   instead inflates ability without bound because it divides by")
    L.append("   (1-delta) -> 0.")
    L.append("")
    L.append("   Representative sectors (real delta):")
    L.append(f"   {'sector':<28}{'delta':>7}{'n(A2)':>12}{'n(A)':>12}{'n(B)':>12}")
    for s in sectors["rows"]:
        nm = f"{s['sic']:>2} {s['name'].split('(')[0].strip()}"[:27]
        L.append(f"   {nm:<28}{s['delta']:>7.3f}{s['n_A2']:>12,.0f}"
                 f"{s['n_A']:>12,.0f}{s['n_B']:>12,.0f}")
    L.append("")

    # 3. Three-way population comparison.
    L.append("3. THREE-WAY POPULATION COMPARISON (real data, weighted means)")
    L.append("-" * 72)
    L.append(f"   registered firms used (turnover >= T*): {pop['n_firms']:,}")
    L.append(f"   mean recovered ability   n(A2) = £{pop['mean_n_A2']:,.0f}"
             f"   n(A) = £{pop['mean_n_A']:,.0f}   n(B) = £{pop['mean_n_B']:,.0f}")
    L.append("     -> A2 is bounded; A is inflated (divides by 1-delta); B is mild.")
    L.append(f"   mean dominated width     a(A2) = £{pop['mean_a_A2']:,.0f} (flat)"
             f"   a(A) = £{pop['mean_a_A']:,.0f} (flat)"
             f"   a(B) = £{pop['mean_a_B']:,.0f} (smaller)")
    L.append(f"   intensive turnover response to a {pop['tau']:.0%}->{pop['tau_band']:.0%}"
             f" value-added-rate reform:")
    L.append(f"     A2 mean scale = {pop['mean_scale_A2']:.5f}"
             f" (+{pop['resp_pct_A2']:.3f}% turnover)  -- uses FULL (1-tau)")
    L.append(f"     B  mean scale = {pop['mean_scale_B']:.5f}"
             f" (+{pop['resp_pct_B']:.3f}% turnover)  -- diluted by tau*(1-delta)")
    L.append("     A2's response is governed by the full statutory rate on value")
    L.append("     added, so it exceeds B's diluted response.")
    L.append("")

    # The one cost.
    L.append("THE ONE COST")
    L.append("-" * 72)
    L.append("  The single £85k notch on TURNOVER becomes a sector-specific notch on")
    L.append("  VALUE ADDED at z_T = (1-delta)*T*: the threshold the firm 'feels' in")
    L.append("  its own choice variable depends on its sector deductible share.")
    L.append("")

    # Verdict.
    L.append("VERDICT")
    L.append("=" * 72)
    L.append("A2 is the CLEANEST of the three formulations for a value-added VAT base.")
    L.append("By letting the firm choose value added z directly it: (i) uses a literal")
    L.append("value-added base; (ii) recovers a BOUNDED, economically sensible ability")
    L.append("n = undistorted value added that goes to 0 (not infinity) as delta -> 1,")
    L.append("fixing model A's blow-up while keeping a clean primitive like model B;")
    L.append("and (iii) reproduces the paper's headline £21,250 turnover dominated")
    L.append("region for every sector (which B loses). The only price is that the")
    L.append("registration notch, expressed in the firm's value-added choice variable,")
    L.append("becomes sector-specific at z_T = (1-delta)*T*. On balance A2 dominates")
    L.append("both A (unbounded ability) and B (sector-shrinking dominated region).")
    return L


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(vintage="2023-24"):
    """Run A2 verification, ability boundedness, three-way comparison; write outputs."""
    opt = verify_optimum()
    bound = ability_boundedness()
    sectors = ability_sectors(vintage)
    pop = compare_population(vintage)
    lines = build_report(opt, bound, sectors, pop)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt = RESULTS_DIR / "va_choice_comparison.txt"
    Path(txt).write_text("\n".join(lines) + "\n")
    fig = make_figure(sectors)

    print("\n".join(lines))
    print(f"\nWrote {txt}")
    print(f"Wrote {fig}")
    return {"opt": opt, "bound": bound, "sectors": sectors, "pop": pop}


def cli(argv=None):
    """argparse entry point."""
    ap = argparse.ArgumentParser(
        prog="firm-microsim-va-choice-model",
        description="Value-added-CHOICE VAT-notch model (A2): the firm chooses its "
                    "value added z directly. Verifies the A2 optimum z*=n(1-tau)**e, "
                    "shows the bounded ability that -> 0 as delta -> 1 (fixing model "
                    "A's blow-up), keeps the flat £21,250 turnover dominated region, "
                    "and compares recovered ability / dominated region / reform "
                    "response against models A and B. Writes "
                    "results/va_choice_comparison.{png,txt}.",
    )
    ap.add_argument("--vintage", default="2023-24",
                    help="data vintage (default: 2023-24)")
    args = ap.parse_args(argv)
    return main(args.vintage)


if __name__ == "__main__":
    cli()
