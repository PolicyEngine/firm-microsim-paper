"""Explicit value-added reformulation of the iso-elastic VAT-notch firm problem.

The dynamic simulator (:mod:`firm_microsim.dynamic.model`) levies the registered
firm's net VAT *implicitly*, as a firm-specific wedge ``tau0`` applied to the
WHOLE turnover: registered net revenue is ``(1 - tau0) * y``. ``tau0`` is the
share of turnover actually remitted (output VAT minus input credits, ~3% on
average), but where it comes from is left opaque.

This module makes the mechanism EXPLICIT. VAT is a tax on **value added**, not on
turnover. Let the statutory rate be ``tau`` (= 0.20) and let each firm carry a
sector **deductible-input share** ``delta in [0, 1)`` (deductible expenses as a
share of turnover), so the **value-added share** is ``(1 - delta)``. A registered
firm remits net VAT on value added only::

    net VAT = tau * (y - d) = tau * (1 - delta) * y,        d = delta * y.

Hence registered profit is

    Unregistered (y <  T*):  pi = y                       - C(y; n, e)
    Registered   (y >= T*):  pi = (1 - tau*(1 - delta))*y - C(y; n, e)

with the SAME iso-elastic production cost as the rest of the simulator,
``C(y; n, e) = (n / (1 + 1/e)) * (y / n) ** (1 + 1/e)`` and marginal cost
``(y / n) ** (1/e)``.

DERIVED OBJECTS (all written out and verified numerically below)
----------------------------------------------------------------
* Effective registered wedge      tau_eff(delta) = tau * (1 - delta)
                                                  = tau x value-added share.
* Interior registered optimum     y*(n, delta, e) = n * (1 - tau_eff(delta)) ** e
                                                   = n * (1 - tau*(1 - delta))**e.
* Notch jump at T*                 tau_eff(delta) * T*  (drop in net revenue when
                                   registration first bites on a firm at T*).
* Dominated-region width          a(delta) = T* * tau_eff/(1 - tau_eff)
                                   (Kleven-Waseem, with the EFFECTIVE rate).
* Marginal buncher n_H(delta, e)   ability indifferent between bunching at T* and
                                   registering at y*; an indifference solve.

THE CENTRAL EQUIVALENCE
-----------------------
This explicit ``(delta, tau)`` model is IDENTICAL to the existing
``(1 - tau0)*y`` simulator whenever ``tau0 = tau*(1 - delta)``: the registered
net rate, the interior optimum, the notch jump, the dominated region and the
marginal buncher all coincide to machine precision. Making deductibles explicit
adds no new behaviour; it RE-LABELS the opaque wedge as ``tau x (value-added
share)`` and lets the wedge — and therefore the whole notch geometry — VARY by
sector through the data-grounded ``delta``. Firms in high-deductible /
low-value-added sectors (agriculture, food) face a much smaller effective notch
than firms in high-value-added services.

Run ``python -m firm_microsim.dynamic.deductible_model`` (or call :func:`main`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

from firm_microsim.config import RESULTS_DIR, SYNTHETIC_DATA_DIR
from firm_microsim.dynamic import sector_vat
from firm_microsim.dynamic.model import (
    ELASTICITIES,
    E_HEADLINE,
    T_STAR,
    TAU_MAX,
    dominated_region_width,
    iso_cost,
    iso_profit,
    load_reform_data,
    marginal_buncher_iso,
)

# Statutory VAT rate (the rate levied on value added once registered).
TAU = TAU_MAX  # 0.20

# delta is a share in [0, 1); clip just below 1 so the value-added share is > 0.
DELTA_MAX = 1.0 - 1e-9

# Wedges below this are treated as "no effective notch" (no marginal buncher).
WEDGE_FLOOR = 1e-6

# Representative SIC divisions: a high-deductible / low-value-added end
# (agriculture, food) through to a high-value-added services end. Chosen to span
# the empirical delta spread so the sector-specific notch geometry is visible.
REPRESENTATIVE_SECTORS = (
    (1, "Agriculture (crop & animal production)"),
    (41, "Construction of buildings"),
    (56, "Food & beverage service"),
    (62, "Computer programming & IT services"),
)

# House style (matches firm_microsim.dynamic.verify_optimum / figures).
PALETTE = ["#326b77", "#122740", "#1b485e", "#568b87", "#80ae9a", "#b5d1ae"]
PRIMARY = "#326b77"
ACCENT = "#d62728"
LABEL_SIZE = 15
TICK_SIZE = 13


# ---------------------------------------------------------------------------
# Core closed forms of the explicit value-added model
# ---------------------------------------------------------------------------
def effective_wedge(delta, tau=TAU):
    """Effective registered VAT wedge ``tau_eff = tau*(1 - delta)`` (= tau x VA share)."""
    return float(tau) * (1.0 - np.asarray(delta, dtype=float))


def optimal_turnover(n, e, delta, tau=TAU):
    """Interior registered optimum ``y* = n*(1 - tau*(1 - delta))**e`` (£).

    This is the FLAT-rate iso-elastic optimum with the effective rate
    ``tau_eff(delta) = tau*(1 - delta)`` in place of a whole-turnover rate.
    """
    return np.asarray(n, dtype=float) * (1.0 - effective_wedge(delta, tau)) ** e


def deductible_profit(y, n, e, delta, tau=TAU, registered=None, T=T_STAR):
    """Explicit value-added profit ``pi(y; n, e, delta)`` (£).

    VAT is levied on value added = turnover - deductible expenses, so a
    REGISTERED firm's net revenue is ``(1 - tau*(1 - delta))*y`` and an
    UNREGISTERED firm keeps the whole turnover ``y``. Production cost is the
    shared iso-elastic ``C(y; n, e)``.

    ``registered`` forces the regime; if ``None`` it is inferred from ``y >= T``
    elementwise (the hard-notch convention).
    """
    y = np.asarray(y, dtype=float)
    net_reg = 1.0 - effective_wedge(delta, tau)
    if registered is None:
        reg = y >= T
    else:
        reg = np.broadcast_to(np.asarray(registered, dtype=bool), y.shape)
    net = np.where(reg, net_reg, 1.0)
    return net * y - iso_cost(y, n, e)


def dominated_width(delta, tau=TAU, T=T_STAR):
    """Kleven-Waseem dominated-region width at the EFFECTIVE rate (£).

    ``a(delta) = T* * tau_eff/(1 - tau_eff)`` with ``tau_eff = tau*(1 - delta)``.
    """
    te = effective_wedge(delta, tau)
    return float(T) * te / (1.0 - te)


def notch_jump(delta, tau=TAU, T=T_STAR):
    """Discontinuous drop in net revenue at T* when registration bites: tau_eff*T*."""
    return effective_wedge(delta, tau) * float(T)


def marginal_buncher_deductible(e, delta, tau=TAU, T=T_STAR):
    """Marginal buncher ``n_H`` under the explicit value-added notch (£).

    Solves the indifference between bunching just below ``T*`` (unregistered) and
    registering at the post-notch interior optimum ``y1 = n*(1 - tau_eff)**e``,
    with ``tau_eff = tau*(1 - delta)``. Mirrors
    :meth:`firm_microsim.notch.model.NotchModel.marginal_buncher` /
    :func:`firm_microsim.dynamic.model.marginal_buncher_iso` exactly, but with the
    effective rate. Returns ``(n_H, dy_star)`` with ``dy_star = n_H - T*``.

    When ``tau_eff`` is below ``WEDGE_FLOOR`` (a near-zero-VAT, all-deductible
    sector) there is effectively no notch and no buncher: returns ``(nan, nan)``.
    """
    te = effective_wedge(delta, tau)
    if te < WEDGE_FLOOR:
        return float("nan"), float("nan")

    def gap(n):
        u_bunch = iso_profit(min(n, T), n, e, net=1.0)
        y1 = n * (1.0 - te) ** e
        u_tax = iso_profit(y1, n, e, net=1.0 - te)
        return float(u_bunch - u_tax)

    lo, hi = T * (1.0 + 1e-6), T * 5.0
    while gap(lo) * gap(hi) > 0 and hi < T * 50:
        hi *= 1.5
    n_H = brentq(gap, lo, hi)
    return float(n_H), float(n_H - T)


# ---------------------------------------------------------------------------
# Verification: closed form vs numerical argmax (grid + scipy)
# ---------------------------------------------------------------------------
def _grid_argmax(n, e, delta, tau=TAU, *, npts=400_001, lo_frac=0.65, hi_frac=1.03):
    """Maximise the registered profit over a fine y-grid in [lo*n, hi*n]."""
    grid = np.linspace(lo_frac * n, hi_frac * n, npts)
    prof = deductible_profit(grid, n, e, delta, tau, registered=True)
    return float(grid[int(np.argmax(prof))])


def _scalar_argmax(n, e, delta, tau=TAU):
    """Maximise the registered profit with bounded ``minimize_scalar``."""
    res = minimize_scalar(
        lambda y: -float(deductible_profit(y, n, e, delta, tau, registered=True)),
        bounds=(1e-6, 2.0 * n),
        method="bounded",
        options={"xatol": 1e-8},
    )
    return float(res.x)


def verify_optimum(
    *,
    n_grid=None,
    deltas=(0.0, 0.2, 0.5, 0.85, 0.95),
    elasticities=ELASTICITIES,
    tau=TAU,
):
    """Closed-form optimum vs fine-grid AND scipy argmax over a (n, delta, e) grid.

    Also confirms strict concavity (pi'' < 0 at the optimum) and the two limits:
    delta->1 => wedge->0 and y*->n; delta->0 => wedge->tau (harsh whole-turnover
    notch). Returns a results dict.
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
                y_cf = float(optimal_turnover(n, e, delta, tau))
                y_g = _grid_argmax(n, e, delta, tau)
                y_s = _scalar_argmax(n, e, delta, tau)
                abs_g = max(abs_g, abs(y_g - y_cf))
                abs_s = max(abs_s, abs(y_s - y_cf))
                rel_g = max(rel_g, abs(y_g - y_cf) / y_cf)
                rel_s = max(rel_s, abs(y_s - y_cf) / y_cf)
                # Central-difference pi'' at the optimum.
                h = 1e-4 * y_cf
                f_p = float(deductible_profit(y_cf + h, n, e, delta, tau, registered=True))
                f_0 = float(deductible_profit(y_cf, n, e, delta, tau, registered=True))
                f_m = float(deductible_profit(y_cf - h, n, e, delta, tau, registered=True))
                worst_pp = max(worst_pp, (f_p - 2.0 * f_0 + f_m) / (h * h))
            worst_rel = max(worst_rel, rel_g, rel_s)
            worst_abs = max(worst_abs, abs_g, abs_s)
            rows.append({"e": e, "delta": delta, "rel_grid": rel_g,
                         "rel_scalar": rel_s, "abs_grid": abs_g, "abs_scalar": abs_s})

    # Limit delta -> 1: wedge -> 0, y* -> n.
    n_ref = 150_000.0
    lim_rows = []
    for d in (0.9, 0.99, 0.999, DELTA_MAX):
        w = float(effective_wedge(d, tau))
        rel_gap = max(
            abs(float(optimal_turnover(n, E_HEADLINE, d, tau)) - n) / n for n in n_grid
        )
        lim_rows.append({"delta": d, "wedge": w, "max_rel_y_minus_n": rel_gap})

    # Limit delta -> 0: wedge -> tau (harsh whole-turnover notch).
    wedge_at_0 = float(effective_wedge(0.0, tau))
    y_at_0 = float(optimal_turnover(n_ref, E_HEADLINE, 0.0, tau))
    y_whole = n_ref * (1.0 - tau) ** E_HEADLINE  # statutory whole-turnover optimum

    return {
        "rows": rows,
        "worst_rel": worst_rel,
        "worst_abs": worst_abs,
        "worst_second_derivative": float(worst_pp),
        "all_concave": bool(worst_pp < 0),
        "delta_to_1": lim_rows,
        "wedge_at_delta0": wedge_at_0,
        "wedge_at_delta0_eq_tau": abs(wedge_at_0 - tau) < 1e-12,
        "y_at_delta0": y_at_0,
        "y_whole_turnover": y_whole,
        "delta0_matches_whole": abs(y_at_0 - y_whole) < 1e-6,
    }


# ---------------------------------------------------------------------------
# Equivalence with the existing (1 - tau0)*y simulator (tau0 = tau*(1-delta))
# ---------------------------------------------------------------------------
def verify_equivalence(
    *, deltas=(0.0, 0.2, 0.5, 0.7, 0.85, 0.95), e=E_HEADLINE, tau=TAU, n=150_000.0
):
    """Show the explicit (delta, tau) model == the (1 - tau0)*y model, tau0=tau(1-delta).

    Compares, for each ``delta``: the registered optimum, the dominated-region
    width, the registered profit at a test ``y``, and the marginal buncher,
    against the existing ``model.py`` closed forms evaluated with
    ``tau0 = tau*(1 - delta)``. Returns the worst discrepancies and a PASS flag.
    """
    max_dy = max_da = max_dpi = max_dnH = 0.0
    rows = []
    for delta in deltas:
        tau0 = float(tau * (1.0 - delta))  # the existing simulator's wedge

        # 1. Interior optimum: explicit vs n*(1 - tau0)**e (the tau0 form).
        y_expl = float(optimal_turnover(n, e, delta, tau))
        y_tau0 = float(n * (1.0 - tau0) ** e)
        dy = abs(y_expl - y_tau0)

        # 2. Dominated-region width: explicit vs model.dominated_region_width(tau0).
        a_expl = dominated_width(delta, tau)
        a_tau0 = float(dominated_region_width(T=T_STAR, tau=tau0))
        da = abs(a_expl - a_tau0)

        # 3. Registered profit at a test turnover via model.iso_profit(net=1-tau0).
        y_test = max(y_expl, T_STAR + 1.0)
        pi_expl = float(deductible_profit(y_test, n, e, delta, tau, registered=True))
        pi_tau0 = float(iso_profit(y_test, n, e, net=1.0 - tau0))
        dpi = abs(pi_expl - pi_tau0)

        # 4. Marginal buncher: explicit vs model.marginal_buncher_iso(tau=tau0).
        if effective_wedge(delta, tau) >= WEDGE_FLOOR:
            nH_expl, _ = marginal_buncher_deductible(e, delta, tau)
            nH_tau0, _ = marginal_buncher_iso(e, T=T_STAR, tau=tau0)
            dnH = abs(nH_expl - nH_tau0)
        else:
            nH_expl = nH_tau0 = float("nan")
            dnH = 0.0

        max_dy = max(max_dy, dy)
        max_da = max(max_da, da)
        max_dpi = max(max_dpi, dpi)
        max_dnH = max(max_dnH, dnH)
        rows.append({"delta": delta, "tau0": tau0, "y_expl": y_expl,
                     "y_tau0": y_tau0, "a_expl": a_expl, "a_tau0": a_tau0,
                     "nH_expl": nH_expl, "nH_tau0": nH_tau0,
                     "dy": dy, "da": da, "dpi": dpi, "dnH": dnH})

    tol = 1e-9
    # Profit magnitudes are large; use a relative tolerance there.
    passed = (max_dy < tol and max_da < tol and max_dnH < 1e-6
              and max_dpi < 1e-3)
    return {"rows": rows, "max_dy": max_dy, "max_da": max_da,
            "max_dpi": max_dpi, "max_dnH": max_dnH, "passed": bool(passed)}


# ---------------------------------------------------------------------------
# Sector simulation using REAL data (sector_vat ratios -> implied delta)
# ---------------------------------------------------------------------------
def implied_delta(ratio, tau=TAU):
    """Implied deductible share from an empirical net-VAT-to-turnover ratio.

    The empirical sector ratio IS the effective wedge tau_eff(delta), so
    ``delta = 1 - ratio/tau`` clipped to ``[0, DELTA_MAX]``.
    """
    return np.clip(1.0 - np.asarray(ratio, dtype=float) / tau, 0.0, DELTA_MAX)


def _load_population_with_sic(vintage):
    """Population (£) from ``load_reform_data`` with ``sic_code`` attached in order."""
    df = load_reform_data().reset_index(drop=True)
    sic = pd.read_csv(
        SYNTHETIC_DATA_DIR / f"synthetic_firms_{vintage}.csv",
        usecols=["sic_code"],
        dtype={"sic_code": str},
    )["sic_code"]
    if len(sic) != len(df):
        raise RuntimeError(f"sic_code rows ({len(sic)}) != population ({len(df)})")
    df["sic_code"] = sic.reset_index(drop=True)
    return df


def sector_analysis(vintage="2023-24", *, e=E_HEADLINE, n_sample=150_000.0, tau=TAU):
    """Per-sector deductible geometry from real data + population-weighted means.

    Treats the empirical sector net-VAT-to-turnover ratio as tau_eff and backs out
    the implied delta; reports, for representative sectors, tau_eff, delta, the
    optimal turnover for a sample ability, the notch jump, the dominated-region
    width a(delta), and the marginal buncher. Also returns population-weighted
    mean implied delta and mean tau_eff.
    """
    ratios = sector_vat.sector_vat_ratios(vintage)
    pop_mean_ratio = float(ratios.attrs["population_mean"])

    # Per-sector representative table.
    sectors = []
    for sic, name in REPRESENTATIVE_SECTORS:
        if sic in ratios.index:
            ratio = float(ratios.loc[sic])
        else:
            ratio = pop_mean_ratio
        delta = float(implied_delta(ratio, tau))
        te = float(effective_wedge(delta, tau))
        nH, dy = marginal_buncher_deductible(e, delta, tau)
        sectors.append({
            "sic": sic, "name": name, "ratio": ratio, "delta": delta,
            "tau_eff": te, "y_star": float(optimal_turnover(n_sample, e, delta, tau)),
            "notch_jump": float(notch_jump(delta, tau)),
            "a": float(dominated_width(delta, tau)),
            "n_H": nH, "dy_star": dy,
        })

    # Population-weighted means via the public sector_vat machinery.
    df = _load_population_with_sic(vintage)
    sector_tau_eff = sector_vat.attach_sector_tau0(df, ratios)  # per-firm tau_eff
    sector_delta = implied_delta(sector_tau_eff, tau)
    w = df["weight"].to_numpy(dtype=float)
    t = df["turnover"].to_numpy(dtype=float)
    pos = t > 0
    mean_delta = float(np.sum(sector_delta[pos] * w[pos]) / np.sum(w[pos]))
    mean_tau_eff = float(np.sum(sector_tau_eff[pos] * w[pos]) / np.sum(w[pos]))

    return {
        "e": e, "n_sample": n_sample,
        "sectors": sectors,
        "n_sectors": int(ratios.shape[0]),
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
        "ratio_median": float(ratios.median()),
        "mean_delta": mean_delta,
        "mean_tau_eff": mean_tau_eff,
        "delta_min": float(implied_delta(ratios.max(), tau)),
        "delta_max": float(implied_delta(ratios.min(), tau)),
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


def make_figure(sector, path=None, *, e=E_HEADLINE, n_demo=150_000.0, tau=TAU):
    """Two-panel figure: sector profit curves (A) and a(delta) with sectors (B)."""
    if path is None:
        path = RESULTS_DIR / "deductible_model.png"
    path = Path(path)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Panel A: notch profit curves for representative sectors (varying delta).
    grid = np.linspace(60_000.0, 200_000.0, 1600)
    below = grid < T_STAR
    # Sort sectors by tau_eff so colours run low->high deductible.
    secs = sorted(sector["sectors"], key=lambda s: s["tau_eff"])
    for i, s in enumerate(secs):
        col = PALETTE[i % len(PALETTE)]
        net = np.where(grid >= T_STAR, 1.0 - s["tau_eff"], 1.0)
        prof = net * grid - iso_cost(grid, n_demo, e)
        axL.plot(grid[below] / 1000.0, prof[below] / 1000.0, color=col, lw=2.0)
        axL.plot(grid[~below] / 1000.0, prof[~below] / 1000.0, color=col, lw=2.0,
                 label=f"$\\delta$={s['delta']:.2f}, $\\tau_{{eff}}$={s['tau_eff']:.3f}"
                       f" ({s['name'].split('(')[0].strip()})")
        # Mark the registered optimum.
        y_star = float(optimal_turnover(n_demo, e, s["delta"], tau))
        if y_star >= T_STAR:
            axL.plot(y_star / 1000.0,
                     (((1.0 - s["tau_eff"]) * y_star - iso_cost(y_star, n_demo, e))
                      / 1000.0),
                     "o", color=col, ms=8, zorder=6)
    axL.axvline(T_STAR / 1000.0, color="gray", ls="--", lw=1.4)
    axL.set_xlabel("Turnover $y$ (£k)", fontsize=LABEL_SIZE)
    axL.set_ylabel(f"Profit $\\pi(y;n)$ (£k),  $n$=£{n_demo/1000:.0f}k",
                   fontsize=LABEL_SIZE)
    axL.set_title(f"Sector notch profiles, $e$={e}  (VAT on value added)",
                  fontsize=LABEL_SIZE)
    _style_ax(axL)
    axL.legend(frameon=False, fontsize=TICK_SIZE - 2, loc="lower right")

    # --- Panel B: dominated-region width a(delta), with real sectors marked.
    dd = np.linspace(0.0, 0.98, 400)
    a = np.array([dominated_width(d, tau) for d in dd])
    axR.plot(dd, a / 1000.0, color=PRIMARY, lw=2.4,
             label="$a(\\delta)=T^*\\,\\tau_{eff}/(1-\\tau_{eff})$")
    for i, s in enumerate(secs):
        axR.plot(s["delta"], s["a"] / 1000.0, "o", color=ACCENT, ms=9, zorder=6)
        axR.annotate(s["name"].split("(")[0].strip(),
                     (s["delta"], s["a"] / 1000.0),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=TICK_SIZE - 3, color="black")
    axR.set_xlabel("Deductible share $\\delta$ (value-added share $=1-\\delta$)",
                   fontsize=LABEL_SIZE)
    axR.set_ylabel("Dominated-region width $a(\\delta)$ (£k)", fontsize=LABEL_SIZE)
    axR.set_title("Notch shrinks as deductibles rise", fontsize=LABEL_SIZE)
    _style_ax(axR)
    axR.legend(frameon=False, fontsize=TICK_SIZE, loc="upper right")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def build_report(opt, equiv, sector):
    """Assemble the human-readable report (list of lines)."""
    L = []
    L.append("Explicit value-added (deductible) reformulation of the VAT notch")
    L.append("=" * 72)
    L.append("MODEL")
    L.append("-" * 72)
    L.append("  Statutory rate tau = 0.20; sector deductible share delta in [0,1).")
    L.append("  Value added = turnover - deductibles = (1 - delta)*y.")
    L.append("  Unregistered (y <  T*):  pi = y                       - C(y;n,e)")
    L.append("  Registered   (y >= T*):  pi = (1 - tau*(1-delta))*y   - C(y;n,e)")
    L.append("  C(y;n,e) = (n/(1+1/e))*(y/n)**(1+1/e),  c'(y) = (y/n)**(1/e).")
    L.append("")
    L.append("  Effective wedge      tau_eff(delta) = tau*(1-delta)")
    L.append("  Registered optimum   y* = n*(1 - tau*(1-delta))**e")
    L.append("  Notch jump at T*     tau_eff*T*")
    L.append("  Dominated width      a(delta) = T**tau_eff/(1-tau_eff)")
    L.append("")

    # 1. Optimum verification.
    L.append("1. OPTIMUM VERIFICATION (closed form vs grid AND scipy argmax)")
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
    L.append("   LIMIT delta -> 1 (all deductible, zero value added): wedge->0, y*->n")
    for r in opt["delta_to_1"]:
        L.append(f"     delta={r['delta']:<12g} wedge={r['wedge']:.3e}"
                 f"  max|y*-n|/n={r['max_rel_y_minus_n']:.3e}")
    L.append(f"   LIMIT delta -> 0 (no deductibles): wedge={opt['wedge_at_delta0']:.4f}"
             f" == tau: {'PASS' if opt['wedge_at_delta0_eq_tau'] else 'FAIL'};"
             f" recovers whole-turnover y*: "
             f"{'PASS' if opt['delta0_matches_whole'] else 'FAIL'}")
    L.append("")

    # 2. Equivalence.
    L.append("2. EQUIVALENCE WITH THE (1 - tau0)*y SIMULATOR  [tau0 = tau*(1-delta)]")
    L.append("-" * 72)
    L.append(f"   {'delta':>7} {'tau0':>8} {'y*_expl':>12} {'y*_tau0':>12}"
             f" {'a_expl':>10} {'a_tau0':>10} {'nH_expl':>11} {'nH_tau0':>11}")
    for r in equiv["rows"]:
        nHe = "n/a" if np.isnan(r["nH_expl"]) else f"{r['nH_expl']:,.0f}"
        nHt = "n/a" if np.isnan(r["nH_tau0"]) else f"{r['nH_tau0']:,.0f}"
        L.append(f"   {r['delta']:>7.3f} {r['tau0']:>8.4f} {r['y_expl']:>12,.2f}"
                 f" {r['y_tau0']:>12,.2f} {r['a_expl']:>10,.1f} {r['a_tau0']:>10,.1f}"
                 f" {nHe:>11} {nHt:>11}")
    L.append(f"   max |dy*|={equiv['max_dy']:.2e}  max |da|={equiv['max_da']:.2e}"
             f"  max |dpi|={equiv['max_dpi']:.2e}  max |dnH|={equiv['max_dnH']:.2e}")
    L.append(f"   EQUIVALENCE: {'PASS' if equiv['passed'] else 'FAIL'}"
             "  (explicit (delta,tau) model == implicit (1-tau0)*y model)")
    L.append("")

    # 3. Sector simulation.
    L.append("3. SECTOR SIMULATION FROM REAL DATA (2023-24)")
    L.append("-" * 72)
    L.append(f"   {'sector':<34}{'tau_eff':>8}{'delta':>7}"
             f"{'y*(£)':>11}{'jump(£)':>10}{'a(£)':>10}{'n_H(£)':>11}")
    for s in sector["sectors"]:
        nH = "no notch" if np.isnan(s["n_H"]) else f"{s['n_H']:,.0f}"
        nm = f"{s['sic']:>2} {s['name']}"[:33]
        L.append(f"   {nm:<34}{s['tau_eff']:>8.4f}{s['delta']:>7.3f}"
                 f"{s['y_star']:>11,.0f}{s['notch_jump']:>10,.0f}"
                 f"{s['a']:>10,.0f}{nH:>11}")
    L.append(f"   (optimum/jump/buncher at e={sector['e']}, sample ability "
             f"n=£{sector['n_sample']:,.0f})")
    L.append("")
    L.append(f"   sector ratio (=tau_eff) spread: min {sector['ratio_min']:.4f},"
             f" median {sector['ratio_median']:.4f}, max {sector['ratio_max']:.4f}"
             f"  (n={sector['n_sectors']} sectors)")
    L.append(f"   implied delta spread          : {sector['delta_min']:.3f}"
             f" .. {sector['delta_max']:.3f}")
    L.append(f"   population-weighted mean delta = {sector['mean_delta']:.4f},"
             f"  mean tau_eff = {sector['mean_tau_eff']:.4f}")
    L.append("")

    # Verdict.
    sound = (opt["worst_rel"] < 1e-4 and opt["all_concave"]
             and opt["wedge_at_delta0_eq_tau"] and opt["delta0_matches_whole"]
             and equiv["passed"])
    L.append("VERDICT")
    L.append("=" * 72)
    if sound:
        L.append("The explicit value-added reformulation is SOUND. Levying tau on")
        L.append("value added (turnover - deductibles) instead of on the whole")
        L.append("turnover yields a registered net rate (1 - tau*(1-delta)) whose")
        L.append("iso-elastic optimum y* = n*(1 - tau*(1-delta))**e is recovered by the")
        L.append("numerical optimisers to better than 1e-4 relative error, with a")
        L.append("single strictly-concave interior maximum. It is EXACTLY EQUIVALENT to")
        L.append("the existing (1 - tau0)*y simulator with tau0 = tau*(1-delta): the")
        L.append("optimum, the dominated-region width, the registered profit and the")
        L.append("marginal buncher all coincide to machine precision (PASS above).")
        L.append("")
        L.append("WHAT IT CHANGES relative to the implicit form: the wedge is no longer")
        L.append("an opaque ~3% number but tau x (value-added share), grounded in the")
        L.append("sector deductible share delta. This makes the notch and the dominated")
        L.append("region SECTOR-SPECIFIC. High-deductible / low-value-added sectors")
        L.append("(agriculture, food) have delta near 1, tau_eff near 0, a near-zero")
        L.append("notch jump and a vanishing dominated region -- effectively no notch.")
        L.append("High-value-added services (delta near 0) face tau_eff near 0.20, the")
        L.append("full Kleven-Waseem £21,250 dominated region and the harsh notch. The")
        L.append("implicit uniform wedge HIDES exactly this heterogeneity.")
    else:
        L.append("One or more checks FAILED -- see the per-section PASS/FAIL flags.")
    L.append("")
    L.append("CAVEATS")
    L.append("-" * 72)
    L.append("* delta is identified from sector aggregates via tau_eff = tau*(1-delta);")
    L.append("  net-repayment / zero-rated sectors give tau_eff~0 (delta clipped to <1)")
    L.append("  and high-VA sectors hit the data cap, so delta is clipped to [0,1).")
    L.append("* e is a calibration input, not identified from the synthetic data; all")
    L.append("  optima/bunchers are CONDITIONAL on the assumed e.")
    return L


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main(vintage="2023-24"):
    """Run optimum verification, equivalence, sector sim; write figure + report."""
    opt = verify_optimum()
    equiv = verify_equivalence()
    sector = sector_analysis(vintage)
    lines = build_report(opt, equiv, sector)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    txt = RESULTS_DIR / "deductible_model.txt"
    Path(txt).write_text("\n".join(lines) + "\n")
    fig = make_figure(sector)

    print("\n".join(lines))
    print(f"\nWrote {txt}")
    print(f"Wrote {fig}")
    return {"opt": opt, "equiv": equiv, "sector": sector}


def cli(argv=None):
    """argparse entry point."""
    ap = argparse.ArgumentParser(
        prog="firm-microsim-deductible-model",
        description="Explicit value-added (deductible) reformulation of the "
                    "iso-elastic VAT-notch firm problem: simulate the optima, "
                    "verify them, and demonstrate equivalence to the implicit "
                    "(1 - tau0)*y simulator with tau0 = tau*(1 - delta). Writes "
                    "results/deductible_model.{png,txt}.",
    )
    ap.add_argument("--vintage", default="2023-24",
                    help="data vintage (default: 2023-24)")
    args = ap.parse_args(argv)
    return main(args.vintage)


if __name__ == "__main__":
    cli()
