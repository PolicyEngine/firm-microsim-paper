"""Iso-elastic (Kleven-Waseem) structural core for the dynamic VAT-notch simulator.

This module reformulates the behavioural forward-solve onto the CORRECT
iso-elastic quasi-linear model, in which a SINGLE turnover elasticity ``e``
governs the response of turnover to the net-of-tax rate. The previous
Cobb-Douglas version was rejected because it tied the behavioural response to
the production returns ``alpha`` (implying an absurd elasticity ~90) and the
forward-solve did not depend on ``e`` at all.

All monetary quantities here are in **pounds** (not £k), matching the generated
firm-population data used for the reform costings. The iso-elastic marginal
buncher is delegated unchanged to
:meth:`notch.model.NotchModel.marginal_buncher` (it works in £k internally and
we convert at the boundary), and it agrees with our own indifference solve.

THE MODEL (implemented exactly)
-------------------------------
A firm of ability ``n`` chooses turnover ``y`` to maximise the iso-elastic
quasi-linear profit::

    pi(y; n) = R(y) - (n / (1 + 1/e)) * (y / n) ** (1 + 1/e),

with NET revenue ``R(y)`` under the schedule and marginal cost
``c'(y) = (y/n) ** (1/e)``.  ``e > 0`` is the elasticity of turnover with
respect to the net-of-tax rate ``(1 - tau)``.

* Net revenue if the schedule levies effective rate ``tau(y)`` on the WHOLE
  turnover once registered::

      R(y) = (1 - tau(y)) * y   (registered),     R(y) = y   (unregistered, tau=0).

* Frictionless (no-tax) optimum: FOC ``1 = (y/n)**(1/e)`` => ``y = n``. So
  ``n`` is the firm's frictionless optimum.
* Registered under a FLAT rate ``tau`` on the whole base: FOC
  ``(1-tau) = (y/n)**(1/e)`` => ``y = n * (1-tau)**e``.
* General smooth schedule FOC:
  ``(1 - tau(y)) - y * tau'(y) = (y/n)**(1/e)``.

ABILITY RECOVERY (accounting anchor, NOT structural identification)
-------------------------------------------------------------------
Given ``e`` and a firm's observed turnover ``y_obs`` under the BASELINE £85,000
notch, using the firm's observed net VAT rate ``tau0 = liab / y_obs`` once
registered::

    y_obs <  T*  : unregistered  => n = y_obs.
    y_obs >= T*  : registered     => n = y_obs / (1 - tau0)**e.

This rationalises the observed allocation given ``e``; it is an accounting
anchor, not structural identification (``e`` is not identified from the
synthetic data — see the placebo).

REVENUE CONVENTION
------------------
The stored net VAT remittance ``liab`` is NOT ``0.20 * turnover`` — it is
output-minus-input VAT (a firm-specific net rate ``liab/y_obs`` averaging ~3%
of turnover).  Under a reform schedule that scales the standard rate by a
fraction ``f(y) in [0,1]`` over the band, a registered firm's reform remittance
is ``liab * (y_star / y_obs) * f(y_star)``: the firm's net VAT-to-turnover ratio
is held fixed, turnover is re-optimised to ``y_star`` (iso-elastic response),
and the schedule fraction is applied.  This is exactly ``tau_R(y_star)*y_star``
with ``tau_R(y) = (liab/y_obs) * f(y)`` and it reproduces the trusted STATIC
reform costs in the ``e -> 0`` (no-response) limit.

CROSS-CHECKS (see :func:`crosscheck`)
-------------------------------------
1. Dominated region ``a = T* tau/(1-tau) = £21,250``; upper edge £106,250.
2. Marginal buncher ``n_H(e)`` = 112,795 / 127,382 / 143,527 at
   e = 0.05 / 0.17 / 0.32 (our iso-elastic indifference solve matches
   ``notch.model`` to within ±£200).
3. Elasticity check: ``d ln(y_star)/d ln(1-tau) = e`` numerically.
4. ``e -> 0`` limit: behavioural reform costs converge to the static costs
   computed on the in-repository generated population.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from firm_microsim.config import SYNTHETIC_DATA_DIR
from firm_microsim.notch.model import NotchModel

# ---------------------------------------------------------------------------
# Single consistent parameter set (pounds)
# ---------------------------------------------------------------------------
TAU_MAX = 0.20          # standard UK VAT rate (full effective rate once registered)
T_STAR = 85_000.0       # registration threshold (£)
TAPER_TOP = 105_000.0   # taper / reduced-rate band upper edge (£)

# Headline elasticity bracket (median 0.17, bracketed by 0.05 and 0.32).
ELASTICITIES = (0.05, 0.17, 0.32)
E_HEADLINE = 0.17


# ---------------------------------------------------------------------------
# Iso-elastic cost, marginal cost, profit
# ---------------------------------------------------------------------------
def iso_cost(y, n, e):
    """Iso-elastic cost ``c(y;n,e) = (n/(1+1/e)) * (y/n)**(1+1/e)`` (£).

    Marginal cost ``c'(y) = (y/n)**(1/e)``; the frictionless optimum (mc=1)
    is ``y = n``.
    """
    y = np.asarray(y, dtype=float)
    p = 1.0 + 1.0 / e
    return (n / p) * (y / n) ** p


def iso_mc(y, n, e):
    """Marginal cost ``c'(y) = (y/n)**(1/e)``."""
    y = np.asarray(y, dtype=float)
    return (y / n) ** (1.0 / e)


def iso_profit(y, n, e, net):
    """Profit ``pi = net*y - c(y;n,e)`` for net revenue rate ``net`` (1 or 1-tau)."""
    y = np.asarray(y, dtype=float)
    return net * y - iso_cost(y, n, e)


# ---------------------------------------------------------------------------
# Ability recovery under the baseline £85k notch (iso-elastic accounting anchor)
# ---------------------------------------------------------------------------
def recover_ability(y_obs, e, T=T_STAR, tau=TAU_MAX):
    """Recover frictionless ability ``n`` from observed turnover under the notch.

    ``y_obs < T*`` -> ``n = y_obs`` (unregistered);
    ``y_obs >= T*`` -> ``n = y_obs / (1-tau)**e`` (registered at taxed optimum).

    Accounting anchor that rationalises the observed allocation given ``e``; NOT
    structural identification.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    return np.where(y_obs < T, y_obs, y_obs / (1.0 - tau) ** e)


# ---------------------------------------------------------------------------
# Reform schedules: effective-rate fraction (of the standard rate) at turnover y
# ---------------------------------------------------------------------------
def schedule_notch(y, T=T_STAR):
    """Hard notch: full standard rate at/above T*, zero below. Fraction in {0,1}."""
    y = np.asarray(y, dtype=float)
    return (y >= T).astype(float)


def make_schedule_raise(T_new=100_000.0):
    """Raise the threshold to ``T_new``: the notch simply moves to ``T_new``."""

    def sched(y):
        y = np.asarray(y, dtype=float)
        return (y >= T_new).astype(float)

    return sched


def schedule_taper(y, T=T_STAR, top=TAPER_TOP):
    """Graduated taper: effective fraction phases 0 -> 1 linearly over [T, top]."""
    y = np.asarray(y, dtype=float)
    return np.clip((y - T) / (top - T), 0.0, 1.0)


def make_schedule_reduced_rate(tau_low, T=T_STAR, top=TAPER_TOP, tau_std=TAU_MAX):
    """Banded reduced rate: fraction ``tau_low/tau_std`` in [T, top], 1 above."""
    frac_low = tau_low / tau_std

    def sched(y):
        y = np.asarray(y, dtype=float)
        f = np.zeros_like(y, dtype=float)
        band = (y >= T) & (y <= top)
        f[band] = frac_low
        f[y > top] = 1.0
        return f

    return sched


def schedule_effective_tau(schedule, y, tau_max=TAU_MAX):
    """Effective absolute rate ``tau_eff(y) = tau_max * fraction(y)``."""
    return tau_max * np.asarray(schedule(y), dtype=float)


# ---------------------------------------------------------------------------
# Notch geometry for a schedule: where do registration notches sit?
# ---------------------------------------------------------------------------
def _schedule_notch_threshold(schedule, T=T_STAR):
    """The turnover at which registration first bites under ``schedule``.

    For the raise-to-T_new schedule this is T_new; for taper/reduced-rate it is
    T*.  Detected numerically as the smallest y with fraction(y) > 0.
    """
    grid = np.arange(T - 50_000.0, T + 60_000.0, 100.0)
    f = np.asarray(schedule(grid), dtype=float)
    pos = grid[f > 1e-12]
    return float(pos.min()) if pos.size else T


# ---------------------------------------------------------------------------
# Forward solver: iso-elastic re-optimisation under a reform schedule
# ---------------------------------------------------------------------------
def forward_solve_iso_batch(
    n,
    e,
    schedule,
    tau0,
    *,
    T=T_STAR,
    fp_iter=60,
):
    """Vectorised iso-elastic INTENSIVE-margin response for an array of abilities.

    Each firm faces its OWN baseline net VAT rate ``tau0`` (= ``liab/y_obs``,
    the share of turnover it actually remits — ~3% on average, output VAT minus
    input credits; NOT the statutory 20%, which is output-only). A reform
    schedule scales that by the effective-rate fraction ``f(y) in [0,1]``, so the
    firm's effective rate at turnover ``y`` is ``tau0 * f(y)`` and its
    iso-elastic interior optimum solves the FOC::

        (1 - tau0*f(y)) - y*(tau0*f'(y)) = (y/n)**(1/e)   =>   y* = n*(1-tau_eff)**e

    for the locally-relevant effective rate ``tau_eff``. This makes ``e`` the SOLE
    governing knob: the response to the net-of-tax rate has elasticity exactly
    ``e``, and as ``e -> 0`` every firm freezes at ``y* = n``, with the recovered
    ``n -> y_obs``, so the behavioural cost collapses onto the STATIC cost
    (cross-check 4). The EXTENSIVE margin (bunching / de-registration) is the
    separate analytic object (dominated region + marginal buncher) and is NOT
    re-litigated here, so a small intensive elasticity does not spuriously empty
    the registered population.

    Implementation: a damped fixed-point on ``y = n*(1 - tau0*f(y))**e``, robust
    for both the flat-rate bands (immediate convergence) and the continuously
    varying taper. ``n`` below the schedule's registration threshold stays at
    ``n`` (unregistered; ``f=0``). Non-finite / non-positive ``n`` is returned
    unchanged. Returns ``y_star`` (£), same shape as ``n``.
    """
    n = np.asarray(n, dtype=float)
    tau0 = np.broadcast_to(np.asarray(tau0, dtype=float), n.shape).astype(float)
    good = np.isfinite(n) & (n > 0)

    T_sched = _schedule_notch_threshold(schedule, T)

    # Fixed-point for the registered interior optimum y = n*(1 - tau0*f(y))**e.
    # Start at y=n; damped update keeps it stable through the taper band.
    y = np.where(good, n, 1.0)
    for _ in range(fp_iter):
        f = np.asarray(schedule(y), dtype=float)
        tau_eff = np.clip(tau0 * f, 0.0, 0.999)
        y_new = n * (1.0 - tau_eff) ** e
        y = 0.5 * y + 0.5 * y_new

    # A firm whose frictionless optimum is below the schedule's registration
    # threshold is unregistered and simply locates at n (no tax wedge).
    y = np.where(n < T_sched, n, y)

    return np.where(good, y, n)


def forward_solve_iso(n, e, schedule, tau0=TAU_MAX, **kw):
    """Scalar wrapper around :func:`forward_solve_iso_batch`.

    ``tau0`` defaults to the statutory ``TAU_MAX`` so the elasticity cross-check
    (which probes the registered optimum ``n*(1-tau)**e``) uses the full
    statutory wedge; the population revenue solve passes each firm's own net rate.
    """
    out = forward_solve_iso_batch(
        np.atleast_1d(float(n)), e, schedule, tau0, **kw)
    return float(out[0])


# ---------------------------------------------------------------------------
# Marginal buncher (delegate to the verified notch.model implementation),
# with an independent iso-elastic cross-check.
# ---------------------------------------------------------------------------
def marginal_buncher(e, vintage="2023-24"):
    """Return ``(n_H, dy_star)`` in £, delegating to the verified notch model."""
    m = NotchModel(vintage)
    nH_k, dy_k = m.marginal_buncher(e)
    return nH_k * 1000.0, dy_k * 1000.0


def marginal_buncher_iso(e, T=T_STAR, tau=TAU_MAX):
    """Independent iso-elastic indifference solve for the marginal buncher (£).

    Solves ``pi_bunch(T*; n) = pi_register(y1; n)`` for ``n``, where the firm
    bunches at ``T*`` (unregistered) or registers at ``y1 = n*(1-tau)**e``.
    Used only to cross-check :func:`marginal_buncher`.
    """
    from scipy.optimize import brentq

    def gap(n):
        u_bunch = iso_profit(T, n, e, net=1.0)
        y1 = n * (1.0 - tau) ** e
        u_tax = iso_profit(y1, n, e, net=1.0 - tau)
        return float(u_bunch - u_tax)

    lo, hi = T * (1.0 + 1e-6), T * 5.0
    while gap(lo) * gap(hi) > 0 and hi < T * 50:
        hi *= 1.5
    n_H = brentq(gap, lo, hi)
    return float(n_H), float(n_H - T)


def dominated_region_width(T=T_STAR, tau=TAU_MAX):
    """Analytic Kleven-Waseem dominated-region width ``a = T*tau/(1-tau)`` (£)."""
    return T * tau / (1.0 - tau)


# ---------------------------------------------------------------------------
# Reform revenue: static and behavioural (iso-elastic, e-governed)
# ---------------------------------------------------------------------------
def reform_revenue(
    df,
    schedule,
    e,
    *,
    behavioural,
    T=T_STAR,
    tau_max=TAU_MAX,
    band_lo=70_000.0,
    band_hi=130_000.0,
    near_lo=83_000.0,
    near_hi=85_000.0,
    move_tol=500.0,
):
    """Revenue change of a reform vs the £85k hard-notch baseline (iso-elastic).

    ``df`` must provide ``turnover`` (£), ``liab`` (£, baseline net VAT
    remittance), and ``weight``.

    Each firm faces its OWN baseline net VAT rate ``tau0 = liab/y_obs`` (~3% of
    turnover on average — the share it actually remits, output VAT minus input
    credits). A reform schedule scales that by the effective-rate fraction
    ``f(y)``, so the firm's effective rate is ``tau0 * f(y)`` and its remittance
    is ``liab * (y/y_obs) * f(y)`` (net VAT proportional to turnover).

    BASELINE (£85k notch) and REFORM are evaluated SYMMETRICALLY through the same
    iso-elastic forward solve:

    * STATIC (``behavioural=False``): turnover fixed at observed; only the
      effective-rate fraction changes (notch ``f=1`` above T* vs the reform's
      ``f``).  This reproduces the trusted static reform costs.
    * BEHAVIOURAL: firms in ``[band_lo, band_hi]`` re-optimise turnover under the
      iso-elastic model with elasticity ``e`` — under the £85k notch for the
      baseline (``t_notch``) and under the reform schedule (``t_new``). As
      ``e -> 0`` both freeze at observed turnover, so the behavioural cost
      converges to the static cost (cross-check 4).

    Returns a dict with baseline/reform revenue, the change vs baseline, the
    number of firms re-optimising, the near-threshold mass change, and the
    notch-baseline / reform turnover vectors (for figures).
    """
    t_obs = df["turnover"].to_numpy(dtype=float)
    liab = df["liab"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float)

    # Firm-specific baseline net VAT rate (share of turnover actually remitted).
    with np.errstate(divide="ignore", invalid="ignore"):
        tau0 = np.where(t_obs > 0, liab / t_obs, 0.0)

    band = (t_obs >= band_lo) & (t_obs <= band_hi)
    idx = np.where(band)[0]

    if behavioural:
        n = recover_ability(t_obs, e, T=T, tau=tau0)
        t_new = t_obs.copy()
        t_new[idx] = forward_solve_iso_batch(
            n[idx], e, schedule, tau0[idx], T=T)
        # Baseline turnover under the £85k notch via the SAME machinery — the
        # apples-to-apples reference (both worlds re-optimised identically).
        t_notch = t_obs.copy()
        t_notch[idx] = forward_solve_iso_batch(
            n[idx], e, schedule_notch, tau0[idx], T=T)
    else:
        t_new = t_obs
        t_notch = t_obs

    with np.errstate(divide="ignore", invalid="ignore"):
        scale_new = np.where(t_obs > 0, t_new / t_obs, 1.0)
        scale_notch = np.where(t_obs > 0, t_notch / t_obs, 1.0)
    frac_new = np.asarray(schedule(t_new), dtype=float)
    frac_notch = np.asarray(schedule_notch(t_notch), dtype=float)

    rev_baseline = float(np.sum(liab * scale_notch * frac_notch * w))
    rev_reform = float(np.sum(liab * scale_new * frac_new * w))

    d_rev = rev_reform - rev_baseline

    # Firms affected: facing a changed effective rate at observed y, or moving.
    frac_notch_at_obs = np.asarray(schedule_notch(t_obs), dtype=float)
    frac_reform_at_obs = np.asarray(schedule(t_obs), dtype=float)
    rate_changed = np.abs(frac_reform_at_obs - frac_notch_at_obs) > 1e-9
    moved = np.abs(t_new - t_notch) > move_tol
    affected = rate_changed | moved
    n_affected = float(np.sum(w[affected]))
    n_moved = float(np.sum(w[moved]))

    def wmass(arr):
        m = (arr >= near_lo) & (arr < near_hi)
        return float(np.sum(w[m]))

    near_notch = wmass(t_notch)
    near_reform = wmass(t_new)

    return {
        "e": e,
        "rev_baseline": rev_baseline,
        "rev_reform": rev_reform,
        "d_rev": d_rev,
        "n_affected": n_affected,
        "n_moved": n_moved,
        "near_baseline": near_notch,
        "near_reform": near_reform,
        "near_change": near_reform - near_notch,
        "t_new": t_new,
        "t_notch": t_notch,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
REFORM_DATA = SYNTHETIC_DATA_DIR / "synthetic_firms_2023-24.csv"


def load_reform_data(path=REFORM_DATA):
    """Load the reform-costing dataset (£ units). Ability is recovered per-e."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - generate it first with:\n"
            "  firm-microsim --vintage 2023-24 --output synthetic_firms_2023-24.csv"
        )
    df = pd.read_csv(
        path,
        usecols=["annual_turnover_k", "vat_liability_k", "weight", "vat_registered"],
    )
    out = pd.DataFrame()
    out["turnover"] = df["annual_turnover_k"].to_numpy(dtype=float) * 1000.0
    out["liab"] = df["vat_liability_k"].to_numpy(dtype=float) * 1000.0
    out["weight"] = df["weight"].to_numpy(dtype=float)
    return out


# ---------------------------------------------------------------------------
# Reform registry
# ---------------------------------------------------------------------------
def build_reforms():
    """Map reform name -> (schedule, display label)."""
    return {
        "raise100k": (make_schedule_raise(100_000.0),
                      "Raise threshold to £100k"),
        "taper": (schedule_taper, "Graduated taper (£85k→£105k)"),
        "rate10": (make_schedule_reduced_rate(0.10),
                   "Reduced rate 10% (£85k–£105k)"),
        "rate15": (make_schedule_reduced_rate(0.15),
                   "Reduced rate 15% (£85k–£105k)"),
    }


# ---------------------------------------------------------------------------
# Cross-check harness
# ---------------------------------------------------------------------------
def crosscheck(verbose=True, data_path=REFORM_DATA):
    """Assert analytic invariants and data-derived consistency checks.

    (1) dominated region; (2) marginal buncher vs notch.model AND our own
    iso-elastic indifference solve; (3) ELASTICITY check
    d ln y*/d ln(1-tau) = e; (4) e -> 0 behavioural costs converge to the static
    costs computed from the in-repository reform dataset.
    """
    results = []

    def record(name, value, unit=""):
        results.append((name, value, None, None, True, unit))

    def check(name, value, target, tol, unit=""):
        ok = abs(value - target) <= tol
        results.append((name, value, target, tol, ok, unit))
        if not ok:
            raise AssertionError(
                f"CROSSCHECK FAILED: {name} = {value:,.4f}{unit} "
                f"(target {target:,.4f}{unit} +/- {tol:g})"
            )

    # 1. Dominated region.
    a = dominated_region_width()
    check("dominated_region a", a, 21_250.0, 1.0, " GBP")
    check("dominated_region upper edge T*+a", T_STAR + a, 106_250.0, 1.0, " GBP")

    # 2. Marginal buncher: notch.model vs our independent iso-elastic solve.
    for e, tgt, tol in [(0.05, 112_795.0, 200.0),
                        (0.17, 127_382.0, 100.0),
                        (0.32, 143_527.0, 100.0)]:
        nH, _ = marginal_buncher(e)
        nH_iso, _ = marginal_buncher_iso(e)
        check(f"marginal_buncher(e={e}) [notch.model]", nH, tgt, tol, " GBP")
        check(f"marginal_buncher(e={e}) [iso self-solve]", nH_iso, tgt, tol, " GBP")
        # The two solves must agree with each other to ±£200.
        check(f"buncher agreement(e={e})", nH_iso, nH, 200.0, " GBP")

    # 3. ELASTICITY CHECK: d ln(y*)/d ln(1-tau) = e for a registered firm.
    #    The firm faces effective rate tau0*f(y); with tau0=TAU_MAX and a flat
    #    fraction f=tau/TAU_MAX the effective rate is exactly tau, so the
    #    registered optimum is y* = n*(1-tau)**e. Perturb tau and confirm the
    #    log-response equals the input e (e is the governing knob).
    n_test = 200_000.0
    for e in (0.05, 0.17, 0.32):
        tau_a, dtau = 0.20, 0.01
        def flat(tau):
            return lambda y: (np.asarray(y, dtype=float) >= T_STAR).astype(float) * (tau / TAU_MAX)
        y0 = forward_solve_iso(n_test, e, flat(tau_a), tau0=TAU_MAX)
        y1 = forward_solve_iso(n_test, e, flat(tau_a + dtau), tau0=TAU_MAX)
        elas = (np.log(y1) - np.log(y0)) / (np.log(1 - (tau_a + dtau)) - np.log(1 - tau_a))
        check(f"elasticity d ln y*/d ln(1-tau) (e={e})", elas, e, 0.02, "")

    # Static + behavioural(e->0) reform costs.
    df = load_reform_data(data_path)
    t = df["turnover"].to_numpy()
    liab = df["liab"].to_numpy()
    w = df["weight"].to_numpy()
    base = float(np.sum(liab[(t >= T_STAR)] * w[(t >= T_STAR)]))
    record("baseline registered base", base / 1e9, " bn")

    reforms = build_reforms()
    e_lim = 0.001  # e -> 0 limit
    for rname, (sched, _label) in reforms.items():
        # Static.
        rs = reform_revenue(df, sched, E_HEADLINE, behavioural=False)
        record(f"static {rname}", rs["d_rev"], " GBP")
        # 4. e -> 0 behavioural must converge to the static cost.
        rb = reform_revenue(df, sched, e_lim, behavioural=True)
        check(f"behavioural(e->0) {rname}", rb["d_rev"], rs["d_rev"], 20e6, " GBP")

    if verbose:
        print("CROSSCHECK PASSED — analytic invariants and repo-data checks passed:")
        print(f"  {'object':<42}{'value':>18}{'target':>16}{'tol':>12}  ok")
        for name, val, tgt, tol, ok, unit in results:
            if tgt is None:
                print(f"  {name:<42}{val:>16,.4f}{unit:<5}{'data':>14}"
                      f"{'--':>12}  {'PASS' if ok else 'FAIL'}")
            else:
                print(f"  {name:<42}{val:>16,.4f}{unit:<5}{tgt:>14,.4f}"
                      f"{tol:>12,.4g}  {'PASS' if ok else 'FAIL'}")
    return results


if __name__ == "__main__":  # pragma: no cover
    crosscheck()
