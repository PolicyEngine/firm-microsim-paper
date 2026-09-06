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

SCOPE: INTENSIVE MARGIN ONLY, REGION-CONFINED
---------------------------------------------
The behavioural layer prices the INTENSIVE margin only: each firm re-optimises
turnover *within the schedule region that contains its observed turnover*
(regions are the maximal intervals on which the effective-rate fraction ``f``
is constant). Relocation across a notch — bunching below a threshold from
above, or de-registering past a band edge — is the EXTENSIVE margin, which is
the separate analytic object (dominated region + marginal buncher) and is
deliberately out of scope here. Confining the response to the firm's own
region makes the solve a closed form, makes the ``e -> 0`` limit collapse onto
the static costing *exactly*, and removes any dependence on solver iteration
details at schedule discontinuities (a damped fixed point has no fixed point
for abilities that straddle a notch, so its iterates oscillate and the
"solution" is an artifact of the iteration count — the previous implementation
had exactly this defect).

RESPONSE (formulation A: value-added tax, deductible share cancels)
-------------------------------------------------------------------
Under formulation A the firm maximises
``pi = (1 - delta)(1 - tau f(y)) y - C(y; n, e)``, so the interior optimum is
``y* = n [(1-delta)(1-tau f)]**e``. The reform response RATIO from the
baseline (fraction ``f0``) to the reform (fraction ``f1``) is therefore::

    y_star / y_obs = [(1 - tau*f1) / (1 - tau*f0)] ** e,

with ``tau`` the STATUTORY rate: the deductible share ``delta`` cancels, so
the response needs neither ``delta`` nor the firm's net rate — ``e`` is the
sole knob (``d ln y / d ln(1 - tau f) = e`` exactly). The ratio is clipped to
the firm's schedule region.

REVENUE CONVENTION
------------------
The stored net VAT remittance ``liab`` is NOT ``0.20 * turnover`` — it is the
standard rate applied to value added, ``0.20 * (turnover - input)`` (a
firm-specific net rate ``liab/y_obs`` of roughly 8% of turnover, i.e.
``0.20 * value-added-share``). Under a reform schedule with effective-rate
fraction ``f(y) in [0,1]``, a registered firm's reform remittance is
``liab * (y_star / y_obs) * f(y_star)`` — value added stays a fixed share of
turnover, turnover is re-optimised, and the schedule fraction is applied. In
the ``e -> 0`` (no-response) limit ``y_star = y_obs`` and this reproduces the
STATIC reform costs exactly.

CROSS-CHECKS (see :func:`crosscheck`)
-------------------------------------
1. Dominated region ``a = T* tau/(1-tau) = £21,250``; upper edge £106,250.
2. Marginal buncher ``n_H(e)`` = 112,795 / 127,382 / 143,527 at
   e = 0.05 / 0.17 / 0.32 (our iso-elastic indifference solve matches
   ``notch.model`` to within ±£200).
3. Elasticity check: ``d ln(y_star)/d ln(1-tau f) = e`` numerically.
4. ``e -> 0`` limit: behavioural reform costs equal the static costs to
   first order in ``e`` (checked at e = 1e-6 to within £0.1m).
5. Baseline reproduction: solving the BASELINE schedule returns every firm's
   observed turnover exactly (the accounting anchor is internally consistent).
6. Raise-to-£100k invariance: the behavioural cost of a pure threshold raise
   equals its static cost at EVERY ``e`` — released firms leave the VAT base
   (their expansion is untaxed) and no other firm's effective rate changes,
   so a level move has no intensive-margin revenue offset in this model.
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
TAPER_TOP = 105_000.0   # reduced-rate band upper edge (£)
# Minimum band top at which a band-confined monotone taper exists: a linear
# marginal remittance rate phasing 0 -> 100% over [T, top] meets L(top) =
# tau_max * top (no relief above the band) iff (top - T)/2 = tau_max * top,
# i.e. top = T / (1 - 2 * tau_max). At tau_max = 0.20 this is £141,666.67 —
# a 20% VAT taper cannot fit in a £20k band.
TAPER_WIDE_TOP = T_STAR / (1.0 - 2.0 * TAU_MAX)

# ASSUMED sweep values for the e-sensitivity analysis. None is identified from
# the synthetic data. The low end (0.05) is the external Kleven-Waseem (2013)
# anchor; 0.17 and 0.32 are assumption choices spanning the range used in the
# threshold literature, reported to show how the costings move with e.
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


# Piecewise-flat region metadata: list of (lo, hi, fraction) half-open [lo, hi)
# intervals covering [0, inf). ``None`` marks a schedule that is NOT piecewise
# flat (the taper), for which the region-confined intensive solve is undefined.
schedule_notch.regions = [(0.0, T_STAR, 0.0), (T_STAR, np.inf, 1.0)]


def make_schedule_raise(T_new=100_000.0):
    """Raise the threshold to ``T_new``: the notch simply moves to ``T_new``."""

    def sched(y):
        y = np.asarray(y, dtype=float)
        return (y >= T_new).astype(float)

    sched.regions = [(0.0, float(T_new), 0.0), (float(T_new), np.inf, 1.0)]
    return sched


def schedule_taper_average(y, T=T_STAR, top=TAPER_TOP):
    """Legacy taper: *average* effective fraction phases 0 -> 1 linearly over
    [T, top].

    Retained only for recosting/comparison. This schedule is NOT monotone in
    net revenue: with liability tau_max * f(y) * y = tau_max * (y - T) / (top - T)
    * y, net revenue R(y) = y * (1 - tau_max * (y - T) / (top - T)) peaks strictly
    inside the band and declines toward ``top``, creating a dominated interval and
    an implied marginal remittance rate above 100%. Use ``schedule_taper`` for the
    monotone design.
    """
    y = np.asarray(y, dtype=float)
    return np.clip((y - T) / (top - T), 0.0, 1.0)


schedule_taper_average.regions = None


def schedule_taper_marginal_relief(y, T=T_STAR, top=TAPER_TOP, tau_max=TAU_MAX):
    """Marginal-relief taper: the *marginal* remittance rate phases 0 -> tau_max
    linearly over [T, top], so net revenue is strictly increasing everywhere and
    no dominated interval exists.

    Retained only for comparison — NOT the shipped design. Above ``top`` the
    liability is tau_max * (y - (top + T)/2) forever, i.e. every firm above the
    nominal band keeps a permanent relief of tau_max * (top + T)/2 on its
    remittance base. The policy's incidence is therefore economy-wide above T,
    not band-confined, and its cost (~£7.7bn) is dominated by firms above the
    band. Use ``schedule_taper`` (the wide-band design) for a band-confined
    monotone taper.

    Liability in the band is the integral of the marginal rate,

        L(y) = tau_max * (y - T)**2 / (2 * (top - T)),

    which as an *average*-rate fraction f (with L = tau_max * f(y) * y) is

        f(y) = (y - T)**2 / (2 * (top - T) * y),          T <= y <= top,
        f(y) = (y - (top + T) / 2) / y,                   y > top.

    The marginal remittance rate is tau_max * (y - T) / (top - T) <= tau_max < 1
    throughout the band, so d/dy [ y * (1 - tau_max * f(y)) ] = 1 - marginal rate
    > 0: net revenue never falls. f is continuous at ``top`` and at ``T`` (f=0).

    Note this design reaches the full standard rate only asymptotically above
    ``top`` rather than exactly at ``top``: a monotone schedule that hit f=1 at
    ``top`` is impossible while R(T) = T > 0.8 * top, which is precisely why the
    original average-rate taper had to dip.
    """
    y = np.asarray(y, dtype=float)
    T = float(T)
    top = float(top)
    band = (y >= T) & (y <= top)
    above = y > top
    f = np.zeros_like(y, dtype=float)
    # Guard division by zero at y == 0 (f stays 0 there since band/above are False).
    safe_y = np.where(y > 0, y, 1.0)
    f = np.where(band, (y - T) ** 2 / (2.0 * (top - T) * safe_y), f)
    f = np.where(above, (y - (top + T) / 2.0) / safe_y, f)
    return f


schedule_taper_marginal_relief.regions = None


def schedule_taper(y, T=T_STAR, top=TAPER_WIDE_TOP, tau_max=TAU_MAX):
    """Wide-band monotone taper: the *marginal* remittance rate phases 0 -> 100%
    linearly over [T, top], with ``top = T / (1 - 2 * tau_max)`` chosen so the
    liability meets the full standard-rate line tau_max * y exactly at ``top``.

    Liability in the band is the integral of the marginal rate m(y) =
    (y - T) / (top - T):

        L(y) = (y - T)**2 / (2 * (top - T)),          T <= y <= top,
        L(y) = tau_max * y,                            y > top,

    which as an *average*-rate fraction f (with L = tau_max * f(y) * y) is

        f(y) = (y - T)**2 / (2 * tau_max * (top - T) * y),   T <= y <= top,
        f(y) = 1,                                             y > top.

    Continuity at ``top`` requires (top - T)/2 = tau_max * top, i.e.
    top = T / (1 - 2 * tau_max) — £141,666.67 at tau_max = 0.20. The design is
    band-confined (zero relief above ``top``), continuous at both edges
    (f(T) = 0, f(top) = 1), and net revenue R(y) = y - L(y) has
    dR/dy = 1 - m(y) >= 0 throughout, vanishing only at the single point
    ``top``: no dominated interval of positive measure, at the price of a
    marginal remittance rate that reaches 100% at the band top.

    A narrower band cannot deliver all three properties at once: with top =
    £105k, R(top) = 0.8 * 105k = £84k < £85k = R(T), so any band-confined
    schedule reaching f = 1 at £105k must dip (the legacy average-rate taper's
    dominated interval), and any monotone alternative must leak relief above
    the band (the marginal-relief variant's £7.7bn).
    """
    y = np.asarray(y, dtype=float)
    T = float(T)
    top = float(top)
    band = (y >= T) & (y <= top)
    above = y > top
    f = np.zeros_like(y, dtype=float)
    # Guard division by zero at y == 0 (f stays 0 there since band/above are False).
    safe_y = np.where(y > 0, y, 1.0)
    f = np.where(band, (y - T) ** 2 / (2.0 * tau_max * (top - T) * safe_y), f)
    f = np.where(above, 1.0, f)
    return f


# The taper's fraction varies continuously with y, so it has no flat regions:
# its intensive response cannot be priced by the region-confined solve (and its
# marginal-rate channel is not represented by the flat-rate FOC), so the
# behavioural layer excludes it.
schedule_taper.regions = None


def taper_band_top(m: float, T=T_STAR, tau_max=TAU_MAX) -> float:
    """Band top ``U(m) = m T / (m - tau_max)`` of a band-confined taper with a
    CONSTANT marginal remittance rate ``m`` on ``[T, U]``.

    Continuity with the standard-rate line requires ``m (U - T) = tau_max U``.
    ``m`` must exceed ``tau_max``; ``m -> 1`` gives the infimum ``T/(1-tau_max)``
    (= £106,250: net revenue flat across the band, weakly dominated), and
    ``m = 0.5`` gives £141,667 -- the same band top the linear 0->100% design
    reaches, at half its peak marginal rate. The band top is therefore a
    property of the chosen shape, not a requirement of removing the notch.
    """
    if m <= tau_max:
        raise ValueError("constant marginal rate must exceed the standard rate")
    return m * T / (m - tau_max)


def make_schedule_taper_flat(m: float = 0.5, T=T_STAR, tau_max=TAU_MAX):
    """Band-confined taper with a CONSTANT marginal remittance rate ``m``.

    Liability ``L(y) = m (y - T)`` on ``[T, U(m)]`` and ``tau_max y`` above;
    the average-rate fraction is ``f = m (y - T) / (tau_max y)`` in the band.
    Net revenue has slope ``1 - m > 0`` throughout, so no dominated interval
    exists and the peak marginal rate is ``m`` rather than the 100% the linear
    design reaches at its top. Default ``m = 0.5`` shares the shipped linear
    taper's band top (£141,667) for a like-for-like cost comparison.
    """
    top = taper_band_top(m, T, tau_max)

    def sched(y):
        y = np.asarray(y, dtype=float)
        T_ = float(T)
        band = (y >= T_) & (y <= top)
        above = y > top
        f = np.zeros_like(y, dtype=float)
        safe_y = np.where(y > 0, y, 1.0)
        f = np.where(band, m * (y - T_) / (tau_max * safe_y), f)
        f = np.where(above, 1.0, f)
        return f

    sched.regions = None
    sched.band_top = top
    sched.marginal_rate = m
    return sched


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

    sched.regions = [
        (0.0, float(T), 0.0),
        (float(T), float(top), float(frac_low)),
        (float(top), np.inf, 1.0),
    ]
    return sched


def schedule_effective_tau(schedule, y, tau_max=TAU_MAX):
    """Effective absolute rate ``tau_eff(y) = tau_max * fraction(y)``."""
    return tau_max * np.asarray(schedule(y), dtype=float)


# ---------------------------------------------------------------------------
# Forward solver: region-confined iso-elastic response under a reform schedule
# ---------------------------------------------------------------------------
def forward_solve_iso_batch(
    n,
    e,
    schedule,
    tau0,
    *,
    T=T_STAR,
    y_obs=None,
    base_schedule=schedule_notch,
    tau_std=TAU_MAX,
    **_deprecated,
):
    """Vectorised region-confined INTENSIVE-margin response (closed form).

    Each firm's observed turnover ``y_obs`` is anchored as optimal under the
    BASELINE schedule (the £85k notch, fraction ``f0``). Under a reform whose
    fraction on the firm's own region is ``f1``, the formulation-A optimum is::

        y_star = y_obs * [(1 - tau_std*f1) / (1 - tau_std*f0)] ** e,

    the deductible share cancelling in the ratio, then clipped to the reform
    region containing ``y_obs``. Firms whose region fraction is unchanged
    (``f1 = f0``) do not move; released firms (``f1 = 0 < f0``) expand toward
    their frictionless optimum but remit nothing; band firms scale up as the
    band rate falls, clipped at the band top. Crossing a notch (the extensive
    margin) is out of scope by construction — see the module docstring.

    ``n`` and ``tau0`` are accepted for API compatibility; the formulation-A
    response ratio requires neither (the statutory ``tau_std`` and the
    schedule fractions pin it). ``schedule`` must be piecewise flat (a
    ``.regions`` attribute); the taper has ``regions = None`` and raises.
    Returns ``y_star`` (£), same shape as ``y_obs``.
    """
    regions = getattr(schedule, "regions", None)
    if regions is None:
        raise ValueError(
            "region-confined intensive solve requires a piecewise-flat "
            "schedule; the graduated taper is excluded from the behavioural "
            "layer (its rate varies continuously with turnover)."
        )
    if y_obs is None:
        raise TypeError("forward_solve_iso_batch requires y_obs (observed £)")

    y_obs = np.asarray(y_obs, dtype=float)
    good = np.isfinite(y_obs) & (y_obs > 0)

    f0 = np.asarray(base_schedule(y_obs), dtype=float)
    f1 = np.asarray(schedule(y_obs), dtype=float)

    ratio = ((1.0 - tau_std * f1) / (1.0 - tau_std * f0)) ** e
    y_star = y_obs * ratio

    # Clip into the reform region containing y_obs. Region membership follows
    # the schedule callable itself (fraction match), so boundary conventions
    # (e.g. an inclusive band top) agree exactly with the static evaluation.
    penny = 0.01
    for lo, hi, frac in regions:
        member = good & (np.abs(f1 - frac) < 1e-12) & (y_obs >= lo)
        if not np.any(member):
            continue
        if np.isfinite(hi):
            f_at_hi = float(np.asarray(schedule(np.array([hi])), dtype=float)[0])
            hi_clip = hi if abs(f_at_hi - frac) < 1e-12 else hi - penny
        else:
            hi_clip = np.inf
        y_star = np.where(member, np.clip(y_star, lo, hi_clip), y_star)

    return np.where(good, y_star, y_obs)


def forward_solve_iso(y_obs, e, schedule, tau0=TAU_MAX, **kw):
    """Scalar wrapper around :func:`forward_solve_iso_batch`.

    The first argument is the firm's observed (baseline-anchored) turnover in
    pounds; the response ratio is governed solely by ``e``, the statutory rate,
    and the baseline/reform schedule fractions at that turnover.
    """
    out = forward_solve_iso_batch(
        None, e, schedule, tau0, y_obs=np.atleast_1d(float(y_obs)), **kw)
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


def marginal_buncher_iso(e, T=T_STAR, tau=TAU_MAX, delta=0.0):
    """Iso-elastic indifference solve for the marginal buncher (£).

    Solves ``pi_bunch(T*; n) = pi_register(y1; n)`` for ``n`` under
    formulation A with deductible-input share ``delta``: bunching yields
    ``(1-delta) T* - C(T*)`` and registering yields
    ``(1-delta)(1-tau) y1 - C(y1)`` at ``y1 = n[(1-delta)(1-tau)]**e``. The
    deductible share does NOT cancel here (it scales revenue but not the
    own-factor cost), so ``n_H`` rises with ``delta``; ``delta = 0`` reproduces
    the turnover-tax solve in :mod:`firm_microsim.notch.model`.
    """
    from scipy.optimize import brentq

    va = 1.0 - delta
    net_reg = va * (1.0 - tau)

    def gap(n):
        # Best unregistered choice: the untaxed optimum, capped at the
        # threshold (a firm cannot stay unregistered above T*).
        y_u = min(T, n * va ** e)
        u_bunch = iso_profit(y_u, n, e, net=va)
        # Registered optimum; registering with y1 < T* is never chosen (it is
        # dominated by staying unregistered at y1), so the branch starts at T*.
        y1 = max(T, n * net_reg ** e)
        u_tax = iso_profit(y1, n, e, net=net_reg)
        return float(u_bunch - u_tax)

    # At n_lo the registered optimum is exactly T*, where bunching strictly
    # wins (same turnover, no tax), so gap(lo) > 0 and the root lies above.
    lo = T / net_reg ** e * (1.0 + 1e-9)
    hi = lo * 2.0
    while gap(hi) > 0 and hi < T * 1e3:
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

    A firm's remittance under a schedule with effective-rate fraction ``f(y)``
    is ``liab * (y/y_obs) * f(y)``: net VAT stays proportional to turnover
    (value added a fixed share) and the schedule fraction is applied.

    BASELINE (£85k notch) and REFORM are evaluated SYMMETRICALLY through the
    same region-confined forward solve:

    * STATIC (``behavioural=False``): turnover fixed at observed; only the
      effective-rate fraction changes (notch ``f=1`` above T* vs the reform's
      ``f``).  This reproduces the trusted static reform costs.
    * BEHAVIOURAL: firms in ``[band_lo, band_hi]`` re-optimise turnover within
      their own schedule region (formulation-A ratio, elasticity ``e``) — under
      the £85k notch for the baseline (``t_notch``, which reproduces observed
      turnover exactly: the accounting anchor) and under the reform schedule
      (``t_new``). As ``e -> 0`` the ratio -> 1, so the behavioural cost
      collapses onto the static cost exactly (cross-check 4).

    Returns a dict with baseline/reform revenue, the change vs baseline, the
    number of firms re-optimising, the near-threshold mass change, and the
    notch-baseline / reform turnover vectors (for figures).
    """
    t_obs = df["turnover"].to_numpy(dtype=float)
    liab = df["liab"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float)

    # Firm-specific baseline net VAT rate (share of turnover actually remitted).
    # Not needed for the formulation-A response ratio (the deductible share
    # cancels); retained for diagnostics.
    with np.errstate(divide="ignore", invalid="ignore"):
        tau0 = np.where(t_obs > 0, liab / t_obs, 0.0)

    band = (t_obs >= band_lo) & (t_obs <= band_hi)
    idx = np.where(band)[0]

    if behavioural:
        t_new = t_obs.copy()
        t_new[idx] = forward_solve_iso_batch(
            None, e, schedule, tau0[idx], T=T, y_obs=t_obs[idx])
        # Baseline turnover under the £85k notch via the SAME machinery — this
        # must reproduce observed turnover exactly (accounting anchor).
        t_notch = t_obs.copy()
        t_notch[idx] = forward_solve_iso_batch(
            None, e, schedule_notch, tau0[idx], T=T, y_obs=t_obs[idx])
        anchor_gap = float(np.max(np.abs(t_notch - t_obs))) if idx.size else 0.0
        if anchor_gap > 1e-6:
            raise AssertionError(
                f"baseline solve failed to reproduce observed turnover "
                f"(max gap £{anchor_gap:.4f}) — accounting anchor violated"
            )
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
    """Load the reform-costing dataset (£ units), in-scope liabilities only."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - generate it first with:\n"
            "  firm-microsim --vintage 2023-24 --output synthetic_firms_2023-24.csv"
        )
    df = pd.read_csv(
        path,
        usecols=["annual_turnover_k", "vat_liability_k", "weight", "vat_scope", "vat_registered"],
    )
    # Out-of-scope enterprises (PAYE-only / exempt-sector, issue #37) face no
    # VAT schedule: drop them so every reform is priced, and every firm
    # counted, on the in-scope population only.
    df = df[df["vat_scope"].astype(bool)].reset_index(drop=True)
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
        "taper": (schedule_taper, "Graduated taper (£85k→£141.7k, monotone)"),
        "taper_flat50": (make_schedule_taper_flat(0.5),
                         "Flat 50% marginal taper (£85k→£141.7k)"),
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

    # 3. ELASTICITY CHECK: d ln(y*)/d ln(1 - tau*f) = e for a registered firm.
    #    Probe two wide reduced-rate bands (fractions 0.5 and 0.6, effective
    #    rates 10% and 12%); the log response ratio must equal e exactly.
    y_probe = 200_000.0
    for e in (0.05, 0.17, 0.32):
        s_a = make_schedule_reduced_rate(0.10, top=1e9)
        s_b = make_schedule_reduced_rate(0.12, top=1e9)
        y_a = forward_solve_iso(y_probe, e, s_a)
        y_b = forward_solve_iso(y_probe, e, s_b)
        elas = (np.log(y_a) - np.log(y_b)) / (np.log(1 - 0.10) - np.log(1 - 0.12))
        check(f"elasticity d ln y*/d ln(1-tau f) (e={e})", elas, e, 1e-9, "")

    # Static + behavioural reform costs on the in-repository population.
    df = load_reform_data(data_path)
    t = df["turnover"].to_numpy()
    liab = df["liab"].to_numpy()
    w = df["weight"].to_numpy()
    base = float(np.sum(liab[(t >= T_STAR)] * w[(t >= T_STAR)]))
    record("baseline registered base", base / 1e9, " bn")

    reforms = build_reforms()
    static_costs = {}
    for rname, (sched, _label) in reforms.items():
        # Static (all reforms, including the taper).
        rs = reform_revenue(df, sched, E_HEADLINE, behavioural=False)
        static_costs[rname] = rs["d_rev"]
        record(f"static {rname}", rs["d_rev"], " GBP")
        if getattr(sched, "regions", None) is None:
            continue  # taper: behavioural layer excluded (non-flat schedule)
        # 4. e -> 0 nesting must be EXACT (to £0.1m at e = 1e-6). The baseline
        # reproduction (check 5) is asserted inside reform_revenue itself.
        rb = reform_revenue(df, sched, 1e-6, behavioural=True)
        check(f"behavioural(e->0) {rname}", rb["d_rev"], rs["d_rev"], 0.1e6, " GBP")

    # 6. Raise-to-£100k invariance: a pure level move has NO intensive-margin
    # revenue offset — behavioural cost equals static cost at EVERY swept e
    # (released firms leave the base; no other firm's effective rate changes).
    sched_raise, _ = reforms["raise100k"]
    for e in ELASTICITIES:
        rb = reform_revenue(df, sched_raise, e, behavioural=True)
        check(f"raise100k behavioural==static (e={e})",
              rb["d_rev"], static_costs["raise100k"], 0.1e6, " GBP")

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
