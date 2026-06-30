"""Parametric, static EXTENSIVE-MARGIN VAT reform costing (Liu et al. 2021 setting).

This module complements the intensive-margin reform costing in
:mod:`firm_microsim.dynamic.model` (which conditions on the observed
registration allocation and re-optimises turnover via the iso-elastic
response). Here we instead model the REGISTRATION DECISION itself as a choice
driven by two sector parameters, exactly as in the Liu, Lockwood, Almunia & Tam
(2021, REStat 103(1)) mechanism:

    * input share ``omega``  -- the fraction of turnover that is bought-in inputs;
    * B2C share  ``beta``    -- the fraction of sales to final consumers.

ACCOUNTING (implemented exactly)
--------------------------------
Statutory rate ``t = TAU_MAX = 0.20``; baseline threshold ``T* = 85,000``.
For a firm with turnover ``y`` and weight ``w``:

    * Government net VAT if REGISTERED   :  ``t * (1 - omega) * y``
      (output VAT minus reclaimed input VAT = VAT on value added).
    * Government VAT if UNREGISTERED     :  ``t * omega * y``
      (the irrecoverable input VAT the firm pays and cannot reclaim).

Registration decision at threshold ``T`` and band rate ``r``:

    * registered if ``y >= T``                               (mandatory), OR
    * voluntary registration if ``omega - beta*(1 - omega) > 0``
      (registering is a net gain even below the threshold -- the input-heavy
      / low-B2C firms in Liu et al.), OR
    * otherwise unregistered.

Baseline revenue ``R0 = sum_i w_i * rev(y_i; T*, omega, beta)``.

REFORMS COSTED
--------------
1. ``raise100k`` -- raise the registration threshold to £100,000: identical
   accounting with ``T = 100,000``. Firms in [85k, 100k) that do NOT
   voluntarily register flip from registered (``t(1-omega)y``) to unregistered
   (``t*omega*y``).
2. ``rate10`` / ``rate15`` -- reduced-rate band of 10% / 15% over [85k, 105k]
   (threshold unchanged at £85k): a REGISTERED firm with turnover in the band
   remits the band rate ``r`` on its value added (``r*(1-omega)*y`` instead of
   ``t*(1-omega)*y``); above the band it remits ``t*(1-omega)*y``. The
   registration decision uses the same ``omega - beta(1-omega) > 0`` rule (the
   sign is unchanged by scaling with ``r``). Unregistered firms are unchanged
   (``t*omega*y``).

Cost of a reform = reform revenue - baseline revenue (population-weighted, £).

CAVEATS (stated transparently)
------------------------------
* This is a STATIC EXTENSIVE-MARGIN accounting: the only behaviour modelled is
  whether a firm is in or out of VAT (registered vs not). There is NO turnover
  response, NO bunching, NO general equilibrium.
* B2B trade involves VAT cascading along the supply chain. We SIMPLIFY this to
  the single figure ``t*omega*y`` for an unregistered firm (the irrecoverable
  input VAT it bears); we do not trace the downstream un-reclaimable VAT on its
  sales to other firms. This is deliberately coarser than Liu et al.'s full
  CES-demand behavioural model.
* It is NOT data-grounded: ``omega`` and ``beta`` are PARAMETERS, not measured
  from data. We sweep them, so the output is a SENSITIVITY RANGE, not a point
  estimate. The population (turnover ``y``, weight ``w``) is the in-repository
  generated 2023-24 firm population.
* The graduated taper is omitted here (as in the intensive-margin model).
"""

from __future__ import annotations

import argparse

import numpy as np

from ..config import RESULTS_DIR
from .liu_mechanism import (
    ACCENT,
    LABEL_SIZE,
    PALETTE,
    TICK_SIZE,
    _save,
    _style_ax,
)
from .model import TAPER_TOP, TAU_MAX, T_STAR, load_reform_data

# ---------------------------------------------------------------------------
# Reform parameters
# ---------------------------------------------------------------------------
T_RAISED = 100_000.0          # raised registration threshold (£)
BAND_LO = T_STAR              # reduced-rate band lower edge (£85,000)
BAND_HI = TAPER_TOP           # reduced-rate band upper edge (£105,000)
BAND_RATES = {"rate10": 0.10, "rate15": 0.15}
REFORMS = ("raise100k", "rate10", "rate15")

# Representative (omega, beta) firm types for the headline table.
REPRESENTATIVE = [
    (0.20, 0.90, "consumer-facing, low-input"),
    (0.45, 0.50, "mixed"),
    (0.70, 0.20, "input-heavy B2B"),
    (0.50, 0.50, "population-average (illustrative)"),
]

# Sensitivity sweep grids.
OMEGA_SWEEP = (0.20, 0.35, 0.50, 0.65, 0.80)
BETA_SWEEP = (0.20, 0.50, 0.80)


# ---------------------------------------------------------------------------
# Core accounting
# ---------------------------------------------------------------------------
def gov_revenue(y, registered, omega, t, rate=None):
    """Per-firm government VAT (£).

    Registered firms remit ``rate*(1-omega)*y`` (VAT on value added), where
    ``rate`` defaults to the statutory ``t`` and may be set to a band rate ``r``
    for in-band registered firms (scalar or per-firm array). Unregistered firms
    yield ``t*omega*y`` -- the irrecoverable input VAT, always at the statutory
    rate ``t``.
    """
    y = np.asarray(y, dtype=float)
    registered = np.asarray(registered)
    eff_rate = t if rate is None else rate
    registered_rev = np.asarray(eff_rate, dtype=float) * (1.0 - omega) * y
    unregistered_rev = t * omega * y
    return np.where(registered, registered_rev, unregistered_rev)


def is_registered(y, T, omega, beta):
    """Registration indicator: mandatory above ``T`` or voluntary below it.

    Voluntary registration occurs iff ``omega - beta*(1 - omega) > 0`` (the
    input-heavy / low-B2C firms for which reclaiming input VAT outweighs the
    output-VAT burden on consumer sales).
    """
    y = np.asarray(y, dtype=float)
    mandatory = y >= T
    voluntary = (omega - beta * (1.0 - omega)) > 0.0
    return mandatory | bool(voluntary)


def reform_cost(df, omega, beta, reform):
    """Population-weighted baseline/reform revenue and cost (£) for a reform.

    ``reform`` in ``{"raise100k", "rate10", "rate15"}``. Returns a dict with
    ``baseline``, ``reform`` (revenues) and ``cost`` (reform - baseline), in £.
    """
    if reform not in REFORMS:
        raise ValueError(f"unknown reform {reform!r}; choose from {REFORMS}")
    y = df["turnover"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float)
    t = TAU_MAX

    # Baseline: statutory rate everywhere, threshold T*.
    reg0 = is_registered(y, T_STAR, omega, beta)
    rev0 = gov_revenue(y, reg0, omega, t)
    baseline = float(np.sum(w * rev0))

    if reform == "raise100k":
        reg = is_registered(y, T_RAISED, omega, beta)
        rev = gov_revenue(y, reg, omega, t)
    else:  # reduced-rate band; threshold unchanged at T*.
        r = BAND_RATES[reform]
        reg = is_registered(y, T_STAR, omega, beta)
        in_band = (y >= BAND_LO) & (y <= BAND_HI)
        rate = np.where(in_band, r, t)
        rev = gov_revenue(y, reg, omega, t, rate=rate)

    reform_rev = float(np.sum(w * rev))
    return {"baseline": baseline, "reform": reform_rev,
            "cost": reform_rev - baseline}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _bn(x):
    return x / 1e9


def build_report(df):
    """Assemble the text report: representative costs + raise-100k sweep."""
    lines = [
        "Liu et al. (2021) setting -- PARAMETRIC static EXTENSIVE-MARGIN VAT",
        "reform costing (registration in/out, omega & beta as parameters)",
        "=" * 70,
        "",
        "ACCOUNTING (per firm, turnover y, weight w):",
        f"  registered    -> government VAT = t*(1-omega)*y   (t={TAU_MAX})",
        "  unregistered  -> government VAT = t*omega*y       (irrecoverable input VAT)",
        "  registered iff  y >= T  OR  omega - beta*(1-omega) > 0  (voluntary)",
        f"  baseline threshold T* = GBP {T_STAR:,.0f}",
        "",
        "REFORMS:",
        f"  raise100k : raise threshold to GBP {T_RAISED:,.0f}",
        f"  rate10/15 : reduced band rate 10%/15% over "
        f"[GBP {BAND_LO:,.0f}, GBP {BAND_HI:,.0f}], threshold unchanged",
        "  cost = reform revenue - baseline revenue (population-weighted)",
        "",
        "CAVEATS: static; only the in/out registration choice is modelled (no",
        "turnover response, no bunching, no GE). B2B cascading simplified to the",
        "single t*omega*y unregistered figure. omega, beta are PARAMETERS (swept),",
        "not data -- output is a sensitivity range. Graduated taper omitted.",
        "",
        "-" * 70,
        "Table 1 -- reform cost (GBP bn) at representative (omega, beta) pairs",
        "-" * 70,
        f"  {'firm type':<34}{'omega':>6}{'beta':>6}"
        f"{'raise100k':>12}{'rate10':>10}{'rate15':>10}",
    ]
    for omega, beta, label in REPRESENTATIVE:
        costs = {r: reform_cost(df, omega, beta, r)["cost"] for r in REFORMS}
        lines.append(
            f"  {label:<34}{omega:>6.2f}{beta:>6.2f}"
            f"{_bn(costs['raise100k']):>12.3f}"
            f"{_bn(costs['rate10']):>10.3f}{_bn(costs['rate15']):>10.3f}"
        )
    lines += [
        "  (costs in GBP bn; negative = revenue LOSS to government)",
        "",
        "-" * 70,
        "Table 2 -- raise-to-GBP100k cost (GBP bn) sensitivity to (omega, beta)",
        "-" * 70,
        f"  {'omega \\ beta':<14}"
        + "".join(f"{b:>12.2f}" for b in BETA_SWEEP),
    ]
    for omega in OMEGA_SWEEP:
        row = [reform_cost(df, omega, beta, "raise100k")["cost"]
               for beta in BETA_SWEEP]
        lines.append(
            f"  {omega:<14.2f}" + "".join(f"{_bn(c):>12.3f}" for c in row)
        )
    # Sign sanity-check.
    lo = reform_cost(df, 0.20, 0.90, "raise100k")["cost"]
    lines += [
        "  (costs in GBP bn; negative = revenue LOSS)",
        "",
        "SIGN SANITY-CHECK (raise threshold to GBP100k):",
        "  At omega=0.20, beta=0.90 (consumer-facing, low input share) the cost is",
        f"  GBP {_bn(lo):.3f} bn -- a revenue LOSS, as expected: a firm in [85k,100k)",
        "  that de-registers flips from yielding t*(1-omega)*y to t*omega*y, and with",
        "  omega<0.5 we have t*omega*y < t*(1-omega)*y, so government revenue falls.",
        "  For input-heavy firms (omega>0.5) the comparison reverses (or they",
        "  voluntarily register and are unaffected).",
    ]
    return lines


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(df, name="liu_costing.png"):
    """Two-panel figure: reform cost vs omega; raise-100k cost over (omega,beta)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    omega_grid = np.linspace(0.10, 0.90, 33)
    rep_beta = 0.50

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))

    # Panel 1: three reforms vs omega at representative beta.
    colours = {"raise100k": PALETTE[0], "rate10": PALETTE[3], "rate15": PALETTE[1]}
    labels = {"raise100k": r"raise threshold $\to$ £100k",
              "rate10": "reduced band 10%", "rate15": "reduced band 15%"}
    for reform in REFORMS:
        costs = [_bn(reform_cost(df, om, rep_beta, reform)["cost"])
                 for om in omega_grid]
        ax1.plot(omega_grid, costs, color=colours[reform], lw=2.4,
                 label=labels[reform])
    ax1.axhline(0.0, color=ACCENT, lw=1.2, ls="--", alpha=0.7)
    ax1.set_title(rf"Reform cost vs input share ($\beta$={rep_beta})", fontsize=12)
    ax1.set_xlabel(r"Input share $\omega$", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Cost to government (£bn)", fontsize=LABEL_SIZE)
    ax1.legend(frameon=False, fontsize=TICK_SIZE, loc="best")
    _style_ax(ax1)

    # Panel 2: raise-100k cost over omega for each beta.
    beta_cols = [PALETTE[0], PALETTE[3], PALETTE[1]]
    for beta, col in zip(BETA_SWEEP, beta_cols):
        costs = [_bn(reform_cost(df, om, beta, "raise100k")["cost"])
                 for om in omega_grid]
        ax2.plot(omega_grid, costs, color=col, lw=2.4,
                 label=rf"$\beta$={beta:.1f}")
    ax2.axhline(0.0, color=ACCENT, lw=1.2, ls="--", alpha=0.7)
    ax2.set_title(r"Raise-to-£100k cost across $(\omega,\beta)$", fontsize=12)
    ax2.set_xlabel(r"Input share $\omega$", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Cost to government (GBP bn)", fontsize=LABEL_SIZE)
    ax2.legend(frameon=False, fontsize=TICK_SIZE, loc="best")
    _style_ax(ax2)

    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def main(figure=True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_reform_data()
    lines = build_report(df)
    txt = RESULTS_DIR / "liu_costing.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {txt}")
    if figure:
        png = make_figure(df)
        print(f"wrote {png}")


def cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="firm-microsim-liu-costing",
        description=(
            "Parametric static extensive-margin VAT reform costing in the Liu et "
            "al. (2021) setting, where registration is a choice driven by input "
            "share omega and B2C share beta. Costs a raised threshold (GBP100k) "
            "and reduced-rate bands (10%/15%) and sweeps (omega, beta) to give a "
            "sensitivity range. Writes results/liu_costing.txt and "
            "results/liu_costing.png (also copied to paper/figures/)."
        ),
    )
    parser.add_argument(
        "--no-figure", action="store_true", help="write the text report only"
    )
    args = parser.parse_args(argv)
    main(figure=not args.no_figure)


if __name__ == "__main__":
    cli()
