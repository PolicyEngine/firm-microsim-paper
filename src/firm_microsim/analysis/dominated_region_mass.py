"""Empirical mass in the VAT-notch DOMINATED REGION.

Gives the dominated-region arithmetic identity empirical bite by quantifying the
WEIGHTED firm mass it concerns:

  * OBSERVED mass        -- weighted firms the data place in [T*, T*+a];
  * COUNTERFACTUAL mass  -- weighted firms the mass-conserving "no-bunching"
                            counterfactual density places in [T*, T*+a], i.e.
                            how many firms WOULD locate in the dominated region
                            absent the notch (the displaced mass the region
                            concerns).

The dominated-region width is the exact Kleven-Waseem ``a = T* * tau/(1-tau)``
from ``notch/model.py``. We also report the analogous masses for the
reform-shrunk bands at lower headline rates (15% -> width £15,000, 10% ->
width £9,444), and sanity-check the counterfactual band mass against the paper's
excess-mass E and mass-conservation geometry recovered from ``bunching/model.py``.

Run:  firm-microsim-dominated-region
Out:  results/dominated_region_mass.txt
"""

from __future__ import annotations

import argparse

import numpy as np

# Reuse the authoritative model code so definitions never drift.
from firm_microsim.bunching.model import (
    BunchingEstimator,
    BIN_WIDTH,
    bin_density,
    fit_counterfactual,
    DEFAULT_DEGREE,
    DEFAULT_WINDOW,
)
from firm_microsim.config import RESULTS_DIR
from firm_microsim.notch.model import NotchModel, TAU

VINTAGE = "2023-24"          # the £85k baseline used by the paper
RESULTS = RESULTS_DIR / "dominated_region_mass.txt"

# Headline-rate variants and the dominated-region width each implies:
#   a(tau) = T* * tau / (1 - tau)
#   20% -> 85 * 0.20/0.80 = 21.250 (£21,250)  [baseline notch]
#   15% -> 85 * 0.15/0.85 = 15.000 (£15,000)
#   10% -> 85 * 0.10/0.90 =  9.444 (£ 9,444)
RATE_VARIANTS = [
    ("20% (baseline notch)", 0.20),
    ("15% band", 0.15),
    ("10% band", 0.10),
]

# Reduced-rate band runs [T*, T*+BAND_WIDTH]; at the band top it reverts to the
# standard rate tau, creating a SECOND notch whose dominated width is
#   a' = T1 * (tau - r) / (1 - tau),  T1 = T* + BAND_WIDTH.
REDUCED_RATE_BAND_WIDTH = 20.0  # £k (band [85k, 105k] in the paper)


def mass_in_band(centres, density, lo, hi, bin_width=BIN_WIDTH):
    """Integrate a binned density over [lo, hi) -> weighted firm count.

    Each bin spans [centre - bin_width/2, centre + bin_width/2); its
    contribution is its density times its overlap with [lo, hi), so partial
    bins at the band edges count fractionally. (An earlier version used
    bin-centre membership, which shifted every band by half a bin.)
    """
    left = centres - bin_width / 2.0
    right = centres + bin_width / 2.0
    overlap = np.clip(np.minimum(right, hi) - np.maximum(left, lo), 0.0, None)
    return float(np.sum(density * overlap / bin_width))


def mass_in_band_exact(turnover, weight, lo, hi):
    """Weighted firm mass with turnover in [lo, hi), exact microdata masks."""
    import numpy as _np
    m = (turnover >= lo) & (turnover < hi)
    return float(_np.sum(weight[m]))


def main() -> None:
    t_star = float(NotchModel(VINTAGE).t_star)  # 85.0 (£k), from VINTAGES config

    # --- Build observed + mass-conserving counterfactual densities ----------
    # Exposure counts are for firms that face the notch: in-scope VAT firms
    # (issue #37). Out-of-scope enterprises are not liable at any turnover.
    est = BunchingEstimator(VINTAGE, scope_only=True)
    res = est.estimate()  # reduced-form solve on the in-scope density
    centres = res["centres"]
    f_obs = res["f_obs"]
    f_cf = res["f_cf"]
    E = res["E"]            # excess mass below T* (weighted firms)
    y_R = res["y_R"]        # endogenous marginal buncher (£k)
    Delta_R = res["Delta_R"]

    # Cross-check the counterfactual is the one used internally (same fn/args).
    c2, fobs2 = bin_density(
        est.firms["annual_turnover_k"].to_numpy(),
        est.firms["weight"].to_numpy(),
    )
    fcf2 = fit_counterfactual(c2, fobs2, t_star, DEFAULT_DEGREE, DEFAULT_WINDOW, DEFAULT_WINDOW)
    assert np.allclose(f_cf, fcf2), "counterfactual density mismatch"

    total_obs_mass = float(np.sum(f_obs) * BIN_WIDTH)

    # --- Per-variant dominated-region masses --------------------------------
    rows = []
    tk = est.firms["annual_turnover_k"].to_numpy()
    wt = est.firms["weight"].to_numpy()
    for label, tau in RATE_VARIANTS:
        a = t_star * tau / (1.0 - tau)          # Kleven-Waseem width (£k)
        lo, hi = t_star, t_star + a
        obs = mass_in_band_exact(tk, wt, lo, hi)
        cf = mass_in_band(centres, f_cf, lo, hi)
        rows.append({
            "label": label,
            "tau": tau,
            "a": a,
            "lo": lo,
            "hi": hi,
            "obs": obs,
            "cf": cf,
            "net": cf - obs,   # net missing (displaced) mass in band
        })

    # --- Secondary dominated region at the reduced-rate band top ------------
    # A banded reduced rate reverts to the standard rate tau at the band top
    # T1, creating a SECOND notch with dominated width a' = T1*(tau-r)/(1-tau).
    # The total dominated turnover under a reduced rate is primary + secondary.
    band_top = t_star + REDUCED_RATE_BAND_WIDTH      # £105k
    sec_rows = []
    for label, tau_r in RATE_VARIANTS[1:]:           # 15% and 10% only
        a_sec = band_top * (TAU - tau_r) / (1.0 - TAU)
        lo, hi = band_top, band_top + a_sec
        obs = mass_in_band_exact(tk, wt, lo, hi)
        cf = mass_in_band(centres, f_cf, lo, hi)
        prim = next(r for r in rows if r["tau"] == tau_r)
        sec_rows.append({
            "label": label, "tau": tau_r, "a_sec": a_sec,
            "lo": lo, "hi": hi, "obs": obs, "cf": cf,
            "prim_obs": prim["obs"], "total_obs": prim["obs"] + obs,
        })

    # Baseline (20%) is the paper's actual dominated region.
    base = rows[0]

    # --- Context shares -----------------------------------------------------
    # "All firms near the threshold": below-window + above up to marginal buncher.
    near_lo = t_star - DEFAULT_WINDOW
    near_hi = max(y_R, base["hi"])
    near_mass_obs = mass_in_band(centres, f_obs, near_lo, near_hi)
    cf_share_of_near = base["cf"] / near_mass_obs if near_mass_obs else float("nan")

    # Missing mass above T* (cf - obs) within the baseline dominated band:
    # this is the directly displaced mass the dominated region concerns.
    missing_in_band = base["cf"] - base["obs"]

    # --- Write report -------------------------------------------------------
    lines = []
    W = lines.append
    W("=" * 74)
    W("EMPIRICAL MASS IN THE VAT-NOTCH DOMINATED REGION")
    W(f"vintage = {VINTAGE}   T* = GBP {t_star*1000:,.0f}   tau = {TAU:.2f}")
    W("=" * 74)
    W("")
    W("Dominated region (Kleven-Waseem):  a = T* * tau/(1-tau)")
    W("Bands measured on the WEIGHTED in-scope (VAT-liable) synthetic firms;")
    W("  out-of-scope PAYE-only/exempt enterprises face no notch and are excluded.")
    W("  OBS uses exact band masks on the microdata; CF integrates the binned")
    W("  counterfactual with fractional edge-bin overlap.")
    W("  OBS = observed weighted firms in band")
    W("  CF  = weighted firms the mass-conserving no-bunching counterfactual")
    W("        density places in band  (total smooth density across the band)")
    W("  NET = CF - OBS: positive = mass the counterfactual places in the band")
    W("        beyond what is observed (missing/displaced); NEGATIVE = observed")
    W("        surplus over the smooth counterfactual (no displacement read).")
    W("")
    hdr = (f"{'rate / band':<22}{'width a (GBP)':>14}{'band (GBP k)':>18}"
           f"{'OBS':>12}{'CF':>12}{'NET disp.':>12}")
    W(hdr)
    W("-" * len(hdr))
    for r in rows:
        band = f"[{r['lo']:.0f}, {r['hi']:.3f})"
        W(f"{r['label']:<22}{r['a']*1000:>14,.0f}{band:>18}"
          f"{r['obs']:>12,.0f}{r['cf']:>12,.0f}{r['net']:>12,.0f}")
    W("-" * len(hdr))
    W("")
    W("INTERPRETATION (baseline 20% notch, band [85,000, 106,250)):")
    W(f"  Observed firms in dominated region ........ {base['obs']:>12,.0f}")
    W(f"  Counterfactual total density in band (CF).. {base['cf']:>12,.0f}")
    W(f"  NET displaced mass in band (CF - OBS) ..... {missing_in_band:>12,.0f}")
    W(f"  CF mass as share of firms near threshold .. {cf_share_of_near:>12.3f}")
    W("")
    W("  Note: the headline displaced-mass number is NET (CF - OBS), not the")
    W("  total CF density. The band is wide, so most CF firms in it would")
    W("  relocate to OTHER registered turnover levels, not disappear; only the")
    W("  NET deficit is the mass the notch evacuates from the dominated region.")
    W("")
    W("REFORM-SHRUNK BANDS (net displaced mass):")
    W(f"  15% -> band width GBP {rows[1]['a']*1000:,.0f}: net displaced = {rows[1]['net']:,.0f} firms"
      f"  (CF total {rows[1]['cf']:,.0f})")
    W(f"  10% -> band width GBP {rows[2]['a']*1000:,.0f}: net displaced = {rows[2]['net']:,.0f} firms"
      f"  (CF total {rows[2]['cf']:,.0f})")
    W("")
    W(f"SECONDARY NOTCH at the reduced-rate band top (T1 = GBP {band_top*1000:,.0f}):")
    W("  A banded reduced rate reverts to tau=20% at T1, adding a SECOND")
    W("  dominated region a' = T1*(tau-r)/(1-tau). Total dominated turnover and")
    W("  mass = primary [T*, T*+a] + secondary [T1, T1+a'].")
    for s in sec_rows:
        W(f"  {s['label']:<9} secondary [{s['lo']:.0f}, {s['hi']:.3f}) "
          f"width GBP {s['a_sec']*1000:,.0f}  OBS = {s['obs']:,.0f}")
        W(f"            primary OBS {s['prim_obs']:,.0f} + secondary OBS {s['obs']:,.0f}"
          f" = TOTAL {s['total_obs']:,.0f}  (baseline {base['obs']:,.0f},"
          f" {100*(s['total_obs']/base['obs']-1):+.1f}%)")
    W("")
    W("REDUCED-FORM BUNCHING on this population (context for the masses above):")
    W(f"  excess mass below T*       E       = {E:,.0f} firms")
    W(f"  missing mass above T*      Delta_R = {Delta_R:,.0f} firms")
    W(f"  marginal buncher           y_R     = GBP {y_R*1000:,.0f}")
    W(f"  NET displaced mass in 20% dominated band = {missing_in_band:,.0f} firms")
    W("")
    if E < 100:
        W("  The estimator finds NO excess mass below the threshold on this")
        W("  population (E ~ 0): the corrected net-liability calibration produces")
        W("  no synthetic bunching, so the dominated-region masses above are")
        W("  TARGET-BAND GEOMETRY (weighted firms located in each band), not")
        W("  behavioural displacement. The band masses answer 'how many weighted")
        W("  firms sit where the schedule makes location dominated', which is the")
        W("  policy-relevant exposure count for each reform variant.")
    else:
        W("  By mass conservation the excess mass E that bunches just below T*")
        W("  is the mass that, absent the notch, would have spread into the")
        W("  region above T*. Compare magnitudes of E, Delta_R, and the NET band")
        W("  mass; the wide band also captures smooth-density deficit beyond y_R.")
        W("  (The TOTAL CF density in the band is NOT the displaced mass and")
        W("  should not be compared to E -- that would be a category error.)")
    W("")
    W(f"  total observed weighted mass on [{est.firms['annual_turnover_k'].min():.0f},"
      f"{est.firms['annual_turnover_k'].max():.0f}] est. range = {total_obs_mass:,.0f}")
    W("")
    W("script: firm-microsim-dominated-region")
    text = "\n".join(lines)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(text + "\n")
    print(text)
    print(f"\n[written] {RESULTS}")


def cli(argv: list[str] | None = None) -> None:
    """Console entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    main()


if __name__ == "__main__":
    cli()
