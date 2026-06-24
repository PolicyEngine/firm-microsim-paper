#!/usr/bin/env python3
"""
Reform-menu costs on a SINGLE COMMON BASE (tab:schedule_costs).
================================================================

Puts the ENTIRE schedule-reform menu on the SAME basis:
    * data year   : 2023-24 microdata, UNAGED
    * threshold   : the GBP85,000 notch
    * VAT base    : GBP179.3bn (weighted sum of liabilities over firms >= 85k)

Four reforms, all anchored at GBP85k / GBP179.3bn:
    1. Raise threshold to GBP100,000          (LOCATION)   -- recomputed here
    2. Graduated taper [85k, 105k]            (SHAPE)      -- from taper_reform.py
    3. Banded reduced rate 10% [85k, 105k]    (RATE)       -- from rate_reform.py
    4. Banded reduced rate 15% [85k, 105k]    (RATE)       -- from rate_reform.py

The taper / reduced-rate static numbers are already on the GBP85k/GBP179.3bn
unaged base (they read the same productivity dataset and difference against the
85k registered base). This script verifies the base and firm-in-band count, and
RECOMPUTES the threshold-relocation reform as GBP85k -> GBP100k on the SAME
base, replacing the old GBP90k -> GBP100k aged-to-2025-26 figure (-374m).

Two costings of the relocation are reported:
    (A) DIRECT band-sum: revenue lost = sum of weighted liabilities of firms
        with turnover in [85k, 100k). This is the clean object here, because the
        released band sits ABOVE the GBP85k registration threshold, so the band
        is cleanly populated with already-registered firms (the same reasoning
        as StaticVATModel.anchor_reform: no de-bunching needed above 85k).
    (B) SMOOTH counterfactual: the static-sweep method
        (StaticVATModel._counterfactual_bins) fits the clean above-threshold
        per-GBP1k profile and integrates it over [85k, 100k). Reported for
        comparability with the GBP90k sweep table; it differs trivially from (A)
        because the band is already clean above 85k.

We adopt (A) as the headline relocation figure for the common-base menu.

Run:  python3 analysis/reform_menu_common_base.py
Writes: results/reform_menu_common_base.txt
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# --- locate the recovered-productivity dataset (same as taper/rate scripts) ---
DATA_CANDIDATES = [
    "/Users/janansadeqian/PolicyEngine_VATLab/analysis/synthetic_firms_with_productivity.csv",
    "/Users/janansadeqian/uk-vatlab/analysis/synthetic_firms_with_productivity.csv",
]
OUT = "/Users/janansadeqian/firm-microsim-paper/results/reform_menu_common_base.txt"

T_STAR = 85000.0          # GBP85k notch baseline (common base)
T_NEW = 100000.0          # raise-to threshold
BAND_TOP = 105000.0       # taper / reduced-rate band top


def smooth_counterfactual_release(tk_k, liab_pounds, w, baseline_k, new_k,
                                  bin_k=1.0, lo_k=60.0, hi_k=160.0,
                                  fit_pad_k=2.0, fit_top_k=150.0, degree=1):
    """Smooth-counterfactual release of the [baseline, new) band, mirroring
    StaticVATModel._counterfactual_bins + threshold_sweep (UNAGED).

    Fits the clean above-threshold per-GBP1k profile of VAT-paying firms and of
    liability and integrates the fit over bins in [baseline, new)."""
    edges = np.arange(lo_k, hi_k + bin_k, bin_k)
    centres = (edges[:-1] + edges[1:]) / 2.0
    paying = liab_pounds > 0
    firms, _ = np.histogram(tk_k[paying], bins=edges, weights=w[paying])
    liab_bin, _ = np.histogram(tk_k, bins=edges, weights=liab_pounds * w)
    fit = (centres >= baseline_k + fit_pad_k) & (centres <= fit_top_k)
    cf_firms = np.clip(np.polyval(np.polyfit(centres[fit], firms[fit], degree), centres), 0.0, None)
    cf_liab = np.polyval(np.polyfit(centres[fit], liab_bin[fit], degree), centres)
    band = (centres >= baseline_k) & (centres < new_k)
    return -cf_liab[band].sum(), -cf_firms[band].sum()


def main():
    data = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    if data is None:
        raise FileNotFoundError("synthetic_firms_with_productivity.csv not found")

    df = pd.read_csv(data, usecols=["annual_turnover_k", "vat_liability_k", "weight"])
    tk = df["annual_turnover_k"].to_numpy()          # GBP thousand
    t = tk * 1000.0                                   # GBP
    liab = df["vat_liability_k"].to_numpy() * 1000.0  # GBP
    w = df["weight"].to_numpy()

    # --- verify the common base ------------------------------------------
    reg = t >= T_STAR
    base_bn = (liab[reg] * w[reg]).sum() / 1e9
    firms_band_85_105 = w[(t >= T_STAR) & (t < BAND_TOP)].sum()

    # --- (A) DIRECT relocation 85k -> 100k -------------------------------
    rel_band = (t >= T_STAR) & (t < T_NEW)
    rev_direct_m = -(liab[rel_band] * w[rel_band]).sum() / 1e6
    firms_direct_k = -w[rel_band].sum() / 1000.0

    # --- (B) SMOOTH counterfactual relocation 85k -> 100k ----------------
    rev_smooth, firms_smooth = smooth_counterfactual_release(
        tk, liab, w, baseline_k=T_STAR / 1000.0, new_k=T_NEW / 1000.0)
    rev_smooth_m = rev_smooth / 1e6
    firms_smooth_k = firms_smooth / 1000.0

    pct_direct = rev_direct_m / (base_bn * 1000.0) * 100.0

    lines = []
    p = lines.append
    p("=" * 72)
    p("REFORM MENU ON A SINGLE COMMON BASE  (tab:schedule_costs)")
    p("Base: GBP85k notch, 2023-24 microdata, UNAGED")
    p("=" * 72)
    p(f"Dataset            : {data}")
    p(f"VAT base (>=85k)   : GBP {base_bn:.3f} bn   (target ~GBP179.3bn)")
    p(f"Firms in [85k,105k): {firms_band_85_105:,.0f}   (~125,600)")
    p("")
    p("--- Threshold relocation GBP85k -> GBP100k (recomputed) -------------")
    p(f"  (A) DIRECT band-sum [85k,100k) :  {rev_direct_m:+.1f} m   "
      f"firms {firms_direct_k:+.1f} (000s)")
    p(f"      as % of GBP179.3bn base    :  {pct_direct:+.3f}%")
    p(f"  (B) SMOOTH counterfactual      :  {rev_smooth_m:+.1f} m   "
      f"firms {firms_smooth_k:+.1f} (000s)")
    p("  HEADLINE (adopt A)             :  "
      f"{rev_direct_m:+.0f} m   firms {firms_direct_k:+.1f} (000s)")
    p("")
    p("--- Schedule reforms (verified on the SAME GBP85k/GBP179.3bn base) --")
    p("  Graduated taper [85k,105k]     :  -335 m   "
      f"(125,608 firms lower rate)   [taper_reform.py: static_change_m=-335.2]")
    p("  Reduced rate 10% [85k,105k]    :  -357 m   "
      f"(125,608 firms)              [rate_reform.py]")
    p("  Reduced rate 15% [85k,105k]    :  -178 m   "
      f"(125,608 firms)              [rate_reform.py]")
    p("")
    p("=" * 72)
    p("CORRECTED tab:schedule_costs  (all on GBP85k / GBP179.3bn, 2023-24 unaged)")
    p("=" * 72)
    p(f"{'Reform':<42}{'Lever':<18}{'Static (GBPm)':>14}")
    p("-" * 74)
    p(f"{'Raise threshold to GBP100,000':<42}{'Location':<18}{round(rev_direct_m):>14d}")
    p(f"{'Graduated taper [85k,105k]':<42}{'Shape (phase-in)':<18}{-335:>14d}")
    p(f"{'Reduced rate 10% [85k,105k]':<42}{'Rate (step)':<18}{-357:>14d}")
    p(f"{'Reduced rate 15% [85k,105k]':<42}{'Rate (step)':<18}{-178:>14d}")
    p("=" * 72)
    p("")
    p("Firm-in-band counts:")
    p(f"  Raise-to-100k band [85k,100k): {-firms_direct_k*1000:,.0f} firms released")
    p(f"  Taper / reduced-rate band [85k,105k): {firms_band_85_105:,.0f} firms")

    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(text + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
