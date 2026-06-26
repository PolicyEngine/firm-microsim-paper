#!/usr/bin/env python3
"""
Reform-menu costs on a SINGLE COMMON BASE (tab:schedule_costs).
================================================================

Puts the ENTIRE schedule-reform menu on the SAME basis:
    * data year   : 2023-24 microdata, UNAGED
    * threshold   : the GBP85,000 notch
    * VAT base    : computed from the repo-generated 2023-24 synthetic data

Four reforms, all anchored at the repo-generated GBP85k baseline:
    1. Raise threshold to GBP100,000          (LOCATION)   -- recomputed here
    2. Graduated taper [85k, 105k]            (SHAPE)      -- recomputed here
    3. Banded reduced rate 10% [85k, 105k]    (RATE)       -- recomputed here
    4. Banded reduced rate 15% [85k, 105k]    (RATE)       -- recomputed here

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

Run:  firm-microsim-reform-menu
Writes: results/reform_menu_common_base.txt
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from firm_microsim.config import REPO_ROOT, RESULTS_DIR, SYNTHETIC_DATA_DIR
from firm_microsim.dynamic.model import E_HEADLINE, build_reforms, reform_revenue

DATA_CANDIDATES = [
    SYNTHETIC_DATA_DIR / "synthetic_firms_2023-24.csv",
]
OUT = RESULTS_DIR / "reform_menu_common_base.txt"

T_STAR = 85000.0          # GBP85k notch baseline (common base)
T_NEW = 100000.0          # raise-to threshold
BAND_TOP = 105000.0       # taper / reduced-rate band top

TABLE_LABELS = {
    "raise100k": "Raise threshold to GBP100,000",
    "taper": "Graduated taper [85k,105k]",
    "rate10": "Reduced rate 10% [85k,105k]",
    "rate15": "Reduced rate 15% [85k,105k]",
}
LEVERS = {
    "raise100k": "Location",
    "taper": "Shape (phase-in)",
    "rate10": "Rate (step)",
    "rate15": "Rate (step)",
}


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


def display_path(path):
    """Return a stable repo-relative path when possible."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def main():
    data = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data is None:
        raise FileNotFoundError(
            "synthetic_firms_2023-24.csv not found - generate it first with:\n"
            "  firm-microsim --vintage 2023-24 --output synthetic_firms_2023-24.csv"
        )

    df = pd.read_csv(data, usecols=["annual_turnover_k", "vat_liability_k", "weight"])
    tk = df["annual_turnover_k"].to_numpy()          # GBP thousand
    t = tk * 1000.0                                   # GBP
    liab = df["vat_liability_k"].to_numpy() * 1000.0  # GBP
    w = df["weight"].to_numpy()
    reform_df = pd.DataFrame({"turnover": t, "liab": liab, "weight": w})

    # --- verify the common base ------------------------------------------
    reg = t >= T_STAR
    base_bn = (liab[reg] * w[reg]).sum() / 1e9
    firms_band_85_105 = w[(t >= T_STAR) & (t < BAND_TOP)].sum()

    # --- Compute every static reform through the shared dynamic schedule code.
    static_rows = {}
    for key, (schedule, _label) in build_reforms().items():
        result = reform_revenue(reform_df, schedule, E_HEADLINE, behavioural=False)
        static_rows[key] = {
            "result": result,
            "cost_m": result["d_rev"] / 1e6,
            "firms_k": -result["n_affected"] / 1000.0,
        }

    # --- (A) DIRECT relocation 85k -> 100k -------------------------------
    rel_band = (t >= T_STAR) & (t < T_NEW)
    rev_direct_m = -(liab[rel_band] * w[rel_band]).sum() / 1e6
    firms_direct_k = -w[rel_band].sum() / 1000.0
    if abs(rev_direct_m - static_rows["raise100k"]["cost_m"]) > 1e-6:
        raise AssertionError("raise-to-100k direct sum and shared static cost diverged")

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
    p(f"Dataset            : {display_path(data)}")
    p(f"VAT base (>=85k)   : GBP {base_bn:.3f} bn")
    p(f"Firms in [85k,105k): {firms_band_85_105:,.0f}")
    p("")
    p("--- Threshold relocation GBP85k -> GBP100k (recomputed) -------------")
    p(f"  (A) DIRECT band-sum [85k,100k) :  {rev_direct_m:+.1f} m   "
      f"firms {firms_direct_k:+.1f} (000s)")
    p(f"      as % of VAT base           :  {pct_direct:+.3f}%")
    p(f"  (B) SMOOTH counterfactual      :  {rev_smooth_m:+.1f} m   "
      f"firms {firms_smooth_k:+.1f} (000s)")
    p("  HEADLINE (adopt A)             :  "
      f"{rev_direct_m:+.0f} m   firms {firms_direct_k:+.1f} (000s)")
    p("")
    p("--- Schedule reforms (computed on the SAME repo-generated base) -----")
    for key in ["taper", "rate10", "rate15"]:
        row = static_rows[key]
        p(f"  {TABLE_LABELS[key]:<33}:  {row['cost_m']:+.1f} m   "
          f"affected firms {row['firms_k']:+.1f} (000s)")
    p("")
    p("=" * 72)
    p("CORRECTED tab:schedule_costs  (repo-generated GBP85k, 2023-24 unaged)")
    p("=" * 72)
    p(f"{'Reform':<42}{'Lever':<18}{'Static (GBPm)':>14}")
    p("-" * 74)
    for key in ["raise100k", "taper", "rate10", "rate15"]:
        p(f"{TABLE_LABELS[key]:<42}{LEVERS[key]:<18}"
          f"{round(static_rows[key]['cost_m']):>14d}")
    p("=" * 72)
    p("")
    p("Firm-in-band counts:")
    p(f"  Raise-to-100k band [85k,100k): {-firms_direct_k*1000:,.0f} firms released")
    p(f"  Taper / reduced-rate band [85k,105k): {firms_band_85_105:,.0f} firms")

    text = "\n".join(lines)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n")
    print(f"\nWrote {OUT}")


def cli(argv: list[str] | None = None) -> None:
    """Console entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    main()


if __name__ == "__main__":
    cli()
