"""Stratum-shape sensitivity: exponential (headline) vs lognormal unregistered stratum.

BPE publishes counts and turnover totals for unregistered businesses but no
size distribution (it imputes their turnover from zero-employee registered
businesses at sector level). The headline draw is exponential with the
division mean; this script compares it with a mean-preserving lognormal
(sigma = 1.0) on the objects the stratum drives: the implied number of
unregistered businesses above the threshold (exempt, out of scope), the
below-threshold density near the threshold, the threshold-cut rows of the
sweep, and the anchor (which the stratum enters only through the frame's
residual identification).

Requires ``data/synthetic/synthetic_firms_2023-24.csv`` (exponential) and
``data/synthetic/synthetic_firms_2023-24_lognormal.csv`` (built with
``firm-microsim --vintage 2023-24 --unregistered-shape lognormal --output
synthetic_firms_2023-24_lognormal.csv``). Writes results/stratum_shape_sensitivity.txt.
"""
from __future__ import annotations

import pandas as pd

from firm_microsim.config import RESULTS_DIR, SYNTHETIC_DATA_DIR
from firm_microsim.static.model import StaticVATModel

OUT = RESULTS_DIR / "stratum_shape_sensitivity.txt"
FILES = {
    "exponential (headline)": SYNTHETIC_DATA_DIR / "synthetic_firms_2023-24.csv",
    "lognormal (sigma=1.0)": SYNTHETIC_DATA_DIR / "synthetic_firms_2023-24_lognormal.csv",
}


def _model(path):
    m = StaticVATModel.__new__(StaticVATModel)
    m.vintage = "2023-24"
    m.data_threshold = 85_000.0
    df = pd.read_csv(path, usecols=["annual_turnover_k", "vat_liability_k", "weight",
                                    "vat_scope", "vat_registered", "unregistered", "in_frame"])
    reg = df["vat_registered"].astype(bool)
    df["voluntary"] = reg & (df["annual_turnover_k"] * 1000.0 <= m.data_threshold)
    df["mandatory"] = reg & (df["annual_turnover_k"] * 1000.0 > m.data_threshold)
    m.firms = df
    return m, df


def main() -> None:
    lines = ["STRATUM-SHAPE SENSITIVITY (2023-24 vintage)", "=" * 70, ""]
    W = lines.append
    for label, path in FILES.items():
        if not path.exists():
            W(f"[{label}] {path.name} not found - build it first (see module docstring)")
            continue
        m, df = _model(path)
        u = df[df["unregistered"].astype(bool)]
        t = u["annual_turnover_k"]
        W(f"[{label}]")
        W(f"  stratum: {len(u):,} businesses; mean turnover GBP {t.mean()*1000:,.0f}; "
          f"median GBP {t.median()*1000:,.0f}; above GBP85k (exempt, out of scope): "
          f"{u['weight'][t > 85].sum():,.0f}; within GBP70k-85k: {u['weight'][(t > 70) & (t <= 85)].sum():,.0f}")
        all_t = df["annual_turnover_k"]
        for lo in (70, 80, 84):
            mm = (all_t > lo) & (all_t <= lo + 1)
            W(f"  density bin {lo}k: all {df['weight'][mm].sum():,.0f}  frame "
              f"{df['weight'][mm & df['in_frame']].sum():,.0f}  stratum {df['weight'][mm & df['unregistered']].sum():,.0f}")
        sw = m.threshold_sweep(year="2025-26", baseline=85_000)
        for _, r in sw[sw["threshold_k"].isin([70.0, 75.0, 80.0])].iterrows():
            W(f"  cut to {r['threshold_k']:.0f}k from 85k (2025-26): {r['revenue_change_m']:+,.1f} m, "
              f"{r['firms_change_k']:+,.1f}k firms")
        anc = m.anchor_reform()
        W("  anchor: " + ", ".join(f"{v:+.0f}" for v in anc["policyengine_impact_m"]))
        W("")
    text = "\n".join(lines)
    OUT.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
