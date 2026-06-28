"""TASK 2: Fiscal-drag projection under a frozen £90,000 threshold.

The VAT threshold is frozen at £90,000. As nominal turnover grows year by year
(cumulative growth factors from static/model.py FISCAL_YEARS), more firms are
drawn toward and over the threshold. We project, for each fiscal year:
  - weighted firms in the £90k dominated region [£90,000, £112,500]
    (a = £90k·0.2/0.8 = £22,500);
  - weighted firms just below the threshold [£85k,£90k] aged (bunching-exposed);
and report how these counts GROW (the fiscal-drag effect).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from firm_microsim.config import SYNTHETIC_DATA_DIR, RESULTS_DIR
DATA = str(SYNTHETIC_DATA_DIR / "synthetic_firms_2023-24.csv")
OUT = str(RESULTS_DIR / "fiscal_drag_projection.txt")

T = 90_000.0
TAU = 0.20
A = T * TAU / (1.0 - TAU)        # £22,500
DOM_HI = T + A                   # £112,500

# Cumulative nominal-growth factors from static/model.py FISCAL_YEARS.
FISCAL_YEARS = [
    {"year": "2024-25", "firm_growth": 1.0310},
    {"year": "2025-26", "firm_growth": 1.0516},
    {"year": "2026-27", "firm_growth": 1.0779},
    {"year": "2027-28", "firm_growth": 1.1102},
    {"year": "2028-29", "firm_growth": 1.1424},
]


def wcount(t, w, lo, hi):
    m = (t >= lo) & (t < hi)
    return float(np.sum(w[m]))


def main():
    df = pd.read_csv(DATA, usecols=["annual_turnover_k", "weight"])
    t0 = df["annual_turnover_k"].to_numpy(float) * 1000.0
    w = df["weight"].to_numpy(float)

    lines = []
    P = lines.append
    P("=" * 84)
    P("TASK 2 — FISCAL-DRAG PROJECTION UNDER A FROZEN £90,000 THRESHOLD")
    P("=" * 84)
    P("")
    P(f"  Threshold frozen at £{T:,.0f}.  Dominated region a = T·0.2/0.8 = £{A:,.0f}")
    P(f"  Dominated band [£{T:,.0f}, £{DOM_HI:,.0f}).  Just-below band [£85,000, £90,000).")
    P("  Turnover aged each year by the cumulative nominal-growth factor.")
    P("")
    hdr = f"  {'year':<9}{'growth':>8}{'dom-region firms':>20}{'just-below firms':>20}"
    P(hdr)
    P("  " + "-" * (len(hdr) - 2))

    rows = []
    for fy in FISCAL_YEARS:
        g = fy["firm_growth"]
        t = t0 * g
        dom = wcount(t, w, T, DOM_HI)
        below = wcount(t, w, 85_000.0, 90_000.0)
        rows.append((fy["year"], g, dom, below))
        P(f"  {fy['year']:<9}{g:>8.4f}{dom:>20,.0f}{below:>20,.0f}")

    P("")
    P("  GROWTH over the projection (vs first year 2024-25):")
    P("  " + "-" * 60)
    dom0, below0 = rows[0][2], rows[0][3]
    P(f"  {'year':<9}{'dom %Δ vs 24-25':>22}{'below %Δ vs 24-25':>22}")
    for yr, g, dom, below in rows:
        P(f"  {yr:<9}{(dom/dom0-1)*100:>21.2f}%{(below/below0-1)*100:>21.2f}%")

    P("")
    dom_last, below_last = rows[-1][2], rows[-1][3]
    P(f"  Dominated-region population grows from {dom0:,.0f} (2024-25) to "
      f"{dom_last:,.0f} (2028-29): +{(dom_last/dom0-1)*100:.1f}%.")
    P(f"  Bunching-exposed (just-below) population grows from {below0:,.0f} to "
      f"{below_last:,.0f}: +{(below_last/below0-1)*100:.1f}%.")
    P("  This is the fiscal-drag effect of freezing the threshold in nominal terms.")

    txt = "\n".join(lines) + "\n"
    with open(OUT, "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
