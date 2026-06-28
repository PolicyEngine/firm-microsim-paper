"""TASK 1: Value-added correction to the VAT dominated region.

The textbook Kleven-Waseem dominated region a = T*·tau/(1-tau) = £21,250 assumes
VAT is a tax on the WHOLE turnover (tau=0.20). Real VAT is on VALUE ADDED: a firm
that remits NET rate tau0 = liab/turnover (output VAT minus input credits, ~3% of
turnover on average) faces a value-added dominated region a_i = T*·tau0/(1-tau0),
which is far smaller. This script computes the firm-level distribution of a_i for
near-threshold firms.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from firm_microsim.config import SYNTHETIC_DATA_DIR, RESULTS_DIR
DATA = str(SYNTHETIC_DATA_DIR / "synthetic_firms_2023-24.csv")
OUT = str(RESULTS_DIR / "value_added_dominated_region.txt")

T_STAR = 85_000.0
TAU = 0.20


def wquantile(x, w, q):
    """Weighted quantile(s)."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return np.interp(q, cw, x)


def wmean(x, w):
    return float(np.sum(np.asarray(x, float) * w) / np.sum(w))


def a_of_tau(tau0):
    return T_STAR * tau0 / (1.0 - tau0)


def main():
    df = pd.read_csv(DATA, usecols=["annual_turnover_k", "vat_liability_k", "weight"])
    t = df["annual_turnover_k"].to_numpy(float) * 1000.0
    liab = df["vat_liability_k"].to_numpy(float) * 1000.0
    w = df["weight"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau0 = np.where(t > 0, liab / t, 0.0)

    lines = []
    P = lines.append
    P("=" * 78)
    P("TASK 1 — VALUE-ADDED CORRECTION TO THE DOMINATED REGION")
    P("=" * 78)
    P("")
    P("Textbook turnover-tax dominated region:  a = T*·tau/(1-tau)")
    P(f"  T* = £{T_STAR:,.0f},  tau = {TAU:.2f}  =>  a = £{a_of_tau(TAU):,.2f}")
    P("")
    P("Value-added correction: firm with net rate tau0 = liab/turnover faces")
    P("  a_i = T*·tau0/(1-tau0).  Mean tau0 is ~3% near the threshold, so a_i << £21,250.")
    P("")

    for lo, hi, label in [(80_000.0, 90_000.0, "[£80k,£90k]"),
                          (85_000.0, 90_000.0, "[£85k,£90k]")]:
        m = (t >= lo) & (t < hi)
        # Restrict the dominated-region statistic to firms with a genuine
        # (positive) net VAT rate; negative tau0 (net input creditors) have no
        # notch at all (a_i <= 0), reported separately.
        tm, wm = tau0[m], w[m]
        pos = tm > 0
        wpop = float(np.sum(wm))

        P("-" * 78)
        P(f"NEAR-THRESHOLD BAND {label}   weighted firms = {wpop:,.0f}")
        P("-" * 78)
        # tau0 distribution (all firms in band)
        P("  Net VAT rate tau0 = liab/turnover (weighted):")
        P(f"    mean   = {wmean(tm, wm)*100:7.3f}%")
        P(f"    median = {wquantile(tm, wm, 0.50)*100:7.3f}%")
        q25, q75 = wquantile(tm, wm, [0.25, 0.75])
        P(f"    p25/p75= {q25*100:7.3f}% / {q75*100:7.3f}%")
        P(f"    share tau0 < 1%  (effectively NO notch) = "
          f"{np.sum(wm[tm < 0.01])/wpop*100:6.2f}%")
        P(f"    share tau0 <= 0  (net input creditor, no notch) = "
          f"{np.sum(wm[tm <= 0])/wpop*100:6.2f}%")
        P("")
        # value-added dominated region a_i, over positive-tau0 firms
        a = a_of_tau(tm[pos])
        wp = wm[pos]
        P(f"  Value-added dominated region a_i = T*·tau0/(1-tau0)  (tau0>0 firms,"
          f" weight {np.sum(wp)/wpop*100:.1f}% of band):")
        P(f"    weighted MEAN   a_i = £{wmean(a, wp):,.0f}")
        P(f"    weighted MEDIAN a_i = £{wquantile(a, wp, 0.50):,.0f}")
        aq25, aq75 = wquantile(a, wp, [0.25, 0.75])
        P(f"    weighted p25/p75    = £{aq25:,.0f} / £{aq75:,.0f}")
        P("")
        # mean over WHOLE band (creditors -> a_i clipped at 0, no notch)
        a_all = np.where(tm > 0, a_of_tau(tm), 0.0)
        P(f"    weighted MEAN a_i over WHOLE band (tau0<=0 set to 0) = "
          f"£{wmean(a_all, wm):,.0f}")
        P("")
        # high-net-rate firms: top quartile of tau0 (the consumer-facing firms
        # the turnover-tax model is meant to describe)
        thr = wquantile(tm, wm, 0.75)
        hh = tm >= thr
        a_h = a_of_tau(tm[hh])
        w_h = wm[hh]
        P(f"  HIGH-NET-RATE firms (top tau0 quartile, tau0 >= {thr*100:.2f}%):")
        P(f"    weighted-mean tau0  = {wmean(tm[hh], w_h)*100:.3f}%")
        P(f"    weighted-mean a_i   = £{wmean(a_h, w_h):,.0f}")
        P(f"    weighted-median a_i = £{wquantile(a_h, w_h, 0.50):,.0f}")
        P("")

    P("=" * 78)
    P("HEADLINE CONTRAST")
    P("=" * 78)
    m = (t >= 80_000.0) & (t < 90_000.0)
    tm, wm = tau0[m], w[m]
    a_all = np.where(tm > 0, a_of_tau(tm), 0.0)
    P(f"  Turnover-tax dominated region (textbook):         £{a_of_tau(TAU):,.0f}")
    P(f"  Value-added dominated region (mean, [£80k,£90k]): "
      f"£{wmean(a_all, wm):,.0f}")
    P(f"  => only ~{wmean(a_all, wm)/a_of_tau(TAU)*100:.1f}% of the textbook"
      f" £21,250 survives once input reclaim is accounted for.")

    txt = "\n".join(lines) + "\n"
    with open(OUT, "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
