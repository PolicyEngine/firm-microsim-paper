#!/usr/bin/env python3
"""Write results/bunching_inference.txt — the paper's inference artifact.

For each vintage: the headline point estimates (default degree-7 polynomial,
±£15k window) and the degree × window specification grid. Point estimates
only — the paper reports no standard errors (deterministic construction, no
estimand; see the inference appendix) — with generator-seed dispersion in
results/seed_sensitivity.txt.

Run:  .venv/bin/python scripts/bunching_inference.py
"""
from __future__ import annotations

from firm_microsim.bunching.model import BunchingEstimator
from firm_microsim.config import RESULTS_DIR, VINTAGES

OUT = RESULTS_DIR / "bunching_inference.txt"

HEADER = """BUNCHING INFERENCE — definitive build (OBR shape targets, side-consistent
frame scaling; no below-threshold liability calibration). Definitions:
bins are GBP 1k wide, (e, e+1], with edges aligned to T* (a firm at exactly
T* counts below it); estimation range (20k, 140k]; excluded window W either
side of T*. E = gross positive-part excess below T* over the window;
E_net = signed excess over the same window; Delta_R = missing mass above T*
up to y_R (= min(cumulative missing mass, E)); y_R is searched only up to
T*+W and is CENSORED there when Delta_R < E; E <= 0 gives y_R = T*.
b_llat = E_net / mean counterfactual bin count over (T*-W, T*+W), after
LLAT (2021) eq. 20 (signed; symmetric upper edge instead of an estimated one).
Points + grids
only; see 'Why no standard errors are reported' in the paper appendix and
results/seed_sensitivity.txt for generator-seed dispersion.
=========================================================================="""


def main() -> None:
    lines = [HEADER]
    for vintage in sorted(VINTAGES):
        est = BunchingEstimator(vintage)
        res = est.estimate()
        threshold = VINTAGES[vintage]["threshold"]
        lines += [
            "",
            f"--- Vintage {vintage} (threshold GBP {threshold:.0f}k) ---",
            f"E = {res['E']:,.0f} (gross) | E_net = {res['E_net']:,.0f} | "
            f"Delta_R = {res['Delta_R']:,.0f} | b_llat = {res['b_llat']:.3f} | "
            f"b = {res['b']:.4f} | y_R = {res['y_R']:.2f}"
            + (" [CENSORED at search cap: mass conservation does not bind]"
               if res['y_R_censored'] else ""),
            "",
            "Degree x window sensitivity (point estimates):",
            est.sensitivity()["degree_window"].to_string(index=False),
        ]
    lines.append("")
    text = "\n".join(lines)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
