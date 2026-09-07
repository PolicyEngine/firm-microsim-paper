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
E = gross positive-part excess below T* over the window; E_net = signed
excess over the same window; Delta_R = missing mass above T* up to y_R;
y_R is searched only up to the window top and is CENSORED there when
Delta_R < E. b_llat = positive-part gap over [T*-W, y_R] / mean f_cf.
Points + grids
only; see 'Why no standard errors are reported' in the paper appendix and
results/seed_sensitivity.txt for generator-seed dispersion.
=========================================================================="""


def main() -> None:
    lines = [HEADER]
    for vintage in sorted(VINTAGES):
        threshold = VINTAGES[vintage]["threshold"]
        lines += ["", f"--- Vintage {vintage} (threshold GBP {threshold:.0f}k) ---"]
        for universe, label in (("chart", "OBR-chart universe (frame rows; plus the DBT stratum below the threshold only when it is generated) [HEADLINE]"),
                                ("all", "all rows (incl. any stratum rows above the threshold)"),
                                ("scope", "in-scope VAT firms")):
            r = BunchingEstimator(vintage, universe=universe).estimate()
            lines.append(
                f"[{label}] E = {r['E']:,.0f} (gross) | E_net = {r['E_net']:,.0f} | "
                f"Delta_R = {r['Delta_R']:,.0f} | b_llat = {r['b_llat']:.3f} | b = {r['b']:.4f} | "
                f"y_R = {r['y_R']:.2f}" + (" [CENSORED]" if r['y_R_censored'] else ""))
        est = BunchingEstimator(vintage, universe="chart")
        lines += [
            "",
            "Degree x window sensitivity (point estimates; OBR-chart universe):",
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
