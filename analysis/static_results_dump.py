"""Dump the static sweep, anchor comparison, and revenue bases as a text artifact.

The paper's flagship static table (tab:static_revenue) previously existed only
as figure pixels and an HTML preview; this writes the full machine-readable
numbers to results/static_sweep.txt so every printed row is reproducible.
Also reports the voluntary-retention sensitivity on the anchor reform: the
headline convention deregisters every released firm; the sensitivity retains
the Liu et al. (2021) voluntary-registration share of released-firm liability.
"""

from __future__ import annotations

from firm_microsim.config import RESULTS_DIR
from firm_microsim.static.model import StaticVATModel, SWEEP_THRESHOLDS

LLAT_VOLUNTARY_SHARE = 0.43  # Liu-Lockwood-Almunia-Tam (2021): ~43% below-threshold


def main() -> None:
    lines = []
    W = lines.append

    sweep_model = StaticVATModel("2024-25")
    anchor_model = StaticVATModel("2023-24")

    W("STATIC RESULTS DUMP")
    W("=" * 74)
    W("")
    W("Anchor reform (85k -> 90k, 2023-24 vintage = HMRC's pre-reform basis)")
    W("model vs HMRC published costing (Spring Budget 2024), by fiscal year;")
    W("baseline column = counterfactual threshold path (85/85/87/89/92k).")
    W("-" * 74)
    anchor = anchor_model.anchor_reform()
    W(anchor.to_string(index=False))
    W("")
    W("Deregistration-threshold sensitivity: the headline releases registered")
    W("firms only below the GBP88k deregistration threshold ([85k, 88k) in the")
    W("raise years). Releasing the whole [baseline, 90k) band instead (gap = 0):")
    for _, row in anchor_model.anchor_reform(gap=0.0).iterrows():
        W(f"  {row['year']}: whole-band release {float(row['policyengine_impact_m']):+,.1f}m")
    W("")
    W("Voluntary-retention sensitivity (anchor, per year): headline assumes")
    W("every released firm deregisters (full liability lost). If the Liu et")
    W(f"al. (2021) voluntary share ({LLAT_VOLUNTARY_SHARE:.0%}) of released-firm liability is")
    W("retained, the impact scales accordingly:")
    ret = anchor_model.anchor_reform(retention=LLAT_VOLUNTARY_SHARE)
    for (_, row), (_, r2) in zip(anchor.iterrows(), ret.iterrows()):
        W(f"  {row['year']}: headline {float(row['policyengine_impact_m']):+,.1f}m -> "
          f"retention-adjusted {float(r2['policyengine_impact_m']):+,.1f}m "
          f"(HMRC {float(row['hmrc_impact_m']):+,.0f}m)")
    W("")
    W("Fixed-preference sensitivity: baseline voluntary registrants keep their")
    W("registration wherever the threshold moves (treats the frame's ~89% below-")
    W("threshold registered share as revealed preference; frame-selection bias")
    W("makes this a lower bound on the revenue loss):")
    for _, row in anchor_model.anchor_reform(retain_voluntary=True).iterrows():
        W(f"  {row['year']}: fixed-preference {float(row['policyengine_impact_m']):+,.1f}m")
    W("")
    W("Threshold sweep (2024-25 vintage, GBP 90k baseline, 2025-26 fiscal year)")
    W("method: direct mechanical reclassification of in-scope VAT firms; baseline")
    W("voluntary registrants stay registered; turnover and liability aged together")
    W("-" * 74)
    sweep = sweep_model.threshold_sweep(year="2025-26")
    W(sweep.to_string(index=False))
    W("")
    for year in ("2025-26", "2026-27"):
        W(f"Total VAT revenue at GBP 90k, {year}: "
          f"{sweep_model.total_revenue_bn(year=year):.1f}bn")
    W("")
    W("Bases and calibration-target comparison")
    W("-" * 74)
    for vintage, model in (("2023-24", anchor_model), ("2024-25", sweep_model)):
        df = model._aged(1.0)
        base = model._mandatory_base(df, model.data_threshold) / 1e9
        vol = (model._revenue(df, model.data_threshold) - base * 1e9) / 1e9
        W(f"  {vintage}: in-scope base (>= threshold, unaged) = {base:.1f}bn; "
          f"model-implied voluntary below-threshold remittance = {vol:.1f}bn (not calibrated)")
    W("")
    W(f"Sweep thresholds: {SWEEP_THRESHOLDS}")

    out = RESULTS_DIR / "static_sweep.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
