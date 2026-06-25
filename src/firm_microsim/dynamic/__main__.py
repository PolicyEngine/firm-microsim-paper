"""CLI entry point for the iso-elastic dynamic VAT-notch simulator.

Runs :func:`crosscheck` first (aborts if any cross-check fails), then for each
reform reports the STATIC cost and the e-governed BEHAVIOURAL cost at each
elasticity in {0.05, 0.17, 0.32}, with the number of firms re-optimising and the
near-threshold mass change. Writes ``results/dynamic_reform_results.txt`` and the
house-style figures.

Examples::

    firm-microsim-dynamic
    firm-microsim-dynamic --reform taper --elasticity 0.17
    firm-microsim-dynamic --reform rate10 --static
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .figures import (
    RESULTS_DIR,
    fig_cost_vs_elasticity,
    fig_notch_fit,
    fig_reform_distribution,
)
from .model import (
    ELASTICITIES,
    E_HEADLINE,
    build_reforms,
    crosscheck,
    load_reform_data,
    marginal_buncher,
    reform_revenue,
)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="firm-microsim-dynamic")
    ap.add_argument("--vintage", default="2023-24")
    ap.add_argument("--elasticity", type=float, default=None,
                    help="single elasticity; default sweeps {0.05,0.17,0.32}")
    ap.add_argument("--reform",
                    choices=["raise100k", "taper", "rate10", "rate15", "all"],
                    default="all")
    beh = ap.add_mutually_exclusive_group()
    beh.add_argument("--behavioural", dest="behavioural", action="store_true")
    beh.add_argument("--static", dest="behavioural", action="store_false")
    ap.set_defaults(behavioural=True)
    args = ap.parse_args(argv)

    # 1. Cross-check — abort on any failure (dominated region, marginal buncher,
    #    ELASTICITY identity, e->0 -> static convergence, static costs).
    print("=" * 78)
    crosscheck()
    print("=" * 78)

    elasticities = (args.elasticity,) if args.elasticity is not None else ELASTICITIES
    reforms = build_reforms()
    if args.reform != "all":
        reforms = {args.reform: reforms[args.reform]}

    print("\nLoading reform-costing dataset ...")
    df = load_reform_data()
    print(f"  {len(df):,} firm rows, weighted {df['weight'].sum():,.0f}.")
    reg = df["turnover"] >= 85_000.0
    base_bn = float((df.loc[reg, "liab"] * df.loc[reg, "weight"]).sum() / 1e9)
    print(f"  baseline registered VAT base: £{base_bn:,.3f}bn.")

    lines = []
    lines.append("Iso-elastic (Kleven-Waseem) dynamic VAT-notch simulator — reform results")
    lines.append(f"vintage={args.vintage}  T*=£85,000  tau_max=0.20  "
                 f"taper/band top=£105,000")
    lines.append(f"Repo-generated baseline registered VAT base = £{base_bn:,.3f}bn.")
    lines.append("A SINGLE turnover elasticity e governs the response. The "
                 "behavioural cost is")
    lines.append("now a defensible e-SENSITIVITY RANGE over e in {0.05,0.17,0.32} "
                 "(headline 0.17).")
    lines.append("Ability n is an accounting anchor recovered under the £85k notch "
                 "given e; e is")
    lines.append("NOT identified from the synthetic data (see the placebo) — read "
                 "results as")
    lines.append("CONDITIONAL on the assumed e.")
    lines.append("=" * 78)

    # --- Analytic marginal buncher n_H(e). ----------------------------------
    lines.append("\nMarginal buncher n_H(e) [analytic; £]:")
    for e in elasticities:
        nH, dy = marginal_buncher(e)
        row = f"  e={e:<5} n_H=£{nH/1000:7.1f}k   dy*=£{dy/1000:6.1f}k"
        print(row)
        lines.append(row)

    # --- Results table: static + behavioural(e) per reform. -----------------
    lines.append("\nReform costs (change vs £85k-notch baseline):")
    hdr = (f"  {'reform':<32}{'static':>10}"
           + "".join(f"{'e=' + str(e):>12}" for e in elasticities)
           + f"{'#reopt(e=' + str(E_HEADLINE) + ')':>18}"
           + f"{'near-thr Δ':>14}")
    print(hdr)
    lines.append(hdr)

    for rname, (sched, label) in reforms.items():
        rs = reform_revenue(df, sched, E_HEADLINE, behavioural=False)
        beh = {}
        if args.behavioural:
            for e in elasticities:
                beh[e] = reform_revenue(df, sched, e, behavioural=True)

        cells = f"  {label:<32}£{rs['d_rev']/1e6:>+8.0f}m"
        for e in elasticities:
            if e in beh:
                cells += f"  £{beh[e]['d_rev']/1e6:>+8.0f}m"
            else:
                cells += f"{'—':>12}"
        if E_HEADLINE in beh:
            rb = beh[E_HEADLINE]
            cells += f"{rb['n_moved']:>16,.0f}"
            cells += f"{rb['near_change']:>+14,.0f}"
        print(cells)
        lines.append(cells)

        # Headline reform-distribution figure (e=0.17).
        if args.behavioural and E_HEADLINE in beh:
            fig_reform_distribution(df, beh[E_HEADLINE], label, E_HEADLINE)

    # Direction-of-e note.
    lines.append("")
    lines.append("Direction: a LARGER e makes every reform CHEAPER (smaller "
                 "revenue loss). Each")
    lines.append("reform lowers the effective rate on the band, so the "
                 "iso-elastic response scales")
    lines.append("turnover UP toward the firm's frictionless optimum "
                 "(y* = y_obs*((1-tau_reform)/(1-tau_base))^e > y_obs), widening "
                 "the taxed base;")
    lines.append("the larger e, the larger this base-broadening offset to the "
                 "static cost.")
    lines.append("NOTE: the taper figure captures this intensive base-broadening "
                 "only; it omits the")
    lines.append("taper's marginal-rate distortion (the iso-elastic cost cannot "
                 "credibly price the")
    lines.append("downward response a taper induces), so the taper's behavioural "
                 "cost is a lower")
    lines.append("bound on its true cost. See Section 7 of the paper.")

    # Notch-fit figure (observed density + dominated region + n_H) per e.
    for e in elasticities:
        fig_notch_fit(df, e)
    fig_cost_vs_elasticity(df)

    out = RESULTS_DIR / "dynamic_reform_results.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
