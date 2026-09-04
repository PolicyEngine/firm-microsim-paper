#!/usr/bin/env python3
"""Generator-seed sensitivity of the headline 2023-24 statistics.

Rebuilds the full-size 2023-24 population under alternative generator seeds
and recomputes, on each build, the headline numbers the paper quotes:

  * excess mass E and b_llat (default estimator settings),
  * the raise-to-100k cost (direct band-sum of [85k, 100k) liability),
  * the graduated-taper cost (shared dynamic schedule code, static mode),
  * the registered VAT base (>= 85k).

The paper reports point estimates with no standard errors (the construction
is deterministic given the seed; there is no sampling estimand), so this
dispersion across seeds is the honest scale of generator Monte Carlo noise.

The first seed listed is read from the canonical CSV if it matches the
config default (that CSV *is* the paper build); other seeds are full
generate+calibrate runs held in memory (several minutes each).

Run:  .venv/bin/python scripts/seed_sensitivity.py --seeds 42 7 99
Writes: results/seed_sensitivity.txt
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from firm_microsim.bunching.model import RANGE_HI, RANGE_LO, _run_estimator
from firm_microsim.config import DEFAULT_CONFIG, RESULTS_DIR, SYNTHETIC_DATA_DIR
from firm_microsim.dynamic.model import E_HEADLINE, build_reforms, reform_revenue
from firm_microsim.generate import generate

VINTAGE = "2023-24"
T_STAR = 85_000.0
T_NEW = 100_000.0
OUT = RESULTS_DIR / "seed_sensitivity.txt"
COLS = ["annual_turnover_k", "vat_liability_k", "weight", "vat_scope"]


def headline_stats(df: pd.DataFrame) -> dict:
    """Compute the paper's headline 2023-24 statistics on one build."""
    tk = df["annual_turnover_k"].to_numpy()
    t = tk * 1000.0
    scope = df["vat_scope"].to_numpy(dtype=bool)
    # In-scope liabilities only (issue #37); bunching runs on the frame density.
    liab = np.where(scope, df["vat_liability_k"].to_numpy() * 1000.0, 0.0)
    w = df["weight"].to_numpy()

    # Same population filter as BunchingEstimator.__init__ — the headline
    # E/b_llat are defined on the [RANGE_LO, RANGE_HI] estimation range.
    in_range = (tk >= RANGE_LO) & (tk <= RANGE_HI)
    est = _run_estimator(tk[in_range], w[in_range], T_STAR / 1000.0)

    band = (t >= T_STAR) & (t < T_NEW)
    raise_m = (liab[band] * w[band]).sum() / 1e6

    reform_df = pd.DataFrame({"turnover": t, "liab": liab, "weight": w})
    taper_schedule = build_reforms()["taper"][0]
    taper = reform_revenue(reform_df, taper_schedule, E_HEADLINE, behavioural=False)
    taper_m = -taper["d_rev"] / 1e6

    reg = t >= T_STAR
    base_bn = (liab[reg] * w[reg]).sum() / 1e9

    return {
        "E": est["E"],
        "b_llat": est["b_llat"],
        "raise_m": raise_m,
        "taper_m": taper_m,
        "base_bn": base_bn,
    }


def build_for_seed(seed: int) -> pd.DataFrame:
    """Return the full-size 2023-24 build for ``seed`` (canonical CSV if it
    is the config-default seed, else a fresh in-memory generate run)."""
    canonical = SYNTHETIC_DATA_DIR / f"synthetic_firms_{VINTAGE}.csv"
    if seed == DEFAULT_CONFIG.seed and canonical.exists():
        return pd.read_csv(canonical, usecols=COLS)
    df = generate(vintage=VINTAGE, seed=seed, write=False)
    return df[COLS]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99])
    args = parser.parse_args(argv)

    rows = []
    for seed in args.seeds:
        print(f"[seed {seed}] building...", flush=True)
        stats = headline_stats(build_for_seed(seed))
        label = f"seed {seed} (paper)" if seed == DEFAULT_CONFIG.seed else f"seed {seed}"
        rows.append((label, stats))
        print(f"[seed {seed}] E={stats['E']:,.0f} b_llat={stats['b_llat']:.2f} "
              f"raise={stats['raise_m']:.1f} taper={stats['taper_m']:.1f} "
              f"base={stats['base_bn']:.1f}", flush=True)

    def half_range(key: str) -> float:
        vals = [s[key] for _, s in rows]
        return (max(vals) - min(vals)) / 2.0

    lines = [
        "GENERATOR-SEED SENSITIVITY (full-size 2023-24 builds; "
        f"seeds {'/'.join(str(s) for s in args.seeds)})",
        "two-universe build (#37): ONS-frame rows + HMRC registered-subset rows via",
        "per-band registration propensity; in-scope liabilities only; scope is a seeded draw",
        "=" * 70,
        f"{'run':<18}{'E':>8}{'b_llat':>8}{'raise £m':>10}{'taper £m':>10}{'base £bn':>10}",
    ]
    for label, s in rows:
        lines.append(
            f"{label:<18}{s['E']:>8,.0f}{s['b_llat']:>8.2f}"
            f"{s['raise_m']:>10.1f}{s['taper_m']:>10.1f}{s['base_bn']:>10.1f}"
        )
    lines += [
        "",
        f"half-range across seeds: E ±{half_range('E'):,.0f} | "
        f"b_llat ±{half_range('b_llat'):.2f} | "
        f"raise ±£{half_range('raise_m'):.1f}m | "
        f"taper ±£{half_range('taper_m'):.1f}m | "
        f"base ±£{half_range('base_bn'):.2f}bn",
    ]
    text = "\n".join(lines)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
