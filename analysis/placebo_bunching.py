"""Placebo test for the £85k VAT bunching estimate.

Referee worry
-------------
The synthetic generator draws within-band turnover *smoothly* and has NO firm
location-choice. The density step at £85k that the bunching estimator reads as
"bunching" is produced by the calibration: the HMRC turnover-band targets are
coarse — one count for everything in ``£1_to_Threshold`` (<= £85k) and one for
``£Threshold_to_£150k`` (£85k-£150k). Those two bands have very different
average per-£1k densities (7,981 vs 4,697 firms per £1k in 2023-24), so the
weight optimiser produces a mechanical step at exactly £85k. If so, the
estimator is just reading back the calibration target, not behaviour.

The test
--------
Build a PLACEBO population calibrated to the SAME aggregate mass but with the
near-threshold density step REMOVED (a smooth interpolation across £85k that has
no step), run the SAME estimator, and report b and E.

  * placebo b -> 0  => the spike is MECHANICAL (inherited from the target).
  * placebo b ~ 0.060 => the spike is an estimator/DGP artefact independent of
    the target.

Two independent placebo constructions are run for robustness:

  A. REWEIGHT placebo (light, no regeneration). Take the existing synthetic
     firm-level data and rescale per-£1k weights so the turnover density across
     the threshold follows a single smooth log-linear trend (fit on the wings,
     OUTSIDE a +/-20k window around £85k, exactly as a counterfactual would be),
     with no step at £85k. Total mass over the estimation range is preserved.

  B. REGENERATE placebo (heavy, definitive). Replace the two HMRC band counts
     straddling £85k with a smooth split implied by a single common per-£1k
     density (the combined mass spread uniformly across £1-£150k), then re-run
     the full generator + multi-objective calibration with those placebo
     targets, and run the estimator on the resulting weighted population.

The ACTUAL population is also run as a control (should reproduce b=0.060).

Outputs a comparison and writes results/placebo_bunching.txt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bunching.model import (
    RANGE_LO,
    RANGE_HI,
    DEFAULT_WINDOW,
    _run_estimator,
)
from firm_microsim.config import SYNTHETIC_DATA_DIR, RESULTS_DIR, VINTAGES

VINTAGE = "2023-24"
T_STAR = float(VINTAGES[VINTAGE]["threshold"])  # 85.0


def load_firms(vintage: str) -> pd.DataFrame:
    path = SYNTHETIC_DATA_DIR / f"synthetic_firms_{vintage}.csv"
    df = pd.read_csv(path, usecols=["annual_turnover_k", "weight"])
    df = df[
        (df["annual_turnover_k"] >= RANGE_LO) & (df["annual_turnover_k"] <= RANGE_HI)
    ].reset_index(drop=True)
    return df


def estimate(turnover: np.ndarray, weight: np.ndarray, label: str) -> dict:
    r = _run_estimator(turnover, weight, T_STAR)
    print(
        f"  {label:<28s} b={r['b']:+.4f}  E={r['E']:>10,.0f}  "
        f"b_llat={r['b_llat']:.3f}  y_R={r['y_R']:.2f}  sigma={r['sigma']:.3f}"
    )
    return r


def build_reweight_placebo(df: pd.DataFrame) -> np.ndarray:
    """Placebo A: reweight firms so the per-£1k density is smooth across £85k.

    Fit a log-linear density trend on the wings (outside a +/-`half` window
    around the threshold), evaluate the smooth (step-free) target density inside
    the window, and rescale each firm's weight by (smooth target density /
    observed density) in its £1k bin. Mass over the whole estimation range is
    rescaled back to the original total so the only thing changed is the SHAPE
    across the threshold, not the aggregate.
    """
    half = 20.0  # exclusion half-window for fitting the smooth trend (£k)
    bin_width = 1.0
    edges = np.arange(RANGE_LO - 0.5, RANGE_HI + 0.5 + bin_width, bin_width)
    centres = (edges[:-1] + edges[1:]) / 2.0
    counts, _ = np.histogram(
        df["annual_turnover_k"], bins=edges, weights=df["weight"]
    )
    density = counts / bin_width  # firms per £1k

    # Fit log-linear density on the wings (outside the manipulation window),
    # so the fit is NOT informed by the step we are trying to remove.
    wing = (np.abs(centres - T_STAR) > half) & (density > 0)
    coef = np.polyfit(centres[wing], np.log(density[wing]), deg=2)
    smooth_density = np.exp(np.polyval(coef, centres))

    # Inside the window, replace observed density with the smooth fit; outside,
    # keep observed (so wings are untouched).
    win = np.abs(centres - T_STAR) <= half
    target_density = density.copy()
    target_density[win] = smooth_density[win]

    # Per-bin rescale factor (avoid div-by-zero).
    factor = np.ones_like(density)
    nz = density > 0
    factor[nz] = target_density[nz] / density[nz]

    # Map each firm to its bin and apply the factor.
    bin_idx = np.clip(np.digitize(df["annual_turnover_k"], edges) - 1, 0, len(centres) - 1)
    new_weight = df["weight"].to_numpy() * factor[bin_idx]

    # Preserve total mass over the estimation range.
    new_weight *= df["weight"].sum() / new_weight.sum()
    return new_weight


def build_regenerate_placebo() -> dict:
    """Placebo B: smooth the HMRC band targets across £85k, re-run generator.

    Replace the two counts straddling the threshold (``£1_to_Threshold`` and
    ``£Threshold_to_£150k``) with a split implied by a SINGLE common per-£1k
    density: spread their combined mass uniformly over £1-£150k. This removes
    the step in the calibration target while preserving the £1-£150k total and
    all other band targets. Then run the full generator + calibration and the
    estimator on the resulting weighted population.
    """
    import logging
    from dataclasses import replace

    import importlib
    import sys

    # The package __init__ binds the *function* `generate` to the name
    # `firm_microsim.generate`; the actual submodule is in sys.modules.
    gen_mod = importlib.import_module("firm_microsim.generate")
    if not hasattr(gen_mod, "load_data"):
        gen_mod = sys.modules["firm_microsim.generate"]
    run_generate = gen_mod.generate
    from firm_microsim.config import Config

    logging.getLogger("firm_microsim").setLevel(logging.ERROR)

    # generate.py does `from .data_loader import ... load_data` into its own
    # namespace, so patch the name as referenced inside the generate module.
    orig_loader_in_gen = gen_mod.load_data

    def patched_load(config):
        data = orig_loader_in_gen(config)
        below = data.hmrc_bands["£1_to_Threshold"]
        above = data.hmrc_bands["£Threshold_to_£150k"]
        combined = below + above
        # Single common density over £1-£150k; threshold at 85.
        dens = combined / 150.0
        data.hmrc_bands = dict(data.hmrc_bands)
        data.hmrc_bands["£1_to_Threshold"] = dens * T_STAR
        data.hmrc_bands["£Threshold_to_£150k"] = dens * (150.0 - T_STAR)
        return data

    gen_mod.load_data = patched_load
    try:
        cfg = Config.for_vintage(VINTAGE)
        df = run_generate(cfg, write=False)
    finally:
        gen_mod.load_data = orig_loader_in_gen

    df = df[
        (df["annual_turnover_k"] >= RANGE_LO) & (df["annual_turnover_k"] <= RANGE_HI)
    ].reset_index(drop=True)
    r = estimate(
        df["annual_turnover_k"].to_numpy(),
        df["weight"].to_numpy(),
        "PLACEBO B (regenerate)",
    )
    return r


def main() -> None:
    print("=" * 78)
    print(f"  PLACEBO BUNCHING TEST  (vintage {VINTAGE}, threshold £{T_STAR:.0f}k)")
    print("=" * 78)

    df = load_firms(VINTAGE)
    t = df["annual_turnover_k"].to_numpy()
    w = df["weight"].to_numpy()

    # --- Control: actual population --------------------------------------
    r_actual = estimate(t, w, "ACTUAL (control)")

    # --- Placebo A: reweight to smooth density across £85k ----------------
    w_placebo = build_reweight_placebo(df)
    r_pa = estimate(t, w_placebo, "PLACEBO A (reweight)")

    # Diagnostic: density step just below vs just above £85k, before/after.
    def step(turnover, weight):
        below = ((turnover >= 80) & (turnover < 85))
        above = ((turnover >= 85) & (turnover < 90))
        db = weight[below].sum() / 5.0
        da = weight[above].sum() / 5.0
        return db, da, (db - da) / da

    db0, da0, st0 = step(t, w)
    db1, da1, st1 = step(t, w_placebo)
    print()
    print(f"  Density step at £85k (firms per £1k, 80-85 vs 85-90):")
    print(f"    ACTUAL  : below={db0:,.0f}  above={da0:,.0f}  step={st0:+.1%}")
    print(f"    PLACEBO A: below={db1:,.0f}  above={da1:,.0f}  step={st1:+.1%}")
    print()

    # --- Placebo B: regenerate with smoothed band targets ----------------
    r_pb = build_regenerate_placebo()

    # --- Write results ----------------------------------------------------
    lines = []
    lines.append("PLACEBO BUNCHING TEST — £85k UK VAT threshold (vintage 2023-24)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Method")
    lines.append("------")
    lines.append(
        "The HMRC turnover-band targets are coarse: one count for <= £85k\n"
        "(£1_to_Threshold = 678,350) and one for £85k-£150k\n"
        "(£Threshold_to_£150k = 305,320). Their average per-£1k densities differ\n"
        "(7,981 vs 4,697 firms/£1k), so the calibration target ITSELF contains a\n"
        "density step at exactly £85k. The generator has no firm location-choice,\n"
        "so any density discontinuity is inherited from this target.\n"
    )
    lines.append(
        "PLACEBO A (reweight): rescale the existing synthetic firms' weights so\n"
        "the per-£1k turnover density follows a single smooth log-quadratic trend\n"
        "across £85k (fit on the wings, |y-85|>20k, then applied inside the\n"
        "window). Total mass over [20k,140k] preserved. No regeneration.\n"
    )
    lines.append(
        "PLACEBO B (regenerate): replace the two HMRC band counts straddling £85k\n"
        "with a split implied by a SINGLE common per-£1k density (combined mass\n"
        "spread uniformly over £1-£150k), so the calibration target has NO step,\n"
        "then re-run the full generator + multi-objective calibration and the\n"
        "estimator on the resulting weighted population.\n"
    )
    lines.append("")
    lines.append("Results")
    lines.append("-------")
    hdr = f"{'population':<26s} {'b':>9s} {'E':>12s} {'b_llat':>8s} {'y_R':>7s}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for label, r in [
        ("ACTUAL (control)", r_actual),
        ("PLACEBO A (reweight)", r_pa),
        ("PLACEBO B (regenerate)", r_pb),
    ]:
        lines.append(
            f"{label:<26s} {r['b']:>+9.4f} {r['E']:>12,.0f} "
            f"{r['b_llat']:>8.3f} {r['y_R']:>7.2f}"
        )
    lines.append("")
    lines.append(
        f"Density step at £85k (80-85 vs 85-90 per £1k):\n"
        f"  ACTUAL   below={db0:,.0f} above={da0:,.0f} step={st0:+.1%}\n"
        f"  PLACEBO A below={db1:,.0f} above={da1:,.0f} step={st1:+.1%}\n"
    )

    out = RESULTS_DIR / "placebo_bunching.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
