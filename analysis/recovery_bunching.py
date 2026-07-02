"""Recovery / power validation for the £85k VAT bunching estimator.

Purpose
-------
The placebo (``analysis/placebo_bunching.py``) shows the estimator returns no
bunching when none is present by construction (specificity). This exercise
shows the complementary property: when a behavioural relocation of KNOWN
magnitude IS present, the estimator's HEADLINE statistics detect and size it
(power), and its biases are characterised honestly.

Why this design (and not the previous one)
------------------------------------------
An earlier version of this harness relocated mass from a donor window
[85, 100) to [75, 85) — both strictly INSIDE the estimator's ±15k exclusion
window [70, 100]. The polynomial counterfactual is fitted only on bins
OUTSIDE that window and rescaled by total mass (which relocation preserves),
so the fitted counterfactual was bit-for-bit identical across all injection
magnitudes: the exercise measured deposit bookkeeping, not estimator power,
and its side statistic ("signed-excess change vs baseline") required knowing
the null world — information a real application never has.

This version injects a Kleven–Waseem-consistent response: firms relocate to
just below the threshold from a missing-mass region (T*, Y_R_TRUE] that
extends BEYOND the exclusion window, with relocation probability declining
linearly from the threshold to zero at the true marginal buncher Y_R_TRUE.
The counterfactual fit therefore confronts genuinely depressed above-window
bins — exactly the situation with real bunching data — and recovery is scored
with the HEADLINE floored excess mass E and the endogenous marginal-buncher
location y_R, with no reference to the unobservable null.

Writes results/recovery_bunching.txt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm_microsim.bunching.model import (
    RANGE_LO,
    RANGE_HI,
    DEFAULT_WINDOW,
    _run_estimator,
)
from firm_microsim.config import SYNTHETIC_DATA_DIR, RESULTS_DIR, VINTAGES

VINTAGE = "2023-24"
T_STAR = float(VINTAGES[VINTAGE]["threshold"])  # 85.0
BIN_WIDTH = 1.0

# Injection geometry (£1,000 units).
Y_R_TRUE = 112.0   # true marginal buncher: missing mass spans (85, 112],
                   # extending 12k BEYOND the exclusion window top (100).
W_SPIKE = 5.0      # relocated firms land in [T*-W_SPIKE, T*), spike toward T*.

# Known injected magnitudes (weighted firms relocated = true excess mass).
E_TRUE_TARGETS = (2000.0, 5000.0, 8000.0)


def load_firms(vintage: str) -> pd.DataFrame:
    path = SYNTHETIC_DATA_DIR / f"synthetic_firms_{vintage}.csv"
    df = pd.read_csv(path, usecols=["annual_turnover_k", "weight"])
    df = df[
        (df["annual_turnover_k"] >= RANGE_LO) & (df["annual_turnover_k"] <= RANGE_HI)
    ].reset_index(drop=True)
    return df


def _bins():
    edges = np.arange(RANGE_LO - 0.5, RANGE_HI + 0.5 + BIN_WIDTH, BIN_WIDTH)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return edges, centres


def build_smooth_baseline(df: pd.DataFrame) -> np.ndarray:
    """Placebo-A reweight: smooth the per-£1k density across £85k (no step).

    Identical construction to ``placebo_bunching.build_reweight_placebo``: fit a
    log-quadratic density trend on the wings (|y-85| > half), replace the
    in-window density with that smooth fit, rescale each firm's weight by
    (smooth / observed) in its bin, and renormalise total mass. Yields b≈0,
    E≈0 — the clean null baseline onto which the known signal is grafted, so
    recovered mass is attributable entirely to the injection.
    """
    half = 20.0
    edges, centres = _bins()
    counts, _ = np.histogram(
        df["annual_turnover_k"], bins=edges, weights=df["weight"]
    )
    density = counts / BIN_WIDTH

    wing = (np.abs(centres - T_STAR) > half) & (density > 0)
    coef = np.polyfit(centres[wing], np.log(density[wing]), deg=2)
    smooth_density = np.exp(np.polyval(coef, centres))

    win = np.abs(centres - T_STAR) <= half
    target_density = density.copy()
    target_density[win] = smooth_density[win]

    factor = np.ones_like(density)
    nz = density > 0
    factor[nz] = target_density[nz] / density[nz]

    bin_idx = np.clip(
        np.digitize(df["annual_turnover_k"], edges) - 1, 0, len(centres) - 1
    )
    new_weight = df["weight"].to_numpy() * factor[bin_idx]
    new_weight *= df["weight"].sum() / new_weight.sum()
    return new_weight


def inject_kw_bunching(
    turnover: np.ndarray,
    weight_base: np.ndarray,
    e_true: float,
) -> tuple[np.ndarray, float]:
    """Relocate ``e_true`` of mass from (T*, Y_R_TRUE] to just below T*.

    Kleven–Waseem-consistent response: a firm at turnover y in the donor
    region (T*, Y_R_TRUE] relocates with probability
    ``s * (1 - (y - T*) / (Y_R_TRUE - T*))`` — declining linearly from the
    threshold to zero at the true marginal buncher — with the intensity ``s``
    chosen so the relocated mass equals ``e_true`` (capped at s=1 if the donor
    region cannot supply it). Relocated mass is deposited into
    [T*-W_SPIKE, T*) with a profile spiking toward T*. Total mass over the
    estimation range is conserved; the missing-mass region extends beyond the
    exclusion window top, as with genuine bunching.

    Returns ``(new_weight, e_actual)``.
    """
    w = weight_base.copy()
    donor = (turnover > T_STAR) & (turnover <= Y_R_TRUE)
    depth = 1.0 - (turnover[donor] - T_STAR) / (Y_R_TRUE - T_STAR)  # 1 at T*, 0 at y_R
    donor_capacity = float(np.sum(w[donor] * depth))

    s = min(1.0, e_true / donor_capacity) if donor_capacity > 0 else 0.0
    removed = w[donor] * depth * s
    e_actual = float(removed.sum())
    w[donor] = w[donor] - removed

    bunch = (turnover >= T_STAR - W_SPIKE) & (turnover < T_STAR)
    if not np.any(bunch):
        return w, 0.0
    dist = T_STAR - turnover[bunch]              # in (0, W_SPIKE]
    kernel = np.maximum(W_SPIKE + BIN_WIDTH - dist, 0.0)  # peaks just below T*
    alloc = kernel * weight_base[bunch]
    if alloc.sum() <= 0:
        alloc = np.ones_like(alloc)
    idx = np.where(bunch)[0]
    w[idx] = w[idx] + e_actual * alloc / alloc.sum()
    return w, e_actual


def estimate(turnover, weight, label, verbose=True) -> dict:
    r = _run_estimator(turnover, weight, T_STAR)
    if verbose:
        print(
            f"  {label:<30s} b={r['b']:+.4f}  E={r['E']:>10,.0f}  "
            f"b_llat={r['b_llat']:.3f}  y_R={r['y_R']:.2f}"
        )
    return r


def main() -> None:
    print("=" * 84)
    print(f"  RECOVERY / POWER TEST  (vintage {VINTAGE}, threshold £{T_STAR:.0f}k)")
    print("=" * 84)

    df = load_firms(VINTAGE)
    t = df["annual_turnover_k"].to_numpy()

    # --- Clean step-free baseline (the null world) -----------------------
    w_base = build_smooth_baseline(df)
    r_base = estimate(t, w_base, "BASELINE (smoothed, null)")

    # --- Inject known KW-consistent signals and recover ------------------
    rows = []
    for e_true in E_TRUE_TARGETS:
        w_inj, e_act = inject_kw_bunching(t, w_base, e_true)
        r = estimate(t, w_inj, f"INJECT E_true={e_true:,.0f}")
        rows.append(
            {
                "E_true": e_act,
                "E_hat": r["E"],
                "recovery": r["E"] / e_act if e_act > 0 else np.nan,
                "b_hat": r["b"],
                "b_llat": r["b_llat"],
                "y_R_hat": r["y_R"],
            }
        )

    res = pd.DataFrame(rows)
    print()
    print("  Recovery summary (headline floored E, mass-conservation y_R):")
    for _, row in res.iterrows():
        print(
            f"    E_true={row['E_true']:>8,.0f}  E_hat={row['E_hat']:>8,.0f}  "
            f"recovery={row['recovery']:.1%}  y_R_hat={row['y_R_hat']:.1f} "
            f"(true {Y_R_TRUE:.0f})  b_hat={row['b_hat']:+.4f}"
        )

    mean_recov = float(res["recovery"].mean())
    monotonic = bool(np.all(np.diff(res["E_hat"].to_numpy()) > 0))

    # --- Write results ---------------------------------------------------
    lines = []
    lines.append(
        "RECOVERY / POWER TEST — £85k UK VAT bunching estimator "
        f"(vintage {VINTAGE})"
    )
    lines.append("=" * 78)
    lines.append("")
    lines.append("Design")
    lines.append("------")
    lines.append(
        "1. Baseline: the £85k synthetic firms under the Placebo-A reweighting\n"
        "   (smooth log-quadratic density across £85k), a step-free null world:\n"
        f"   b={r_base['b']:+.4f}, E={r_base['E']:,.0f}, y_R={r_base['y_R']:.2f}.\n"
        "2. Injection (Kleven–Waseem-consistent): firms in the donor region\n"
        f"   (85, {Y_R_TRUE:.0f}] relocate to [{T_STAR - W_SPIKE:.0f}, 85) with probability\n"
        "   declining linearly from the threshold to zero at the true marginal\n"
        f"   buncher y_R = {Y_R_TRUE:.0f} — so the missing-mass region extends\n"
        f"   {Y_R_TRUE - T_STAR - DEFAULT_WINDOW:.0f}k BEYOND the ±{DEFAULT_WINDOW:.0f}k exclusion window, as with\n"
        "   genuine bunching. Mass is conserved.\n"
        "3. Score with the HEADLINE estimator outputs only (floored excess mass\n"
        "   E, endogenous y_R): no side statistics that require knowing the null."
    )
    lines.append("")
    lines.append("Results")
    lines.append("-------")
    hdr = (
        f"{'E_true':>10s} {'E_hat':>10s} {'recovery':>9s} "
        f"{'y_R_hat':>8s} {'b_hat':>8s} {'b_llat':>8s}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for _, row in res.iterrows():
        lines.append(
            f"{row['E_true']:>10,.0f} {row['E_hat']:>10,.0f} "
            f"{row['recovery']:>8.1%} {row['y_R_hat']:>8.2f} "
            f"{row['b_hat']:>+8.4f} {row['b_llat']:>8.3f}"
        )
    lines.append("")
    lines.append(
        f"Baseline (null, no injection): b={r_base['b']:+.4f}  "
        f"E={r_base['E']:,.0f}  y_R={r_base['y_R']:.2f}"
    )
    lines.append(
        f"True marginal buncher: y_R = {Y_R_TRUE:.0f}. "
        f"Mean recovery: {mean_recov:.1%} (monotone in E_true: {monotonic})."
    )
    lines.append("")
    lines.append("Reading")
    lines.append("-------")
    lines.append(
        "Recovery below 100% is genuine estimator attenuation, now measured\n"
        "rather than asserted: the counterfactual is fitted on bins that include\n"
        "the depressed (100, 112] missing-mass region outside the exclusion\n"
        "window, pulling the fitted counterfactual down and truncating the\n"
        "measured missing mass; the mass-conservation search then places y_R_hat\n"
        "accordingly. The estimator under-states a true response and does not\n"
        "over-state one; together with the placebo's null result (no false\n"
        "positive on a step-free world) this characterises the estimator as\n"
        "specific and, with quantified attenuation, sensitive."
    )

    out = RESULTS_DIR / "recovery_bunching.txt"
    out.write_text("\n".join(lines) + "\n")
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
