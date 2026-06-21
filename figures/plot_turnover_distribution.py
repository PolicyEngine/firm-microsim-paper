#!/usr/bin/env python3
"""Full-range weighted turnover distribution for the synthetic firm population.

Reproduces the full-range (1k-300k) turnover histogram in PolicyEngine style:
weighted firm counts in £1,000 bins across the full turnover range, with a
visible density step at the VAT registration threshold and a second step at the
£150k Flat Rate Scheme ceiling.

Reads the synthetic firm population (columns ``annual_turnover_k`` and
``weight``) from ``data/synthetic/synthetic_firms.csv`` (resolved relative to the
repo root) and writes ``figures/turnover_distribution_full.png``.

Usage
-----
    python figures/plot_turnover_distribution.py
    python figures/plot_turnover_distribution.py --input path/to/firms.csv
    python figures/plot_turnover_distribution.py --output path/to/out.png
    python figures/plot_turnover_distribution.py --threshold 90
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- Parameters --------------------------------------------------------------
# VAT registration threshold in £ thousands. Used ONLY for the annotation /
# vertical line; the underlying data is whichever synthetic file is loaded.
VAT_THRESHOLD_K = 85

# Flat Rate Scheme turnover ceiling in £ thousands (fixed policy parameter).
FRS_CEILING_K = 150

# Histogram binning: £1k bins across the full turnover range.
BIN_LO, BIN_HI, BIN_WIDTH = 0.5, 300.5, 1.0

# Path resolution: repo root is the parent of this file's directory (figures/).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "synthetic" / "synthetic_firms.csv"
DEFAULT_OUTPUT = REPO_ROOT / "figures" / "turnover_distribution_full.png"


def load_population(input_path: Path) -> pd.DataFrame:
    """Load the synthetic firm population.

    If the input file is missing, fall back to a small in-memory mock so the
    script can be smoke-tested end-to-end. See ``_mock_population``.
    """
    if input_path.exists():
        df = pd.read_csv(input_path)
        if "weight" not in df.columns:
            df["weight"] = 1.0
        return df

    # ----------------------------------------------------------------- #
    # TEST-ONLY FALLBACK. Not used when the real CSV is present.        #
    # Generates a plausible turnover distribution with a manual density #
    # step at the VAT threshold and at the £150k FRS ceiling, purely so #
    # the script can be run end-to-end without the real generator.      #
    # ----------------------------------------------------------------- #
    print(
        f"[mock] input not found at {input_path};\n"
        f"[mock] generating in-memory test population (smoke-test only)."
    )
    return _mock_population()


def _mock_population(n: int = 400_000, seed: int = 0) -> pd.DataFrame:
    """TEST-ONLY mock population with steps at the threshold and FRS ceiling."""
    rng = np.random.default_rng(seed)
    # Base turnover: broad spread across the 1-300k range.
    turnover = rng.gamma(shape=2.2, scale=40.0, size=n)
    turnover = np.clip(turnover, 1.0, 300.0)
    # Manual density steps: thin out firms just above the threshold and above
    # the FRS ceiling to mimic the bunching/notch behaviour in the real data.
    keep = np.ones(n, dtype=bool)
    above_thr = (turnover > VAT_THRESHOLD_K) & (turnover <= FRS_CEILING_K)
    above_frs = turnover > FRS_CEILING_K
    keep[above_thr] = rng.random(above_thr.sum()) < 0.62
    keep[above_frs] = rng.random(above_frs.sum()) < 0.45
    turnover = turnover[keep]
    weight = rng.normal(1.0, 0.01, size=turnover.size)
    return pd.DataFrame({"annual_turnover_k": turnover, "weight": weight})


def make_figure(df: pd.DataFrame, threshold_k: float, output_path: Path) -> None:
    """Build and save the full-range weighted turnover histogram."""
    bin_edges = np.arange(BIN_LO, BIN_HI, BIN_WIDTH)
    hist, _ = np.histogram(
        df["annual_turnover_k"], bins=bin_edges, weights=df["weight"]
    )
    x = np.arange(len(hist))

    plt.figure(figsize=(15, 6))
    plt.bar(x, hist, color="lightblue", alpha=0.7, edgecolor="black", linewidth=0.1)
    plt.xlabel("Annual Turnover (£k)")
    plt.ylabel("Number of Firms")

    # X tick labels every 10k (bin i covers turnover i+1 in £k).
    label_positions = [i for i in range(9, len(hist), 10)]
    plt.xticks(label_positions, [f"{i + 1}" for i in label_positions])

    # Threshold line (parameterised) and Flat Rate Scheme ceiling at £150k.
    plt.axvline(x=threshold_k - 1, color="red", linestyle="--", alpha=0.7, linewidth=2)
    plt.axvline(x=FRS_CEILING_K - 1, color="red", linestyle="--", alpha=0.7, linewidth=2)
    plt.text(
        threshold_k,
        max(hist) * 0.8,
        f"VAT Threshold (£{int(threshold_k)}k)",
        color="red",
        fontsize=8,
        ha="left",
    )
    plt.text(
        FRS_CEILING_K + 1,
        max(hist) * 0.8,
        "VAT Flat Rate Scheme",
        color="red",
        fontsize=8,
        ha="left",
    )

    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.figtext(
        0.5,
        -0.02,
        "Source: synthetic firm microsimulation population (weighted firm counts, £1k bins).",
        ha="center",
        fontsize=8,
        color="gray",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved {output_path}  (peak {hist.max():.0f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to synthetic firm CSV (default: data/synthetic/synthetic_firms.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output PNG (default: figures/turnover_distribution_full.png).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=VAT_THRESHOLD_K,
        help=f"VAT threshold in £k for the annotation/line (default: {VAT_THRESHOLD_K}).",
    )
    args = parser.parse_args()

    df = load_population(args.input)
    make_figure(df, args.threshold, args.output)


if __name__ == "__main__":
    main()
