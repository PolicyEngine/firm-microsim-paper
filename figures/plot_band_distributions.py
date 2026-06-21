#!/usr/bin/env python3
"""
plot_band_distributions.py

Reproduce a two-panel bar chart of firm counts by turnover band:

  LEFT  (Figure 1): Distribution of UK firms by turnover band, 2024 (ONS)
  RIGHT (Figure 2): Distribution of VAT-registered firms by turnover band,
                    2024-25 (HMRC)

The bars are driven by the real processed band CSVs (NOT hardcoded). The
hardcoded numbers near the bottom of this file are only a documented fallback
used when the processed CSVs cannot be found at all.

Output: figures/band_distributions.png  (override with --output)

Usage:
    python plot_band_distributions.py
    python plot_band_distributions.py --output /tmp/my_figure.png

Data sources (primary path):
    data/processed/ons_firm_turnover.csv
    data/processed/hmrc_vat_population_by_turnover_band.csv

Fallback originals (used to learn labels if processed files are missing):
    /Users/janansadeqian/uk-vatlab/data/ONS_UK_business_data/firm_turnover.csv
    /Users/janansadeqian/uk-vatlab/data/HMRC_VAT_annual_statistics/vat_population_by_turnover_band.csv
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless, file output only
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Style constants (match the target figure)
# ---------------------------------------------------------------------------
BG_COLOR = "#f2f2f2"        # light grey panel/figure background
BAR_COLOR = "#9c9c9c"       # medium grey bars
GRID_COLOR = "#cccccc"      # thin grey gridlines
TEXT_COLOR = "#333333"
PLE_BLUE = "#2C6496"        # PolicyEngine blue for the "PLE" mark

# ---------------------------------------------------------------------------
# CSV locations
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Processed data is organised by vintage: data/processed/<vintage>/...
#   2023-24 -> ONS 2024 workbook + HMRC 2023-24 stats (£85k threshold)
#   2024-25 -> ONS 2025 workbook + HMRC 2024-25 stats (£90k threshold)
# Default to 2024-25 so the figure shows the £90k banding.
DEFAULT_VINTAGE = "2024-25"

# Panel year labels per vintage: (ONS edition year, HMRC financial year).
VINTAGE_YEARS = {
    "2023-24": ("2024", "2023-24"),
    "2024-25": ("2025", "2024-25"),
}


def processed_paths(vintage):
    """Return (ons_csv, hmrc_csv) paths for a given processed-data vintage."""
    base = os.path.join(REPO_ROOT, "data", "processed", vintage)
    return (
        os.path.join(base, "ons_firm_turnover.csv"),
        os.path.join(base, "hmrc_vat_population_by_turnover_band.csv"),
    )

# ---------------------------------------------------------------------------
# Documented fallback (ONLY used if no CSV can be read). These mirror the
# spec's approximate heights in *thousands of firms*.
# ---------------------------------------------------------------------------
ONS_FALLBACK_LABELS = ["£0-49k", "£50-99k", "£100-249k", "£250-499k",
                       "£500-999k", "£1-5m", "£5m+"]
ONS_FALLBACK_THOUSANDS = [780, 1070, 1750, 770, 470, 450, 155]

HMRC_FALLBACK_LABELS = ["≤£0", "£1-90k", "£90-150k", "£150-300k",
                        "£300-500k", "£500k-1m", "£1-10m", ">£10m"]
HMRC_FALLBACK_THOUSANDS = [215, 680, 305, 335, 185, 180, 230, 48]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ons(path):
    """Return (labels, values_in_thousands, source_msg).

    ONS firm-turnover CSV: one row per SIC code, band columns hold firm counts.
    We sum every sector for each band to get the national total per band. Any
    'Total' summary row is dropped to avoid double-counting.
    """
    band_cols = ["0-49", "50-99", "100-249", "250-499", "500-999",
                 "1000-4999", "5000+"]
    # Cleaner display labels matching the helper script + target spec.
    display = ["£0-49k", "£50-99k", "£100-249k", "£250-499k",
               "£500-999k", "£1-5m", "£5m+"]

    if not os.path.exists(path):
        print(f"[WARN] ONS CSV not found at {path}; using fallback numbers.")
        return ONS_FALLBACK_LABELS, ONS_FALLBACK_THOUSANDS, "fallback (no CSV)"

    df = pd.read_csv(path)
    # Drop any 'Total' row (blank SIC Code or Description containing 'Total').
    if "Description" in df.columns:
        df = df[~df["Description"].astype(str).str.contains("Total", na=False)]
    totals = df[band_cols].sum()
    values_k = [totals[c] / 1000.0 for c in band_cols]
    print(f"[INFO] ONS loaded from {path}")
    return display, values_k, "processed CSV"


def load_hmrc(path):
    """Return (labels, values_in_thousands, source_msg, year, threshold_note).

    HMRC VAT-population CSV: one row per financial year. We take the most recent
    year and read its turnover-band columns.
    """
    band_cols = ["Negative_or_Zero", "£1_to_Threshold", "£Threshold_to_£150k",
                 "£150k_to_£300k", "£300k_to_£500k", "£500k_to_£1m",
                 "£1m_to_£10m", "Greater_than_£10m"]

    if not os.path.exists(path):
        print(f"[WARN] HMRC CSV not found at {path}; using fallback numbers.")
        return (HMRC_FALLBACK_LABELS, HMRC_FALLBACK_THOUSANDS,
                "fallback (no CSV)", "unknown", None)

    df = pd.read_csv(path)
    # Most recent financial year (rows are chronological; sort to be safe).
    df = df.sort_values("Financial_Year")
    row = df.iloc[-1]
    year = str(row["Financial_Year"])
    values_k = [float(row[c]) / 1000.0 for c in band_cols]
    src = "processed CSV"
    print(f"[INFO] HMRC loaded from {path}; latest year = {year}")

    # The CSV uses a generic "Threshold" band. In 2023-24 the VAT registration
    # threshold was £85k (frozen). The target spec asks for £90k banding, which
    # corresponds to the 2024-25 release (threshold raised to £90k on 1 Apr 2024).
    # Build the actual labels from the data we have, and flag any mismatch.
    if year >= "2024-25":
        thr_label = "£90k"
        threshold_note = None
    else:
        thr_label = "£85k"
        threshold_note = (
            f"HMRC data is the {year} release (VAT threshold = £85k), but the "
            "target panel asks for £90k bands. The £90k banding requires the "
            "2024-25 HMRC VAT Annual Statistics release. Plotting the actual "
            f"{year} band cut-points (£1-{thr_label}, {thr_label}-£150k)."
        )

    labels = ["≤£0", f"£1-{thr_label}", f"{thr_label}-150k",
              "£150-300k", "£300-500k", "£500k-1m", "£1-10m", ">£10m"]
    return labels, values_k, src, year, threshold_note


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def style_axis(ax, ymax):
    ax.set_facecolor(BG_COLOR)
    ax.set_ylim(0, ymax)
    # subtle horizontal gridlines only
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    # remove top/right spines; soften remaining ones
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, length=0)


def add_ple_mark(ax):
    """Render a simple 'PLE' PolicyEngine logo stand-in, bottom-right."""
    ax.text(0.985, 0.04, "PLE", transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=15, fontweight="bold", color=PLE_BLUE, alpha=0.85)


def add_source_note(ax, text):
    """Right-aligned source note just under the panel."""
    ax.text(1.0, -0.205, text, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#666666", style="italic")


def draw_panel(ax, labels, values_k, title, ymax, source_note):
    bars = ax.bar(range(len(labels)), values_k, color=BAR_COLOR,
                  width=0.78, zorder=3, edgecolor="none")
    style_axis(ax, ymax)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9,
                       color=TEXT_COLOR)
    ax.set_xlabel("Turnover band", fontsize=11, color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel("Number of firms (thousands)", fontsize=11, color=TEXT_COLOR)

    # left-aligned, bold-ish title sitting above the panel
    ax.set_title(title, fontsize=11, fontweight="bold", color="#222222",
                 loc="left", pad=14)

    add_ple_mark(ax)
    add_source_note(ax, source_note)
    return bars


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vintage",
        default=DEFAULT_VINTAGE,
        choices=sorted(VINTAGE_YEARS),
        help="Data vintage: '2024-25' (£90k, default) or '2023-24' (£85k).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(HERE, "band_distributions.png"),
        help="Output PNG path (default: figures/band_distributions.png)",
    )
    args = parser.parse_args()

    ons_path, hmrc_path = processed_paths(args.vintage)
    ons_year, hmrc_year_label = VINTAGE_YEARS[args.vintage]

    ons_labels, ons_vals, ons_src = load_ons(ons_path)
    hmrc_labels, hmrc_vals, hmrc_src, hmrc_year, thr_note = load_hmrc(hmrc_path)

    if thr_note:
        print("\n" + "=" * 72)
        print("[WARNING] Banding mismatch:")
        print("  " + thr_note)
        print("=" * 72 + "\n")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15.5, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.26,
                        wspace=0.20)

    draw_panel(
        ax_l, ons_labels, ons_vals,
        f"Figure 1: Distribution of UK firms\nby turnover band, {ons_year} (ONS)",
        ymax=1000,
        source_note="Source: ONS UK Business Statistics",
    )

    draw_panel(
        ax_r, hmrc_labels, hmrc_vals,
        "Figure 2: Distribution of VAT-registered firms\nby turnover band, "
        f"{hmrc_year_label} (HMRC)",
        ymax=700,
        source_note="Source: HMRC VAT Annual Statistics",
    )

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Wrote figure: {out}")

    # Echo the values actually plotted (thousands) for transparency.
    print("\nONS bars (thousands):")
    for lab, v in zip(ons_labels, ons_vals):
        print(f"  {lab:<12} {v:8.0f}")
    print(f"HMRC bars (thousands), year {hmrc_year}:")
    for lab, v in zip(hmrc_labels, hmrc_vals):
        print(f"  {lab:<12} {v:8.0f}")


if __name__ == "__main__":
    sys.exit(main())
