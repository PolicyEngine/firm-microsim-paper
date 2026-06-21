"""Paper figures, in the project house style.

Each figure is a single clean panel (no embedded title, source note, or logo —
captions and side-by-side layouts are composed in LaTeX), saved as a snake_case
PNG to ``results/`` at 300 dpi with a tight bounding box. The palette, font
sizes, dashed grid, and de-spined axes match the shared house style.

Every figure is produced for both data vintages, giving two full sets of
results:

    * ``85k`` — 2023-24 vintage (ONS 2024 + HMRC 2023-24), £85k VAT threshold
    * ``90k`` — 2024-25 vintage (ONS 2025 + HMRC 2024-25), £90k VAT threshold

Run ``python -m firm_microsim.figures`` to regenerate all of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import PROCESSED_DATA_DIR, RESULTS_DIR, SYNTHETIC_DATA_DIR, VINTAGES

# ---------------------------------------------------------------------------
# House-style constants
# ---------------------------------------------------------------------------
# Teal palette (primary -> light) shared across the project's figures.
PALETTE = ["#1b485e", "#122740", "#326b77", "#568b87", "#80ae9a", "#b5d1ae"]
PRIMARY = "#1b485e"
ACCENT = "#d62728"  # red accent for reference lines

LABEL_SIZE = 15
TICK_SIZE = 13

# Map each data vintage to its figure-name regime suffix and VAT threshold (£k).
VINTAGE_REGIME = {"2023-24": "85k", "2024-25": "90k"}

# ONS turnover-band columns and display labels.
ONS_BANDS = ["0-49", "50-99", "100-249", "250-499", "500-999", "1000-4999", "5000+"]
ONS_LABELS = ["£0-49k", "£50-99k", "£100-249k", "£250-499k", "£500-999k",
              "£1-5m", "£5m+"]

# HMRC turnover-band columns (schema-stable across vintages).
HMRC_BANDS = ["Negative_or_Zero", "£1_to_Threshold", "£Threshold_to_£150k",
              "£150k_to_£300k", "£300k_to_£500k", "£500k_to_£1m",
              "£1m_to_£10m", "Greater_than_£10m"]


# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------
def _style_ax(ax) -> None:
    """Apply the shared house style to an axis."""
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def _save(fig, name: str) -> None:
    """Save a figure to results/<name>.png at 300 dpi, tight bbox."""
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_ons_bands(vintage: str):
    """ONS firm counts (thousands) by turnover band, summed over SIC sectors."""
    df = pd.read_csv(PROCESSED_DATA_DIR / vintage / "ons_firm_turnover.csv")
    # Drop the summary 'Total' row so firms are not double-counted.
    df = df[~df["Description"].astype(str).str.contains("Total", na=False)]
    return [df[c].sum() / 1000.0 for c in ONS_BANDS]


def _load_hmrc_bands(vintage: str, threshold_label: str):
    """HMRC VAT-registered firm counts (thousands) by turnover band, latest year."""
    df = pd.read_csv(
        PROCESSED_DATA_DIR / vintage / "hmrc_vat_population_by_turnover_band.csv"
    )
    row = df.sort_values("Financial_Year").iloc[-1]
    values = [float(row[c]) / 1000.0 for c in HMRC_BANDS]
    labels = ["≤£0", f"£1-{threshold_label}", f"{threshold_label}-150k",
              "£150-300k", "£300-500k", "£500k-1m", "£1-10m", ">£10m"]
    return labels, values


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_firms_by_turnover_band(vintage: str) -> None:
    """ONS: distribution of all UK firms by turnover band (single panel)."""
    regime = VINTAGE_REGIME[vintage]
    print(f"Generating firms_by_turnover_band_{regime}.png...")
    values = _load_ons_bands(vintage)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(ONS_LABELS)), values, color=PRIMARY, width=0.8, zorder=3)
    ax.set_xticks(range(len(ONS_LABELS)))
    ax.set_xticklabels(ONS_LABELS, rotation=30, ha="right")
    ax.set_xlabel("Turnover band", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of firms (thousands)", fontsize=LABEL_SIZE)
    _style_ax(ax)
    _save(fig, f"firms_by_turnover_band_{regime}.png")


def plot_vat_firms_by_turnover_band(vintage: str) -> None:
    """HMRC: distribution of VAT-registered firms by turnover band (single panel)."""
    regime = VINTAGE_REGIME[vintage]
    print(f"Generating vat_firms_by_turnover_band_{regime}.png...")
    threshold_label = f"£{int(VINTAGES[vintage]['threshold'])}k"
    labels, values = _load_hmrc_bands(vintage, threshold_label)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(labels)), values, color=PRIMARY, width=0.8, zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("Turnover band", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of firms (thousands)", fontsize=LABEL_SIZE)
    _style_ax(ax)
    _save(fig, f"vat_firms_by_turnover_band_{regime}.png")


def plot_turnover_distribution(vintage: str) -> None:
    """Full-range weighted synthetic turnover distribution (£1k bins).

    Shows the density step at the VAT registration threshold and the £150k
    Flat Rate Scheme ceiling.
    """
    regime = VINTAGE_REGIME[vintage]
    threshold = float(VINTAGES[vintage]["threshold"])
    frs_ceiling = 150.0
    print(f"Generating turnover_distribution_{regime}.png...")

    path = SYNTHETIC_DATA_DIR / f"synthetic_firms_{vintage}.csv"
    if not path.exists():
        print(f"  [skip] synthetic data not found: {path}")
        return
    df = pd.read_csv(path, usecols=["annual_turnover_k", "weight"])

    edges = np.arange(0.5, 300.5, 1.0)
    counts, _ = np.histogram(
        df["annual_turnover_k"], bins=edges, weights=df["weight"]
    )
    centres = (edges[:-1] + edges[1:]) / 2.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(centres, counts, width=1.0, color=PRIMARY, zorder=3)

    ymax = ax.get_ylim()[1]
    for x, label in ((threshold, f"VAT threshold (£{int(threshold)}k)"),
                     (frs_ceiling, "Flat Rate Scheme (£150k)")):
        ax.axvline(x, color=ACCENT, linestyle="--", linewidth=1.5, zorder=4)
        # Vertical label just right of the line so the two never overlap.
        ax.text(x + 4, ymax * 0.97, label, color=ACCENT, fontsize=TICK_SIZE,
                rotation=90, ha="center", va="top")

    ax.set_xlim(0, 300)
    ax.set_xlabel("Annual turnover (£k)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of firms", fontsize=LABEL_SIZE)
    _style_ax(ax)
    _save(fig, f"turnover_distribution_{regime}.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def generate_all() -> None:
    """Generate every figure for both data vintages (two full sets)."""
    print("Generating figures (both vintages)...")
    for vintage in VINTAGES:
        plot_firms_by_turnover_band(vintage)
        plot_vat_firms_by_turnover_band(vintage)
        plot_turnover_distribution(vintage)
    print("  done.")


if __name__ == "__main__":
    generate_all()
