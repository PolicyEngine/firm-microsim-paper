"""Static VAT-threshold figures, in the project house style.

Three single-panel figures saved to ``results/`` (no embedded titles/logos —
captions go in LaTeX), reusing the shared house-style palette and helpers:

    * vat_threshold_revenue_impact.png — anchor reform (£85k→£90k), model vs
      HMRC, grouped bars by fiscal year.
    * revenue_impact_2025_26.png       — static revenue change vs threshold.
    * firms_impact_2025_26.png         — change in VAT-paying firms vs threshold.

Run ``python -m static`` to regenerate all three.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the exact house-style constants + helpers used by the other figures.
from firm_microsim.figures import (
    LABEL_SIZE,
    PALETTE,
    PRIMARY,
    TICK_SIZE,
    _save,
    _style_ax,
)

from .model import POLICY_THRESHOLD, StaticVATModel

REF_GREY = "#888888"      # current-threshold reference line
HMRC_COLOR = PALETTE[4]   # lighter teal for the HMRC series


def _current_threshold_marker(ax) -> None:
    """Dashed vertical line + label at the current £90k threshold."""
    ax.axvline(POLICY_THRESHOLD / 1000, color=REF_GREY, linestyle="--",
               linewidth=1.2, zorder=2)
    ax.text(POLICY_THRESHOLD / 1000 + 1, ax.get_ylim()[1] * 0.92, "Current £90k",
            fontsize=TICK_SIZE, color="#444444", ha="left", va="top")


def plot_revenue_impact(model: StaticVATModel, year: str = "2025-26") -> None:
    """Static revenue change (£m) vs registration threshold (£k)."""
    print(f"Generating revenue_impact_{year.replace('-', '_')}.png...")
    df = model.threshold_sweep(year)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["threshold_k"], df["revenue_change_m"], color=PRIMARY,
            marker="o", markersize=7, linewidth=2.5, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xlabel("Registration threshold (£k)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Revenue (£m)", fontsize=LABEL_SIZE)
    _style_ax(ax)
    _current_threshold_marker(ax)
    _save(fig, f"revenue_impact_{year.replace('-', '_')}.png")


def plot_firms_impact(model: StaticVATModel, year: str = "2025-26") -> None:
    """Change in VAT-paying firms (000s) vs registration threshold (£k)."""
    print(f"Generating firms_impact_{year.replace('-', '_')}.png...")
    df = model.threshold_sweep(year)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["threshold_k"], df["firms_change_k"], color=PRIMARY,
            marker="o", markersize=7, linewidth=2.5, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xlabel("Registration threshold (£k)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Change in number of firms (thousands)", fontsize=LABEL_SIZE)
    _style_ax(ax)
    _current_threshold_marker(ax)
    _save(fig, f"firms_impact_{year.replace('-', '_')}.png")


def plot_hmrc_comparison(model: StaticVATModel) -> None:
    """Anchor reform £85k→£90k: model vs HMRC costing, grouped bars by year."""
    print("Generating vat_threshold_revenue_impact.png...")
    df = model.anchor_reform()
    x = np.arange(len(df))
    w = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, df["hmrc_impact_m"], w, label="HMRC", color=HMRC_COLOR, zorder=3)
    ax.bar(x + w / 2, df["policyengine_impact_m"], w, label="PolicyEngine",
           color=PRIMARY, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["year"])
    ax.set_xlabel("Fiscal year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Revenue impact (£m)", fontsize=LABEL_SIZE)
    ax.legend(frameon=False, fontsize=LABEL_SIZE)
    _style_ax(ax)
    _save(fig, "vat_threshold_revenue_impact.png")


def generate_all(
    sweep_vintage: str = "2024-25",
    anchor_vintage: str = "2023-24",
    year: str = "2025-26",
) -> None:
    """Generate all three static figures.

    The HMRC anchor-reform figure uses the £85k (``anchor_vintage``) data — the
    pre-reform basis HMRC had at the March 2024 costing, where the affected band
    is clean. The forward-looking sweep figures use the current £90k
    (``sweep_vintage``) data, relative to the £90k baseline.
    """
    anchor_model = StaticVATModel(anchor_vintage)
    sweep_model = StaticVATModel(sweep_vintage)
    print(
        f"Static figures — anchor on {anchor_vintage} (£85k basis), "
        f"sweep on {sweep_vintage} (£90k baseline, {year}); "
        f"total VAT revenue at £90k: £{sweep_model.total_revenue_bn(year):.1f}bn"
    )
    plot_hmrc_comparison(anchor_model)
    plot_revenue_impact(sweep_model, year)
    plot_firms_impact(sweep_model, year)
    print("  done.")
