"""Bunching figure, in the project house style.

Single clean panel (no embedded title/logo — caption goes in LaTeX), reusing the
shared house-style palette and helpers. Saved to ``results/``:

    * bunching_analysis_<regime>.png  — observed vs no-bunching counterfactual at T*

(The structural notch figure lives in the ``notch`` package.)
Run ``python -m bunching`` to regenerate it.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from firm_microsim.figures import (
    ACCENT,
    LABEL_SIZE,
    PALETTE,
    PRIMARY,
    TICK_SIZE,
    VINTAGE_REGIME,
    _save,
    _style_ax,
)

from .model import BunchingEstimator

CF_COLOR = PALETTE[3]   # counterfactual line (mid teal)
SHADE = "#e8eef0"       # light shade for the excluded / dominated region


def plot_bunching(est: BunchingEstimator, vintage: str) -> None:
    """Observed vs mass-conserving counterfactual density around the threshold."""
    regime = VINTAGE_REGIME[vintage]
    print(f"Generating bunching_analysis_{regime}.png...")
    res = est.estimate()
    centres = np.asarray(res["centres"])
    f_obs = np.asarray(res["f_obs"])
    f_cf = np.asarray(res["f_cf"])
    t_star = est.t_star
    y_R = res["y_R"]

    # Focus the view on the neighbourhood of the threshold (out to £130k).
    lo, hi = t_star - 40, 130
    m = (centres >= lo) & (centres <= hi)

    fig, ax = plt.subplots(figsize=(10, 6))
    # excluded estimation window [T*-window, y_R]
    ax.axvspan(t_star - 15, y_R, color=SHADE, zorder=0)
    ax.plot(centres[m], f_obs[m] / 1000.0, color=PRIMARY, marker="o",
            markersize=5, linewidth=1.8, zorder=3, label="Observed")
    ax.plot(centres[m], f_cf[m] / 1000.0, color=CF_COLOR, linestyle="--",
            linewidth=2.2, zorder=4, label="No-bunching scenario")
    # Registration-threshold marker, styled like the OBR figure (Fig. 2):
    # grey dotted line + horizontal grey label at top.
    ax.axvline(t_star, color="0.35", lw=1, ls=":", zorder=5)
    ax.text(t_star, ax.get_ylim()[1] * 0.97,
            f" Registration\n threshold (£{int(t_star)}k)",
            ha="left", va="top", fontsize=TICK_SIZE, color="0.3")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Annual turnover (£k)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of firms (thousands per £1k band)", fontsize=LABEL_SIZE)
    ax.legend(frameon=False, fontsize=TICK_SIZE, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    _style_ax(ax)
    _save(fig, f"bunching_analysis_{regime}.png")


def generate_all(vintages=("2023-24", "2024-25")) -> None:
    """Regenerate the bunching figure for the given vintages."""
    print("Generating bunching figures...")
    for vintage in vintages:
        plot_bunching(BunchingEstimator(vintage), vintage)
    print("  done.")
