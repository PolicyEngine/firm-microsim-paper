"""Notch-model figure, in the project house style.

Single clean panel (no embedded title/logo — caption goes in LaTeX), reusing the
shared house-style palette and helpers. Saved to ``results/``:

    * notch_model_fit_<regime>.png  — observed vs notch-model-implied density,
      with the dominated region shaded and the marginal buncher marked.

This is a mechanism illustration (the model-implied bunching spike + empty
dominated region overlaid on the observed synthetic density), not a fitted
overlay. Run ``python -m notch`` to regenerate it.
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

from .model import NotchModel

SHADE = "#e8eef0"   # light shade for the dominated region


def plot_notch_fit(notch: NotchModel, vintage: str) -> None:
    """Observed density with the model's dominated region and marginal buncher.

    Shown at the *observed* scale: the frictionless model would pile every firm
    below the marginal buncher at the threshold (a spike orders of magnitude
    above observed bunching — the gap is optimisation frictions), so plotting
    that spike 1:1 is uninformative. Instead the structural objects (the empty
    dominated region, the marginal buncher) are overlaid on the observed
    density, whose mass *inside* the dominated region is itself the friction.
    """
    regime = VINTAGE_REGIME[vintage]
    print(f"Generating notch_model_fit_{regime}.png...")
    centres, obs_hist, _model_hist, a_kw, n_H = notch.implied_distribution()
    centres = np.asarray(centres)
    obs = np.asarray(obs_hist) / 1000.0
    t_star = notch.t_star

    lo, hi = t_star - 35, max(n_H + 8, t_star + 50)
    m = (centres >= lo) & (centres <= hi)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(t_star, t_star + a_kw, color=SHADE, zorder=0,
               label=f"Dominated region (£{int(a_kw)}k wide)")
    ax.bar(centres[m], obs[m], width=1.0, color=PALETTE[4], zorder=2,
           label="Observed density")
    ymax = float(obs[m].max()) * 1.18
    ax.set_ylim(0, ymax)
    ax.set_xlim(lo, hi)

    ax.axvline(t_star, color="0.35", lw=1, ls=":", zorder=4)
    ax.text(t_star, ymax * 0.97, f" Registration\n threshold (£{int(t_star)}k)",
            ha="left", va="top", fontsize=TICK_SIZE, color="0.3")
    ax.axvline(n_H, color="0.35", lw=1, ls=":", zorder=4)
    ax.text(n_H, ymax * 0.55, f"Marginal buncher\n£{int(round(n_H))}k ",
            ha="right", va="top", fontsize=TICK_SIZE, color="0.3")

    ax.set_xlabel("Annual turnover (£k)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of firms (thousands per £1k band)", fontsize=LABEL_SIZE)
    ax.legend(frameon=False, fontsize=TICK_SIZE, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2)
    _style_ax(ax)
    _save(fig, f"notch_model_fit_{regime}.png")


def generate_all(vintages=("2023-24", "2024-25")) -> None:
    """Regenerate the notch-model figure for the given vintages."""
    print("Generating notch-model figures...")
    for vintage in vintages:
        plot_notch_fit(NotchModel(vintage), vintage)
    print("  done.")
