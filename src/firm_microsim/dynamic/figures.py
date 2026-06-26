"""House-style figures for the dynamic VAT-notch simulator.

Single clean panels (no embedded titles), £k x-axis, teal palette, 300 dpi,
dashed grid alpha 0.3, top/right spines off — matching ``firm_microsim.figures``.
PNGs are written to ``results/`` and the ones used in the paper are also
copied to ``paper/figures/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from firm_microsim.config import PAPER_DIR, RESULTS_DIR

from .model import (
    T_STAR,
    TAPER_TOP,
    dominated_region_width,
    marginal_buncher,
)

# House style.
PALETTE = ["#326b77", "#122740", "#1b485e", "#568b87", "#80ae9a", "#b5d1ae"]
PRIMARY = "#326b77"
ACCENT = "#d62728"
LABEL_SIZE = 15
TICK_SIZE = 13

PAPER_FIG_DIR = PAPER_DIR / "figures"


def _style_ax(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def _save(fig, name: str, *, copy_to_paper: bool = True) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    if copy_to_paper:
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        dest = PAPER_FIG_DIR / name
        shutil.copyfile(path, dest)
        print(f"  copied {dest}")
    return path


def fig_notch_fit(df, e, *, lo=50_000.0, hi=160_000.0, name=None) -> Path:
    """Observed turnover density with the dominated region shaded and n_H(e) marked.

    Clean reconstruction of the old ``notch_model_fit.png`` intent: observed
    weighted turnover density only, the analytic dominated region ``(T*, T*+a)``
    shaded, and the iso-elastic marginal buncher ``n_H(e)`` marked. NO frictionless
    spike, NO fabricated curves.
    """
    t = df["turnover"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float)
    m = (t >= lo) & (t <= hi)
    t, w = t[m], w[m]

    a = dominated_region_width()
    yH, _ = marginal_buncher(e)

    bins = np.arange(lo, hi + 1000.0, 1000.0)
    centres = 0.5 * (bins[:-1] + bins[1:]) / 1000.0  # £k
    counts, _ = np.histogram(t, bins=bins, weights=w)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(centres, counts, width=1.0, color=PRIMARY, zorder=3)

    # Dominated region (T*, T*+a) shaded.
    ax.axvspan(T_STAR / 1000.0, (T_STAR + a) / 1000.0, color=PALETTE[5],
               alpha=0.45, zorder=2,
               label=f"Dominated region (£{T_STAR/1000:.0f}k–£{(T_STAR+a)/1000:.0f}k)")
    ax.axvline(T_STAR / 1000.0, color=ACCENT, ls="--", lw=1.5, zorder=4,
               label=f"VAT threshold (£{T_STAR/1000:.0f}k)")
    ax.axvline(yH / 1000.0, color=PALETTE[1], ls=":", lw=2.0, zorder=4,
               label=f"Marginal buncher $y_H(e={e})$ = £{yH/1000:.0f}k")

    ax.set_xlim(lo / 1000.0, hi / 1000.0)
    ax.set_xlabel("Annual turnover (£k)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Number of firms", fontsize=LABEL_SIZE)
    ax.legend(frameon=False, fontsize=TICK_SIZE, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=2)
    _style_ax(ax)
    if name is None:
        name = f"dynamic_notch_fit_e{str(e).replace('.', '')}.png"
    return _save(fig, name)


def fig_reform_distribution(df, result, reform_label, e, *,
                            lo=60_000.0, hi=120_000.0, name=None) -> Path:
    """Baseline-notch turnover distribution vs the e-governed re-optimised one.

    Both curves come from the same iso-elastic forward-solve machinery (the £85k
    notch baseline vs the reform schedule), so it is an apples-to-apples
    schedule-shape comparison under a single turnover elasticity ``e``. The
    dominated region is shaded and the elasticity used is labelled. NO 80%
    histogram hack, NO "FOC+Uncertainty" curve, NO fabricated bunching spike.
    """
    w = df["weight"].to_numpy(dtype=float)
    t_notch = result["t_notch"]
    t_reform = result["t_new"]

    a = dominated_region_width()
    bins = np.arange(lo, hi + 1000.0, 1000.0)
    centres = 0.5 * (bins[:-1] + bins[1:]) / 1000.0

    h_notch, _ = np.histogram(t_notch, bins=bins, weights=w)
    h_reform, _ = np.histogram(t_reform, bins=bins, weights=w)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(T_STAR / 1000.0, (T_STAR + a) / 1000.0, color=PALETTE[5],
               alpha=0.4, zorder=1, label="Dominated region")
    ax.plot(centres, h_notch, color=PRIMARY, lw=2.4, zorder=3,
            label=f"Baseline: £85k notch (e={e})")
    ax.plot(centres, h_reform, color=ACCENT, lw=2.4, zorder=3,
            label=f"{reform_label} (re-optimised, e={e})")
    ax.axvline(T_STAR / 1000.0, color="gray", ls="--", lw=1.3, alpha=0.8, zorder=2)
    ax.axvline(TAPER_TOP / 1000.0, color="gray", ls=":", lw=1.3, alpha=0.8, zorder=2)

    ax.set_xlim(lo / 1000.0, hi / 1000.0)
    ax.set_xlabel("Annual turnover (£k)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Weighted number of firms (per £1k band)", fontsize=LABEL_SIZE)
    ax.legend(frameon=False, fontsize=TICK_SIZE, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=2)
    _style_ax(ax)
    if name is None:
        slug = "".join(c for c in reform_label.lower().split(":")[0]
                       if c.isalnum())
        name = f"dynamic_reform_distribution_{slug}_e{str(e).replace('.', '')}.png"
    return _save(fig, name)


def fig_cost_vs_elasticity(df, name="dynamic_cost_vs_elasticity.png"):
    """Behavioural reform cost as a function of the assumed elasticity ``e``.

    For each of the three flat-rate reforms, plots the behavioural revenue cost
    (£m) against ``e``, with the static cost at ``e=0`` (the nesting limit). This
    shows the e-sensitivity range directly, without any near-threshold
    distribution dynamics — which the intensive-margin model cannot credibly
    price (it omits the extensive un-bunching a reduced rate would induce).
    """
    from .model import (reform_revenue, make_schedule_raise,
                        make_schedule_reduced_rate)

    reforms = [
        ("Raise threshold to £100k", make_schedule_raise(100_000.0), PALETTE[0], "o"),
        ("Reduced rate 10% (£85k–£105k)", make_schedule_reduced_rate(0.10), PALETTE[3], "s"),
        ("Reduced rate 15% (£85k–£105k)", make_schedule_reduced_rate(0.15), ACCENT, "^"),
    ]
    es = np.array([0.0, 0.02, 0.05, 0.08, 0.12, 0.17, 0.22, 0.27, 0.32, 0.40])

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, sched, color, mk in reforms:
        costs = []
        for e in es:
            if e == 0.0:
                r = reform_revenue(df, sched, 0.001, behavioural=False)
            else:
                r = reform_revenue(df, sched, float(e), behavioural=True)
            costs.append(r["d_rev"] / 1e6)
        ax.plot(es, costs, color=color, lw=2.4, marker=mk, ms=5, label=label)

    for e_mark in (0.05, 0.17, 0.32):
        ax.axvline(e_mark, color="gray", ls=":", lw=1.0, alpha=0.6, zorder=1)
    ax.set_xlabel(r"Assumed turnover elasticity $e$", fontsize=LABEL_SIZE)
    ax.set_ylabel("Behavioural revenue cost (£m)", fontsize=LABEL_SIZE)
    ax.set_xlim(-0.005, 0.405)
    ax.legend(frameon=False, fontsize=TICK_SIZE, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=1)
    _style_ax(ax)
    return _save(fig, name)
