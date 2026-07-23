"""Optima of the value-added "formulation A" profit function across VAT schedules.

This module draws the firm-problem geometry behind Section~\\ref{sec:behavioural}
of the paper. It plots the value-added (formulation~A) profit

    pi_A(y) = (1 - delta) * (1 - tau * f(y)) * y - C(y; n, e),
    C(y; n, e) = (n / (1 + 1/e)) * (y / n) ** (1 + 1/e),

for the statutory rate ``tau = TAU_MAX`` and a fixed ability ``n`` and turnover
elasticity ``e``, under two reform schedules ``f(y)`` -- the hard registration
notch and the graduated taper -- and three deductible-input shares
``delta in {0.0, 0.4, 0.8}``.  Each curve's local maxima are marked with dots and
its global optimum with a star; a dashed vertical line marks the
\\pounds85{,}000 threshold.  The value-added factor ``(1 - delta)`` simply scales
the whole curve, while a higher ``delta`` shrinks the retained wedge
``(1 - delta)(1 - tau)`` and so pulls the registered optimum
``y* = n[(1 - delta)(1 - tau)]**e`` back toward the threshold.

House style matches :mod:`firm_microsim.dynamic.figures`: single teal palette,
300 dpi, tight bounding box, dashed grid, top/right spines off, small per-axis
titles (no figure suptitle), legend in one panel only.  The PNG is written to
``results/`` and copied to ``paper/figures/``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from firm_microsim.config import PAPER_DIR, RESULTS_DIR

from .model import (
    TAU_MAX,
    T_STAR,
    iso_cost,
    schedule_notch,
    schedule_taper,
)

# ---------------------------------------------------------------------------
# House style (teal palette, sizes) -- mirrors firm_microsim.dynamic.figures.
# ---------------------------------------------------------------------------
PALETTE = ["#326b77", "#1b485e", "#122740", "#568b87", "#80ae9a", "#b5d1ae"]
LABEL_SIZE = 15
TICK_SIZE = 13

PAPER_FIG_DIR = PAPER_DIR / "figures"
FIG_NAME = "formulation_a_optima.png"

# ---------------------------------------------------------------------------
# Fixed parameters for the illustration (match Section 7 \eqref{eq:isoelastic}).
# ---------------------------------------------------------------------------
N_ABILITY = 130_000.0          # frictionless ability / turnover (pounds)
E_DEFAULT = 0.17               # illustrative turnover elasticity
DELTAS = (0.0, 0.4, 0.8)       # deductible-input shares to overlay

# Turnover grid (pounds); step divides T_STAR so the notch edge sits on a node.
Y_MIN, Y_MAX, Y_STEP = 30_000.0, 180_000.0, 25.0

SCHEDULES = (
    ("hard notch", schedule_notch),
    ("graduated taper", schedule_taper),
)


# ---------------------------------------------------------------------------
# Formulation-A value-added profit and optima detection
# ---------------------------------------------------------------------------
def pi_a(y, n, e, delta, schedule, tau=TAU_MAX):
    """Value-added (formulation~A) profit ``pi_A(y)`` (pounds).

    ``pi_A(y) = (1 - delta)(1 - tau f(y)) y - C(y; n, e)`` where the schedule
    factor ``f(y) = schedule(y) in [0, 1]`` is the fraction of the statutory
    rate ``tau`` levied once registered and ``C`` is the iso-elastic own-factor
    cost :func:`firm_microsim.dynamic.model.iso_cost`.
    """
    y = np.asarray(y, dtype=float)
    f = np.asarray(schedule(y), dtype=float)
    net = (1.0 - delta) * (1.0 - tau * f)
    return net * y - iso_cost(y, n, e)


def local_maxima(pi):
    """Indices of interior local maxima of ``pi`` on its grid.

    A node is a local maximum if it is at least as high as its left neighbour
    and strictly higher than its right neighbour.  This catches both the smooth
    interior optimum and the boundary "bunching" peak that the hard notch's
    discontinuity creates just below the threshold.
    """
    pi = np.asarray(pi, dtype=float)
    interior = np.arange(1, pi.size - 1)
    is_max = (pi[interior] >= pi[interior - 1]) & (pi[interior] > pi[interior + 1])
    return interior[is_max]


def schedule_optima(n, e, delta, schedule, *, y=None):
    """Grid, profit, local-maxima indices and global-optimum index for a curve.

    Returns ``(y, pi, maxima_idx, global_idx)`` with ``y``/``pi`` in pounds and
    ``maxima_idx`` the interior local maxima (always including the global one).
    """
    if y is None:
        y = np.arange(Y_MIN, Y_MAX + Y_STEP, Y_STEP)
    pi = pi_a(y, n, e, delta, schedule)
    maxima_idx = local_maxima(pi)
    if maxima_idx.size == 0:
        maxima_idx = np.array([int(np.argmax(pi))])
    global_idx = int(maxima_idx[np.argmax(pi[maxima_idx])])
    return y, pi, maxima_idx, global_idx


# ---------------------------------------------------------------------------
# Styling / saving
# ---------------------------------------------------------------------------
def _style_ax(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(path, *, n=N_ABILITY, e=E_DEFAULT, deltas=DELTAS) -> Path:
    """Draw the two-panel formulation-A optima figure and save it to ``path``."""
    path = Path(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, (title, schedule) in zip(axes, SCHEDULES):
        for color, delta in zip(PALETTE, deltas):
            y, pi, maxima_idx, global_idx = schedule_optima(n, e, delta, schedule)
            ax.plot(
                y / 1000.0, pi / 1000.0, color=color, lw=2.2, zorder=3,
                label=rf"$\delta={delta:.1f}$",
            )
            # Local maxima (dots).
            ax.scatter(
                y[maxima_idx] / 1000.0, pi[maxima_idx] / 1000.0,
                s=42, color=color, edgecolor="white", linewidth=0.8,
                zorder=5,
            )
            # Global optimum (star).
            ax.scatter(
                y[global_idx] / 1000.0, pi[global_idx] / 1000.0,
                marker="*", s=320, color=color, edgecolor="white",
                linewidth=0.9, zorder=6,
            )
        ax.axvline(T_STAR / 1000.0, color="gray", ls="--", lw=1.3, alpha=0.85,
                   zorder=2)
        ax.set_title(title, fontsize=LABEL_SIZE)
        ax.set_xlabel("Annual turnover (£k)", fontsize=LABEL_SIZE)
        ax.set_xlim(Y_MIN / 1000.0, Y_MAX / 1000.0)
        _style_ax(ax)

    axes[0].set_ylabel(r"Value-added profit $\pi_A$ (£k)", fontsize=LABEL_SIZE)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=TICK_SIZE,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        title="deductible share",
    )
    legend.get_title().set_fontsize(TICK_SIZE)

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def main() -> Path:
    """Build the figure in ``results/`` and copy it into ``paper/figures/``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = make_figure(RESULTS_DIR / FIG_NAME)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    dest = PAPER_FIG_DIR / FIG_NAME
    shutil.copyfile(out, dest)
    print(f"  copied {dest}")
    return out


def cli() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot the optima of the value-added (formulation A) profit function "
            "across the hard-notch and graduated-taper VAT schedules."
        )
    )
    parser.add_argument(
        "--no-copy", action="store_true",
        help="write only to results/ and do not copy into paper/figures/",
    )
    args = parser.parse_args()
    if args.no_copy:
        make_figure(RESULTS_DIR / FIG_NAME)
    else:
        main()


if __name__ == "__main__":  # pragma: no cover
    cli()
