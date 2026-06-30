"""Stylised illustration of the Liu et al. (2021) VAT-notch mechanism.

Liu, Lockwood, Almunia & Tam (2021, REStat 103(1)) show that the SAME VAT
registration threshold produces BOTH bunching (consumer-facing, low-input firms)
AND voluntary registration (input-heavy, B2B firms). The driver is that
registration carries two opposing effects:

    * input VAT becomes reclaimable     -> a benefit, proportional to the input
      share ``omega``;
    * output VAT must be charged, but is a real burden only on business-to-
      consumer (B2C) sales (B2B buyers reclaim it) -> a cost, proportional to the
      B2C share ``beta``.

The net registration effect on profit is

    pi_registered - pi_unregistered  =  t * y * [ omega - beta * (1 - omega) ],

so it is NEGATIVE for consumer-facing low-input firms (a downward notch -> a
bunching incentive) and POSITIVE for input-heavy B2B firms (registering is a net
gain -> voluntary registration below the threshold).

IMPORTANT — this is a STYLISED illustration, not the paper's costing engine and
not Liu et al.'s full general-equilibrium model:

* Liu et al. use fixed-coefficients (Leontief) production and CES demand; there is
  no iso-elastic ability ``n``. Here the iso-elastic cost ``C(y;n,e)`` is borrowed
  ONLY to give the profit curves curvature for plotting.
* It is not data-grounded: it needs a sector input share ``omega`` and a sector
  B2C share ``beta``, neither of which is currently in the repository (omega would
  come from ONS Supply-Use / ABS value-added-to-turnover; beta from the Supply-Use
  final-demand split). The values below are illustrative firm types.

It is included to document, transparently, why a single value-added wedge (the
B/A2 models) cannot represent the bunching-vs-voluntary-registration split.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from ..config import PAPER_DIR, RESULTS_DIR
from .model import T_STAR, TAU_MAX, schedule_taper

# ---------------------------------------------------------------------------
# House style (matches firm_microsim.figures / dynamic.figures): teal palette,
# 300 dpi, tight bbox, dashed grid alpha 0.3, top/right spines off, NO suptitle.
# ---------------------------------------------------------------------------
PALETTE = ["#326b77", "#122740", "#1b485e", "#568b87", "#80ae9a", "#b5d1ae"]
PRIMARY = "#326b77"
ACCENT = "#d62728"
LABEL_SIZE = 15
TICK_SIZE = 13

PAPER_FIG_DIR = PAPER_DIR / "figures"

# Illustrative firm types: (input share omega, B2C share beta, colour, label).
# Colours drawn from the paper teal palette.
FIRM_TYPES = [
    (0.20, 0.90, PALETTE[0], "ω=0.2, β=0.9 (consumer-facing, bunch)"),
    (0.45, 0.50, PALETTE[3], "ω=0.45, β=0.5 (mixed)"),
    (0.70, 0.20, PALETTE[1], "ω=0.7, β=0.2 (input-heavy B2B, vol. reg.)"),
]

N_SHAPE = 130_000.0  # borrowed ability, for curve shape only (NOT a Liu primitive)
E_SHAPE = 0.17


def iso_cost(y, n=N_SHAPE, e=E_SHAPE):
    """Iso-elastic cost, used only to give the illustrative curves curvature."""
    return n / (1 + 1 / e) * (y / n) ** (1 + 1 / e)


def registration_effect(y, omega, beta, t=TAU_MAX):
    """Net profit effect of registering: t*y*[omega - beta*(1-omega)].

    Positive -> registering is a net gain (voluntary-registration force).
    Negative -> registering is a net cost (downward notch -> bunching force).
    """
    return t * y * (omega - beta * (1 - omega))


def profit_unregistered(y, omega, t=TAU_MAX):
    """Bears input VAT (no reclaim), charges no output VAT."""
    return ((1 - omega) - t * omega) * y - iso_cost(y)


def profit_registered(y, omega, beta, schedule_factor, t=TAU_MAX):
    """Reclaims input VAT; bears output VAT t*f(y) on its B2C value added."""
    f = np.asarray(schedule_factor(y), dtype=float)
    return (1 - omega) * y - t * f * beta * (1 - omega) * y - iso_cost(y)


def _ones(y):
    return np.ones_like(np.atleast_1d(np.asarray(y, dtype=float)))


def chosen_profit(y, omega, beta, schedule_factor, T=T_STAR):
    """Profit under the registration rule.

    Below the threshold the firm may register (at the full rate) or not, taking
    the higher; above the threshold it must register, at the reform schedule rate.
    """
    full = profit_registered(y, omega, beta, _ones)
    reform = profit_registered(y, omega, beta, schedule_factor)
    unreg = profit_unregistered(y, omega)
    return np.where(y >= T, reform, np.maximum(unreg, full))


def _local_maxima(y, p):
    mask = (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])
    return y[1:-1][mask]


def _style_ax(ax) -> None:
    """Apply the shared house style to an axis."""
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def _save(fig, name: str, *, copy_to_paper: bool = True) -> Path:
    """Save to results/<name> at 300 dpi tight bbox, then copy to paper/figures/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  saved {path}")
    if copy_to_paper:
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        dest = PAPER_FIG_DIR / name
        shutil.copyfile(path, dest)
        print(f"  copied {dest}")
    return path


def _mark_optima(ax, y, profit, color) -> None:
    """Mark local maxima (dots) and the global optimum (star) of a profit curve."""
    for v in _local_maxima(y, profit):
        idx = int(np.argmin(np.abs(y - v)))
        ax.plot(v / 1000, profit[idx] / 1000, "o", color=color, ms=8,
                mec="white", zorder=5)
    gi = int(np.argmax(profit))
    ax.plot(y[gi] / 1000, profit[gi] / 1000, "*", color=ACCENT, ms=16,
            mec="white", zorder=6)


def fig_registration_choice(name: str = "liu_registration_choice.png") -> Path:
    """Figure 1 — the registration choice (mechanism), two firm types.

    Each panel plots the unregistered profit (dotted), the full-rate registered
    profit (dotted) and the firm's CHOSEN profit (solid, primary teal) under the
    hard £85k notch, with local maxima (dots), the global optimum (star) and the
    threshold marked. NO figure-level suptitle (the caption lives in LaTeX).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.linspace(40_000, 160_000, 240_001)
    cases = [
        ("Bunching firm (high B2C, low input share)", 0.20, 0.90),
        ("Voluntary registration (high input share, low B2C)", 0.70, 0.20),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (title, omega, beta) in zip(axes, cases):
        pu = profit_unregistered(y, omega)
        pr = profit_registered(y, omega, beta, _ones)
        chosen = chosen_profit(y, omega, beta, _ones)
        ax.plot(y / 1000, pu / 1000, color=PALETTE[2], lw=1.4, ls=":",
                label=r"unregistered  $\pi_u$")
        ax.plot(y / 1000, pr / 1000, color=PALETTE[4], lw=1.4, ls=":",
                label=r"registered (full rate)  $\pi_r$")
        ax.plot(y / 1000, chosen / 1000, color=PRIMARY, lw=2.8,
                label="firm's choice", zorder=4)
        _mark_optima(ax, y, chosen, PRIMARY)
        ax.axvline(T_STAR / 1000, color="gray", ls="--", lw=1.3, alpha=0.7)
        ax.set_title(f"{title}\n" r"$\omega$=%.2g (input share), "
                     r"$\beta$=%.2g (B2C share)" % (omega, beta),
                     fontsize=12)
        ax.set_xlabel("Turnover / sales $y$ (£k)", fontsize=LABEL_SIZE)
        ax.set_ylabel(r"Profit $\pi$ (£k)", fontsize=LABEL_SIZE)
        ax.set_xlim(50, 150)
        ax.legend(frameon=False, fontsize=TICK_SIZE, loc="upper left")
        _style_ax(ax)
    fig.tight_layout()
    return _save(fig, name)


def fig_notch_taper(name: str = "liu_notch_taper.png") -> Path:
    """Figure 2 — notch vs taper across three firm types.

    Two panels (hard notch, graduated taper). Each shows the chosen-profit curve
    for three firm types in three palette colours, with local maxima (dots), the
    global optimum (star) and the £85k threshold marked. NO suptitle.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.linspace(50_000, 160_000, 220_001)
    panels = [("hard notch", _ones), ("graduated taper", schedule_taper)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, (panel_name, sched) in zip(axes, panels):
        for omega, beta, col, lab in FIRM_TYPES:
            ch = chosen_profit(y, omega, beta, sched)
            ax.plot(y / 1000, ch / 1000, color=col, lw=2.3, label=lab, zorder=4)
            _mark_optima(ax, y, ch, col)
        ax.axvline(T_STAR / 1000, color="gray", ls="--", lw=1.3, alpha=0.6)
        ax.set_title(panel_name, fontsize=12)
        ax.set_xlabel("Turnover / sales $y$ (£k)", fontsize=LABEL_SIZE)
        ax.set_ylabel(r"Profit $\pi$ (£k)", fontsize=LABEL_SIZE)
        ax.set_xlim(55, 150)
        _style_ax(ax)
    axes[0].legend(frameon=False, fontsize=11, loc="lower center")
    fig.tight_layout()
    return _save(fig, name)


def make_figures():
    """Write both paper-quality figures to results/ and paper/figures/."""
    return [fig_registration_choice(), fig_notch_taper()]


def build_report():
    lines = [
        "Liu et al. (2021) VAT-notch mechanism — stylised illustration",
        "=" * 64,
        "Net registration effect:  pi_r - pi_u = t*y*[omega - beta*(1-omega)]",
        f"(t={TAU_MAX}, threshold T*=£{T_STAR:,.0f})",
        "",
        "Per illustrative firm type, at y=£100,000:",
    ]
    for omega, beta, _c, lab in FIRM_TYPES:
        eff = registration_effect(100_000.0, omega, beta)
        sign = "voluntary registration" if eff > 0 else "bunching incentive"
        lines.append(
            f"  {lab}: effect = £{eff:,.0f}  ->  {sign}"
        )
    lines += [
        "",
        "Stylised: iso-elastic cost borrowed for curve shape only (Liu et al. use",
        "fixed-coefficients production + CES demand, so there is NO iso-elastic",
        "ability n). Not data-grounded: needs sector input share omega (ONS",
        "Supply-Use / ABS value-added-to-turnover) and B2C share beta (Supply-Use",
        "final-demand split), neither yet in the repository. Documents why a single",
        "value-added wedge (B/A2) cannot represent bunching vs voluntary registration.",
    ]
    return lines


def main(figure=True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = build_report()
    txt = RESULTS_DIR / "liu_mechanism.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {txt}")
    if figure:
        # Writes results/liu_registration_choice.png and results/liu_notch_taper.png,
        # each also copied to paper/figures/ for the LaTeX build.
        for png in make_figures():
            print(f"wrote {png}")


def cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="firm-microsim-liu-mechanism",
        description=(
            "Stylised illustration of the Liu et al. (2021) VAT-notch mechanism "
            "(input VAT reclaim vs B2C output VAT), showing bunching vs voluntary "
            "registration across firm types. Writes results/liu_mechanism.txt plus "
            "the liu_registration_choice.png and liu_notch_taper.png figures to "
            "results/ and paper/figures/."
        ),
    )
    parser.add_argument(
        "--no-figure", action="store_true", help="write the text report only"
    )
    args = parser.parse_args(argv)
    main(figure=not args.no_figure)


if __name__ == "__main__":
    cli()
