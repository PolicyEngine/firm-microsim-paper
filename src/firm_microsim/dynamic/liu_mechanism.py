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

import numpy as np

from ..config import RESULTS_DIR
from .model import T_STAR, TAU_MAX, schedule_taper

# Illustrative firm types: (input share omega, B2C share beta, colour, label).
FIRM_TYPES = [
    (0.20, 0.90, "#1f4e5f", "omega=0.2, beta=0.9 (consumer-facing -> bunch)"),
    (0.45, 0.50, "#e08214", "omega=0.45, beta=0.5 (mixed)"),
    (0.70, 0.20, "#c0392b", "omega=0.7, beta=0.2 (input-heavy B2B -> vol. reg.)"),
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


def make_figure(path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.linspace(50_000, 160_000, 300_001)
    panels = [("hard notch", _ones), ("graduated taper", schedule_taper)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, (name, sched) in zip(axes, panels):
        for omega, beta, col, lab in FIRM_TYPES:
            ch = chosen_profit(y, omega, beta, sched)
            ax.plot(y / 1000, ch / 1000, color=col, lw=2.1, label=lab)
            for v in _local_maxima(y, ch):
                ax.plot(
                    v / 1000,
                    ch[np.argmin(np.abs(y - v))] / 1000,
                    "o",
                    color=col,
                    ms=8,
                    mec="white",
                    zorder=5,
                )
            gi = int(np.argmax(ch))
            ax.plot(
                y[gi] / 1000,
                ch[gi] / 1000,
                "*",
                color=col,
                ms=15,
                mec="white",
                zorder=6,
            )
        ax.axvline(T_STAR / 1000, color="grey", ls="--", lw=1, alpha=0.6)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("turnover / sales y (£k)")
        ax.set_ylabel(r"profit $\pi$ (£k)")
        ax.set_xlim(55, 150)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower center")
    fig.suptitle(
        "Liu et al. (2021) mechanism (stylised): "
        r"reg. effect $=t\,y\,[\,\omega-\beta(1-\omega)\,]$"
        "\ninput VAT reclaim (∝ω) vs output VAT on B2C (∝β);  "
        "star=global, dot=local max, dashed=£85k",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
        png = RESULTS_DIR / "liu_mechanism.png"
        make_figure(png)
        print(f"wrote {png}")


def cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="firm-microsim-liu-mechanism",
        description=(
            "Stylised illustration of the Liu et al. (2021) VAT-notch mechanism "
            "(input VAT reclaim vs B2C output VAT), showing bunching vs voluntary "
            "registration across firm types. Writes results/liu_mechanism.{png,txt}."
        ),
    )
    parser.add_argument(
        "--no-figure", action="store_true", help="write the text report only"
    )
    args = parser.parse_args(argv)
    main(figure=not args.no_figure)


if __name__ == "__main__":
    cli()
