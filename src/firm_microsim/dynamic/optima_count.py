"""How many optima does the iso-elastic profit function have?

The firm's profit is

    pi(y; n) = (1 - tau * f(y)) * y - C(y; n, e),
    C(y; n, e) = n / (1 + 1/e) * (y / n) ** (1 + 1/e),

where ``f(y) in [0, 1]`` is the fraction of the statutory rate ``tau`` the
schedule levies on whole turnover. Within any *constant*-rate regime the
first-order condition ``pi'(y) = (1 - tau f) - (y/n)**(1/e) = 0`` has a single
root ``y* = n (1 - tau f)**e`` and ``pi`` is strictly concave
(``pi'' = -(1/e)(1/n)(y/n)**(1/e - 1) < 0``), so there is exactly ONE interior
optimum per regime. Extra optima appear only because a *piecewise* schedule
introduces discontinuities (notches): the firm then compares boundary/bunching
candidates across regimes. This module counts the local maxima on a fine grid
for each schedule type and plots them, confirming:

    flat single rate     -> 1 optimum   (unique FOC root)
    hard notch           -> 2 optima    (bunch at T* vs register above)
    reduced-rate band    -> 3 optima    (two notches: T* and band top)
    graduated taper      -> 2, non-concave in band (marginal-rate channel)
"""

from __future__ import annotations

import argparse

import numpy as np

from ..config import RESULTS_DIR
from .model import (
    ELASTICITIES,
    TAU_MAX,
    T_STAR,
    make_schedule_reduced_rate,
    schedule_notch,
    schedule_taper,
)


def _flat_schedule(y):
    """Constant full rate everywhere (registered on all of turnover)."""
    return np.ones_like(np.atleast_1d(np.asarray(y, dtype=float)))


def iso_cost(y, n, e):
    return n / (1 + 1 / e) * (y / n) ** (1 + 1 / e)


def profit(y, n, schedule, e, tau=TAU_MAX):
    """Profit pi(y; n) under ``schedule`` (whole-turnover effective rate)."""
    return (1.0 - tau * np.asarray(schedule(y), dtype=float)) * y - iso_cost(y, n, e)


def local_maxima(y, p):
    """Turnover values of strict interior local maxima on the sampled grid."""
    mask = (p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])
    return y[1:-1][mask]


def build_schedules():
    """The four schedule shapes, label -> (schedule, expected optima)."""
    return {
        "flat single rate": (_flat_schedule, 1),
        "hard notch": (schedule_notch, 2),
        "reduced-rate band 15%": (make_schedule_reduced_rate(0.15), 3),
        "graduated taper": (schedule_taper, 2),
    }


def count_optima(n=130_000.0, e=0.17, y_lo=20_000.0, y_hi=200_000.0, points=400_001):
    """Count local maxima of pi(y; n) for each schedule at ability ``n``."""
    y = np.linspace(y_lo, y_hi, points)
    out = {}
    for label, (schedule, _expected) in build_schedules().items():
        p = profit(y, n, schedule, e)
        ys = local_maxima(y, p)
        out[label] = {
            "n_maxima": int(ys.size),
            "maxima_y": ys,
            "global_argmax": float(y[int(np.argmax(p))]),
        }
    return out


def build_report(e=0.17, abilities=(90_000.0, 120_000.0, 150_000.0)):
    lines = [
        "Optima of the iso-elastic profit function",
        "=" * 64,
        "pi(y;n) = (1 - tau*f(y))*y - n/(1+1/e)*(y/n)**(1+1/e)",
        f"tau={TAU_MAX}, e={e}, T*=£{T_STAR:,.0f}",
        "",
        "Within any constant-rate regime pi is strictly concave, so pi'=0 has a",
        "single root y*=n(1-tau*f)**e. Extra maxima are bunching points created",
        "by the schedule's notch discontinuities, not interior FOC roots.",
        "",
    ]
    for label, (_schedule, expected) in build_schedules().items():
        lines.append(f"### {label}  (expected {expected})")
        for n in abilities:
            res = count_optima(n=n, e=e)[label]
            ys = ", ".join(f"{v / 1000:.0f}k" for v in res["maxima_y"])
            lines.append(
                f"  n=£{n:>8,.0f}: local maxima = {res['n_maxima']}  "
                f"(at {ys})   global argmax = £{res['global_argmax'] / 1000:.1f}k"
            )
        lines.append("")
    return lines


def make_figure(path, n=130_000.0, e=0.17):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.linspace(60_000, 160_000, 300_001)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (label, (schedule, expected)) in zip(
        axes.ravel(), build_schedules().items()
    ):
        p = profit(y, n, schedule, e)
        ax.plot(y / 1000, p / 1000, color="#1f4e5f", lw=2.2)
        for v in local_maxima(y, p):
            ax.plot(
                v / 1000,
                profit(v, n, schedule, e) / 1000,
                "o",
                color="#e08214",
                ms=11,
                zorder=5,
                mec="white",
            )
        gi = int(np.argmax(p))
        ax.plot(
            y[gi] / 1000,
            p[gi] / 1000,
            marker="*",
            color="#c0392b",
            ms=20,
            zorder=6,
            mec="white",
        )
        ax.axvline(T_STAR / 1000, color="grey", ls="--", lw=1, alpha=0.7)
        ax.set_title(f"{label} ({expected} optima)", fontsize=12)
        ax.set_xlabel("turnover y (£k)")
        ax.set_ylabel("profit π(y;n) (£k)")
        ax.set_xlim(70, 150)
        ax.set_ylim(55, 92)
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"Optima of the iso-elastic profit function (n=£{n / 1000:.0f}k, e={e})\n"
        "orange = local maxima,  red star = global optimum,  dashed = £85k threshold",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(e=ELASTICITIES[1], figure=True):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = build_report(e=e)
    txt = RESULTS_DIR / "optima_count.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {txt}")
    if figure:
        png = RESULTS_DIR / "optima_count.png"
        make_figure(png, e=e)
        print(f"wrote {png}")


def cli(argv=None):
    parser = argparse.ArgumentParser(
        prog="firm-microsim-optima-count",
        description=(
            "Count and plot the optima of the iso-elastic profit function for "
            "each VAT schedule shape (flat, notch, reduced-rate band, taper). "
            "Writes results/optima_count.{png,txt}."
        ),
    )
    parser.add_argument(
        "--no-figure", action="store_true", help="write the text report only"
    )
    args = parser.parse_args(argv)
    main(figure=not args.no_figure)


if __name__ == "__main__":
    cli()
