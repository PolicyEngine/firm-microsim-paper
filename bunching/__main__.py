"""CLI: ``python -m bunching`` — regenerate the bunching figure and print estimates.

No arguments: regenerate the figure for both vintages and print the reduced-form
estimates (with bootstrap SEs). The structural notch model lives in the ``notch``
package (``python -m notch``).
"""

from __future__ import annotations

import argparse
import logging

from . import figures
from .model import BunchingEstimator


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m bunching",
        description="Regenerate the bunching figure and print estimates.",
    )
    parser.add_argument("--vintage", action="append", choices=["2023-24", "2024-25"],
                        help="Vintage(s) to run (default: both).")
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Bootstrap replications for SEs (default: 200).")
    parser.add_argument("--no-bootstrap", action="store_true",
                        help="Skip bootstrap SEs (point estimates only).")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only regenerate figures; skip printed estimates.")
    args = parser.parse_args()
    logging.getLogger("firm_microsim").setLevel(logging.WARNING)

    vintages = tuple(args.vintage) if args.vintage else ("2023-24", "2024-25")
    figures.generate_all(vintages)

    if args.figures_only:
        return
    for v in vintages:
        print(f"\n{'=' * 60}\n  Bunching — vintage {v}\n{'=' * 60}")
        est = BunchingEstimator(v)
        if args.no_bootstrap:
            res = est.estimate()
            print(f"  b = {res['b']:.4f}   excess mass E = {res['E']:,.0f}   "
                  f"sigma = {res['sigma']:.3f}")
            print(f"  b (LLAT normalisation) = {res['b_llat']:.3f}  "
                  f"(cf. LLAT 2021: 1.361)")
        else:
            print(est.summary(n_boot=args.n_boot).to_string())


if __name__ == "__main__":
    main()
