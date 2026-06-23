"""CLI: ``python -m notch`` — regenerate the notch-model figure and print the summary."""

from __future__ import annotations

import argparse
import logging

from . import figures
from .model import NotchModel


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m notch",
        description="Regenerate the structural notch-model figure and print its summary.",
    )
    parser.add_argument("--vintage", action="append", choices=["2023-24", "2024-25"],
                        help="Vintage(s) to run (default: both).")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only regenerate the figure; skip the printed summary.")
    args = parser.parse_args()
    logging.getLogger("firm_microsim").setLevel(logging.WARNING)

    vintages = tuple(args.vintage) if args.vintage else ("2023-24", "2024-25")
    figures.generate_all(vintages)

    if args.figures_only:
        return
    for v in vintages:
        print(f"\n{'=' * 60}\n  Notch model — vintage {v}\n{'=' * 60}")
        for k, val in NotchModel(v).summary().items():
            print(f"    {k:24s} {val}")


if __name__ == "__main__":
    main()
