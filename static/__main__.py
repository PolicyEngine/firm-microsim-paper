"""CLI: ``python -m static`` — regenerate the static VAT-threshold figures."""

from __future__ import annotations

import argparse

from .figures import generate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m static",
        description="Generate static VAT-threshold figures into results/.",
    )
    parser.add_argument("--vintage", default="2024-25",
                        help="Synthetic-data vintage to cost (default: 2024-25 / £90k).")
    parser.add_argument("--year", default="2025-26",
                        help="Fiscal year for the sweep figures (default: 2025-26).")
    args = parser.parse_args()
    generate_all(vintage=args.vintage, year=args.year)


if __name__ == "__main__":
    main()
