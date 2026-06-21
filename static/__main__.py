"""CLI: ``python -m static`` — regenerate the static VAT-threshold figures."""

from __future__ import annotations

import argparse

from .figures import generate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m static",
        description="Generate static VAT-threshold figures into results/.",
    )
    parser.add_argument("--sweep-vintage", default="2024-25",
                        help="Vintage for the forward sweep (default: 2024-25 / £90k).")
    parser.add_argument("--anchor-vintage", default="2023-24",
                        help="Vintage for the HMRC anchor reform — HMRC's pre-reform "
                        "basis (default: 2023-24 / £85k).")
    parser.add_argument("--year", default="2025-26",
                        help="Fiscal year for the sweep figures (default: 2025-26).")
    args = parser.parse_args()
    generate_all(
        sweep_vintage=args.sweep_vintage,
        anchor_vintage=args.anchor_vintage,
        year=args.year,
    )


if __name__ == "__main__":
    main()
