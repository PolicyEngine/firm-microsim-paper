"""CLI entry point: ``python -m firm_microsim``.

Runs the full synthetic-firm generation pipeline. The VAT threshold, seed,
and output path may be overridden on the command line; all other settings
come from :mod:`firm_microsim.config`.
"""

from __future__ import annotations

import argparse
import logging

from .config import DEFAULT_CONFIG, VINTAGES
from .generate import generate


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m firm_microsim",
        description="Generate the synthetic UK firm-level dataset "
        "(ONS + HMRC calibrated).",
    )
    parser.add_argument(
        "--vintage",
        type=str,
        default=None,
        choices=sorted(VINTAGES),
        help="Data vintage: '2023-24' (£85k, paper baseline) or '2024-25' "
        "(£90k, latest gov data). Selects the processed-data subdir and "
        "default threshold. Default: %s." % DEFAULT_CONFIG.data_vintage,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="VAT threshold in £thousands. Overrides the vintage default "
        "(2023-24->85, 2024-25->90).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.seed,
        help="Random seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file name (relative to data/synthetic). "
        f"Default: {DEFAULT_CONFIG.output_file}",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: %(default)s).",
    )
    return parser


def main() -> None:
    """Parse arguments and run generation."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    generate(
        vintage=args.vintage,
        threshold=args.threshold,
        seed=args.seed,
        output=args.output,
    )


if __name__ == "__main__":
    main()
