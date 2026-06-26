"""CLI entry point: ``python -m firm_microsim`` — the full data pipeline.

With no arguments this runs the **complete data build**, one command:

    1. generate the synthetic population for *every* vintage
       (``data/synthetic/synthetic_firms_<vintage>.csv``),
    2. write the calibration report (``results/calibration_accuracy.txt``),
    3. render the descriptive figures (``results/*.png``).

Pass ``--vintage`` (and/or ``--threshold``/``--seed``/``--output``) to generate
a single vintage instead. All other settings come from
:mod:`firm_microsim.config`.
"""

from __future__ import annotations

import argparse
import logging

from .config import DEFAULT_CONFIG, VINTAGES
from .generate import generate


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="firm-microsim",
        description="Build the synthetic UK firm dataset (ONS + HMRC "
        "calibrated). No args = full pipeline over every vintage + report + "
        "figures; --vintage = a single vintage.",
    )
    parser.add_argument(
        "--vintage",
        type=str,
        default=None,
        choices=sorted(VINTAGES),
        help="Generate a SINGLE vintage: '2023-24' (£85k) or '2024-25' (£90k). "
        "Omit to run the full pipeline over all vintages.",
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


def run_pipeline(seed: int) -> None:
    """Full data build: every vintage -> calibration report -> figures."""
    from . import figures, report

    for vintage in VINTAGES:
        logging.info("=== Generating vintage %s ===", vintage)
        generate(vintage=vintage, seed=seed,
                 output=f"synthetic_firms_{vintage}.csv")
    report.main()           # writes results/calibration_accuracy.txt
    figures.generate_all()  # writes results/*.png


def main() -> None:
    """Parse arguments and run the pipeline (all vintages) or a single vintage."""
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    single = (
        args.vintage is not None
        or args.threshold is not None
        or args.output is not None
    )
    if single:
        generate(
            vintage=args.vintage,
            threshold=args.threshold,
            seed=args.seed,
            output=args.output,
        )
    else:
        run_pipeline(seed=args.seed)


if __name__ == "__main__":
    main()
