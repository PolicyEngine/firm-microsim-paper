"""Calibration report — per-dimension accuracy/error for every vintage.

Loads each saved synthetic population, scores it against the official ONS + HMRC
targets, and renders an aligned table (the five calibrated dimensions, the
overall, and the informational sector-liability diagnostic). Writes the report
to ``results/calibration_accuracy.txt`` and prints it.

    python -m firm_microsim.report      # standalone
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from .config import RESULTS_DIR, VINTAGES, Config
from .data_loader import load_data
from .validate import validate

# The five dimensions that are calibration targets (drive `overall`).
_CALIBRATED_DIMENSIONS = [
    ("hmrc_bands", "HMRC Turnover Bands"),
    ("ons_population", "ONS Population"),
    ("employment", "Employment Bands"),
    ("sector", "Sector Distribution"),
    ("vat_liability_band", "VAT Liability by Band"),
]


def _vintage_lines(vintage: str) -> list[str]:
    """Return the report lines for one vintage (or a skip notice)."""
    cfg = Config.for_vintage(vintage)
    csv_path = cfg.synthetic_dir / f"synthetic_firms_{vintage}.csv"

    out = ["", "=" * 64,
           f"  Vintage {vintage}  |  threshold £{int(cfg.vat_threshold)}k",
           "=" * 64]
    if not csv_path.exists():
        out.append(f"  skip: {csv_path} not found — run generation first.")
        return out

    df = pd.read_csv(csv_path)
    rep = validate(df, load_data(cfg), cfg)

    out += [
        f"  rows (firm types):   {len(df):,}",
        f"  weighted population: {float(df['weight'].sum()):,.0f} firms",
        "-" * 64,
        f"  {'Dimension':<26}{'accuracy':>12}{'error':>12}",
        "-" * 64,
    ]
    for attr, label in _CALIBRATED_DIMENSIONS:
        acc = getattr(rep, attr) * 100.0
        out.append(f"  {label:<26}{acc:>11.1f}%{100.0 - acc:>11.1f}%")
    out.append("-" * 64)
    overall = rep.overall * 100.0
    out.append(
        f"  {'Overall (5 calibrated dims)':<26}{overall:>11.1f}%"
        f"{100.0 - overall:>11.1f}%"
    )
    out.append("-" * 64)
    diag = rep.vat_liability_sector * 100.0
    out.append("  Informational diagnostic (not a calibration target):")
    out.append(
        f"  {'VAT Liability by Sector':<26}{diag:>11.1f}%{100.0 - diag:>11.1f}%"
    )
    out.append("=" * 64)
    return out


def build_report() -> str:
    """Build the full calibration report for every configured vintage."""
    lines = ["Calibration report — synthetic UK firm populations"]
    produced = 0
    for vintage in VINTAGES:
        block = _vintage_lines(vintage)
        lines += block
        if not any("skip:" in ln for ln in block):
            produced += 1
    lines.append("")
    lines.append(
        f"Done: {produced}/{len(VINTAGES)} vintage(s) reported."
        if produced
        else "No vintages reported — generate the synthetic CSVs first."
    )
    return "\n".join(lines) + "\n"


def main(write: bool = True) -> str:
    """Build the report, write it to results/, print it, and return it."""
    logging.getLogger("firm_microsim").setLevel(logging.WARNING)
    report = build_report()
    if write:
        (RESULTS_DIR / "calibration_accuracy.txt").write_text(report)
    print(report)
    return report


def cli(argv: list[str] | None = None) -> str:
    """Console entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the report without writing results/calibration_accuracy.txt.",
    )
    args = parser.parse_args(argv)
    return main(write=not args.no_write)


if __name__ == "__main__":
    cli()
