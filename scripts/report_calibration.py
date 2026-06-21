#!/usr/bin/env python
"""Reproducible calibration report for the synthetic UK firm populations.

For every data vintage declared in :data:`firm_microsim.config.VINTAGES`
(currently ``2023-24`` @ £85k and ``2024-25`` @ £90k) this script:

    1. builds the matching :class:`~firm_microsim.config.Config` via
       :meth:`Config.for_vintage`,
    2. loads the previously generated synthetic firm CSV from
       ``data/synthetic/synthetic_firms_<vintage>.csv`` (skipping the vintage
       with a clear message if that file has not been generated yet),
    3. loads the official ONS + HMRC targets with
       :func:`firm_microsim.data_loader.load_data`,
    4. scores the synthetic frame against those targets with
       :func:`firm_microsim.validate.validate`, and
    5. prints an aligned per-dimension accuracy / error table plus the
       all-six ``overall`` accuracy and a five-dimension ``headline``
       accuracy (excluding ``vat_liability_sector``).

It is intentionally dependency-light: it only reads CSVs and runs the
existing validator (no torch, no weight generation), so it is fast.

Run from the repository root::

    python scripts/report_calibration.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# --- Make `import firm_microsim` work regardless of the CWD ----------------
# scripts/report_calibration.py lives in <repo>/scripts/, so the repo root is
# one level up. Put it first on sys.path so the package imports cleanly when
# the script is invoked as `python scripts/report_calibration.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from firm_microsim.config import VINTAGES, Config  # noqa: E402
from firm_microsim.data_loader import load_data  # noqa: E402
from firm_microsim.validate import validate  # noqa: E402

# Each tuple is (attribute on ValidationReport, human-friendly label).
# Order matches the original calibration summary.
_DIMENSIONS = [
    ("hmrc_bands", "HMRC Turnover Bands"),
    ("ons_population", "ONS Population"),
    ("employment", "Employment Bands"),
    ("sector", "Sector Distribution"),
    ("vat_liability_sector", "VAT Liability by Sector"),
    ("vat_liability_band", "VAT Liability by Band"),
]


def _report_vintage(vintage: str) -> bool:
    """Print the calibration report for one vintage.

    Returns ``True`` if a report was produced, ``False`` if the synthetic
    CSV was missing (and the vintage was therefore skipped).
    """
    cfg = Config.for_vintage(vintage)
    csv_path = cfg.synthetic_dir / f"synthetic_firms_{vintage}.csv"

    print()
    print("=" * 64)
    print(f"  Vintage {vintage}  |  threshold £{int(cfg.vat_threshold)}k")
    print("=" * 64)

    if not csv_path.exists():
        print(f"  skip: {csv_path} not found — run generation first.")
        return False

    df = pd.read_csv(csv_path)
    data = load_data(cfg)
    rep = validate(df, data, cfg)

    weighted_pop = float(df["weight"].sum())

    print(f"  rows (firm types):   {len(df):,}")
    print(f"  weighted population: {weighted_pop:,.0f} firms")
    print(f"  (validator total):   {rep.total_population:,.0f} firms")
    print("-" * 64)
    print(f"  {'Dimension':<26}{'accuracy':>12}{'error':>12}")
    print("-" * 64)
    for attr, label in _DIMENSIONS:
        acc = getattr(rep, attr) * 100.0
        err = 100.0 - acc
        print(f"  {label:<26}{acc:>11.1f}%{err:>11.1f}%")
    print("-" * 64)

    overall = rep.overall * 100.0
    # Compute the 5-dim headline locally (exclude vat_liability_sector) so the
    # script does not depend on the report class exposing a `headline` property.
    headline = (
        (
            rep.hmrc_bands
            + rep.ons_population
            + rep.employment
            + rep.sector
            + rep.vat_liability_band
        )
        / 5.0
        * 100.0
    )
    print(f"  {'Overall (all 6 dims)':<26}{overall:>11.1f}%{100.0 - overall:>11.1f}%")
    print(
        f"  {'Headline (5 core dims)':<26}{headline:>11.1f}%"
        f"{100.0 - headline:>11.1f}%"
    )
    print("=" * 64)
    return True


def main() -> None:
    """Report calibration accuracy for every configured vintage."""
    # Silence the INFO chatter emitted by data_loader / validate so the table
    # is the only thing on stdout.
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("firm_microsim").setLevel(logging.WARNING)

    print("Calibration report — synthetic UK firm populations")
    produced = 0
    for vintage in VINTAGES:
        if _report_vintage(vintage):
            produced += 1

    print()
    if produced == 0:
        print("No vintages reported — generate the synthetic CSVs first.")
    else:
        print(f"Done: {produced}/{len(VINTAGES)} vintage(s) reported.")


if __name__ == "__main__":
    main()
