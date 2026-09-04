"""The two ONS input tables count the same statistical unit (issue #37)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from firm_microsim.config import PROCESSED_DATA_DIR, VINTAGES

REPO = Path(__file__).resolve().parents[1]


def _sic_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[~df["Description"].str.contains("Total", na=False) & df["SIC Code"].notna()]


@pytest.mark.parametrize("vintage", sorted(VINTAGES))
def test_employment_and_turnover_tables_agree_to_rounding(vintage: str) -> None:
    emp = _sic_rows(PROCESSED_DATA_DIR / vintage / "ons_firm_employment.csv")
    turn = _sic_rows(PROCESSED_DATA_DIR / vintage / "ons_firm_turnover.csv")
    assert len(emp) == len(turn) == 88
    # ONS rounds cells independently to the nearest 5: per-SIC totals may
    # differ by a few units, never by the local-unit/enterprise gap (~15%).
    diff = (emp["Total"].to_numpy() - turn["Total"].to_numpy())
    assert abs(diff).max() <= 20, "employment table is not on the enterprise unit"
    assert abs(emp["Total"].sum() - turn["Total"].sum()) <= 50


@pytest.mark.skipif(
    not (REPO / "data" / "raw" / "ons" / "ukbusinessworkbook2024.xlsx").exists(),
    reason="raw ONS workbooks not present in this checkout",
)
def test_processed_ons_tables_match_etl() -> None:
    """The checked CSVs are exactly what scripts/etl_ons_tables.py produces."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "etl_ons_tables.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
