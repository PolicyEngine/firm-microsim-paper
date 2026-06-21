"""Loaders for the processed ONS + HMRC input tables.

Reads the six processed CSVs from ``data/processed/`` and extracts the
target dictionaries used downstream by :mod:`firm_microsim.calibration`.

Sources:
    * ONS Business Structure Database — firm counts by turnover band and by
      employment-size band, per SIC sector.
    * HMRC VAT Annual Statistics — VAT-registered firm counts and net VAT
      liability, both by turnover band and by trade sector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)

# HMRC turnover-band column order (as it appears in the official tables).
HMRC_BAND_COLUMNS = [
    "Negative_or_Zero",
    "£1_to_Threshold",
    "£Threshold_to_£150k",
    "£150k_to_£300k",
    "£300k_to_£500k",
    "£500k_to_£1m",
    "£1m_to_£10m",
    "Greater_than_£10m",
]

# VAT-liability bands exclude the Negative_or_Zero column for calibration.
VAT_LIABILITY_BAND_COLUMNS = HMRC_BAND_COLUMNS[1:]


@dataclass
class LoadedData:
    """Container for the loaded source frames and derived targets."""

    ons_turnover: pd.DataFrame
    ons_employment: pd.DataFrame
    hmrc_population_band: pd.DataFrame
    hmrc_population_sector: pd.DataFrame
    hmrc_liability_band: pd.DataFrame
    hmrc_liability_sector: pd.DataFrame
    ons_total: int
    hmrc_bands: Dict[str, float]
    vat_liability_bands: Dict[str, float]


def _extract_ons_total(ons_turnover: pd.DataFrame) -> int:
    """Compute the ONS total firm count, ignoring any summary rows."""
    sic_col = ons_turnover["SIC Code"]
    total_row = ons_turnover[sic_col.isna() | (sic_col.astype(str) == "")]
    if len(total_row) > 0 and "Total" in total_row.columns:
        return int(total_row.iloc[0]["Total"])
    sector_rows = ons_turnover[
        ~ons_turnover["Description"].str.contains("Total", na=False)
    ]
    return int(sector_rows["Total"].sum())


def _latest_hmrc_band_targets(hmrc_population_band: pd.DataFrame) -> Dict[str, float]:
    """Extract the latest-year VAT-registered firm counts by turnover band."""
    latest = hmrc_population_band.iloc[-1]
    return {col: float(latest[col]) for col in HMRC_BAND_COLUMNS}


def _latest_vat_liability_bands(
    hmrc_liability_band: pd.DataFrame,
) -> Dict[str, float]:
    """Extract latest-year VAT liability (£m) by turnover band."""
    latest = hmrc_liability_band.iloc[-1]
    return {col: float(latest[col]) for col in HMRC_BAND_COLUMNS}


def load_data(config: Config) -> LoadedData:
    """Load all processed input tables and derive calibration targets.

    Args:
        config: Run configuration providing the processed-data directory and
            input file names.

    Returns:
        A :class:`LoadedData` bundle with raw frames, the ONS total firm
        count, the latest HMRC VAT-registration band targets, and the latest
        VAT-liability-by-band targets.

    Raises:
        FileNotFoundError: If any expected input CSV is missing.
    """
    logger.info("Loading processed input tables from %s", config.processed_dir)

    frames = {
        key: pd.read_csv(config.input_path(key))
        for key in config.input_files
    }

    ons_turnover = frames["ons_turnover"]
    ons_employment = frames["ons_employment"]
    hmrc_population_band = frames["hmrc_population_band"]
    hmrc_population_sector = frames["hmrc_population_sector"]
    hmrc_liability_band = frames["hmrc_liability_band"]
    hmrc_liability_sector = frames["hmrc_liability_sector"]

    for name, frame in frames.items():
        logger.info("  %-24s %6d rows", name, len(frame))

    ons_total = _extract_ons_total(ons_turnover)
    hmrc_bands = _latest_hmrc_band_targets(hmrc_population_band)
    vat_liability_bands = _latest_vat_liability_bands(hmrc_liability_band)

    logger.info("ONS total firms: %s", f"{ons_total:,}")
    logger.info(
        "HMRC VAT-registered firms (latest year): %s",
        f"{sum(hmrc_bands.values()):,.0f}",
    )

    return LoadedData(
        ons_turnover=ons_turnover,
        ons_employment=ons_employment,
        hmrc_population_band=hmrc_population_band,
        hmrc_population_sector=hmrc_population_sector,
        hmrc_liability_band=hmrc_liability_band,
        hmrc_liability_sector=hmrc_liability_sector,
        ons_total=ons_total,
        hmrc_bands=hmrc_bands,
        vat_liability_bands=vat_liability_bands,
    )
