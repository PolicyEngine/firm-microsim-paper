"""Loaders for the processed ONS + HMRC input tables.

Reads the six processed CSVs from ``data/processed/`` and extracts the
target dictionaries used downstream by :mod:`firm_microsim.calibration`.

Sources:
    * ONS Business Structure Database — firm counts by turnover band and by
      employment-size band, per SIC sector.
    * HMRC VAT Annual Statistics — VAT-registered firm counts and net VAT
      liability, both by turnover band and by trade sector.
    * OBR Economic and Fiscal Outlook (March 2023), Chart C — HMRC counts of
      businesses by £1,000 turnover band around the £85k threshold, used as
      near-threshold shape targets for the 2023-24 vintage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .config import PROCESSED_DATA_DIR, Config

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

# OBR March-2023 EFO Chart C underlying data: businesses by £1,000 turnover
# band (£65k-£90k), outturn years plus a 2025-26 projection, in thousands.
# Shared across vintages, so it lives in data/processed/ (not a vintage dir).
OBR_NEAR_THRESHOLD_FILE = "obr_vat_bunching.csv"


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
    near_threshold_bins: Optional[pd.DataFrame] = None
    bpe_unregistered: Optional[pd.DataFrame] = None


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


def _near_threshold_targets(config: Config) -> Optional[pd.DataFrame]:
    """OBR Chart C £1,000-band counts, interpolated to the 2023-24 data year.

    Source: OBR Economic and Fiscal Outlook, March 2023, Chart C ("Bunching in
    the VAT turnover distribution at the registration threshold") — HMRC counts
    of businesses by £1,000 turnover band over £65,000-£90,000, with outturn
    years to 2019-20 and a 2025-26 projection under the then-assumed frozen
    £85,000 threshold. The 2023-24 profile is a linear interpolation
    four-sixths of the way from the 2019-20 outturn to the 2025-26 projection,
    matching the OBR's own expected deepening of bunching under the freeze.
    Values are thousands of businesses, converted to firm counts.

    Applied only at the £85,000 threshold (the chart's threshold era): the
    2024-25 (£90k) vintage keeps coarse bands only, since no published fine
    bands exist for the post-rise threshold. Returns a frame with columns
    ``bin_lo_k`` (bin lower edge, £k; bins are (lo, lo+1]) and ``count``.
    """
    if not config.calibrate_near_threshold or config.vat_threshold != 85:
        return None
    path = PROCESSED_DATA_DIR / OBR_NEAR_THRESHOLD_FILE
    if not path.exists():
        logger.warning("Near-threshold targets requested but %s missing", path)
        return None
    df = pd.read_csv(path)
    # Keep bins strictly below £90k: the chart's stated range is £65k-£90k
    # and its right-edge bin shows a boundary upturn we do not trust.
    df = df[df["turnover"] < 90_000].reset_index(drop=True)
    w = 4.0 / 6.0  # 2023-24 sits four years along the 2019-20 -> 2025-26 span
    interp = df["2019-20"] + w * (df["2025-26"] - df["2019-20"])
    out = pd.DataFrame(
        {
            "bin_lo_k": df["turnover"].astype(float) / 1000.0,
            "count": interp.astype(float) * 1000.0,
        }
    )
    logger.info(
        "Near-threshold targets: %d OBR £1k bins over [%.0fk, %.0fk], "
        "interpolated 2023-24 profile, total %s firms",
        len(out),
        out["bin_lo_k"].min(),
        out["bin_lo_k"].max() + 1,
        f"{out['count'].sum():,.0f}",
    )
    return out


def load_data(config: Config) -> LoadedData:
    """Load all processed input tables and derive calibration targets.

    Args:
        config: Run configuration providing the processed-data directory and
            input file names.

    Returns:
        A :class:`LoadedData` bundle with raw frames, the ONS total firm
        count, the latest HMRC VAT-registration band targets, the latest
        VAT-liability-by-band targets, and (for the £85k vintage) the OBR
        near-threshold £1k-band targets.

    Raises:
        FileNotFoundError: If any expected input CSV is missing.
    """
    logger.info("Loading processed input tables from %s", config.processed_dir)

    frames = {
        key: pd.read_csv(config.input_path(key), dtype={"SIC Code": str})
        if key == "bpe_unregistered" and config.input_path(key).exists()
        else pd.read_csv(config.input_path(key))
        for key in config.input_files
        if key != "bpe_unregistered" or config.input_path(key).exists()
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
    near_threshold_bins = _near_threshold_targets(config)

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
        near_threshold_bins=near_threshold_bins,
        bpe_unregistered=frames.get("bpe_unregistered"),
    )
