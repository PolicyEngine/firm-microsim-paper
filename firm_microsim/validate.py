"""Calibration-accuracy validation against ONS + HMRC targets.

Ports the validation summary of the original generator: compares the
weighted synthetic population against HMRC turnover bands, the ONS total,
ONS employment bands, HMRC sector counts, and HMRC VAT-liability targets
(by sector and by band). Reporting uses the logging framework.

Band edges are driven by the single configurable VAT threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .calibration import EMPLOYMENT_BANDS, VAT_LIABILITY_BANDS
from .config import Config
from .data_loader import LoadedData

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Accuracy scores (0-1) for each calibration dimension."""

    hmrc_bands: float
    ons_population: float
    employment: float
    sector: float
    vat_liability_sector: float
    vat_liability_band: float
    total_population: float

    @property
    def overall(self) -> float:
        """Mean accuracy across the six calibration dimensions."""
        return (
            self.hmrc_bands
            + self.ons_population
            + self.employment
            + self.sector
            + self.vat_liability_sector
            + self.vat_liability_band
        ) / 6.0


def _accuracy(synthetic: float, target: float) -> float:
    """Accuracy = 1 - relative absolute error (with sign-aware handling)."""
    if abs(target) < 1e-9:
        return 1.0 if abs(synthetic) < 1e-9 else 0.0
    if target > 0 and synthetic > 0:
        return 1 - min(abs(synthetic - target) / target, 1.0)
    if target < 0 and synthetic < 0:
        return 1 - min(abs(synthetic - target) / abs(target), 1.0)
    if target > 0:  # simple positive target
        return 1 - abs(synthetic - target) / target
    return max(0.0, 1 - abs(synthetic - target) / max(abs(target), 1.0))


def _hmrc_band_name(turnover_k: float, threshold: float) -> str:
    """Map a turnover value to its HMRC band name."""
    if turnover_k <= 0:
        return "Negative_or_Zero"
    if turnover_k <= threshold:
        return "£1_to_Threshold"
    if turnover_k <= 150:
        return "£Threshold_to_£150k"
    if turnover_k <= 300:
        return "£150k_to_£300k"
    if turnover_k <= 500:
        return "£300k_to_£500k"
    if turnover_k <= 1000:
        return "£500k_to_£1m"
    if turnover_k <= 10000:
        return "£1m_to_£10m"
    return "Greater_than_£10m"


def _employment_band_name(employment: float) -> str:
    """Map an employment count to its ONS band name."""
    if employment <= 4:
        return "0-4"
    if employment <= 9:
        return "5-9"
    if employment <= 19:
        return "10-19"
    if employment <= 49:
        return "20-49"
    if employment <= 99:
        return "50-99"
    if employment <= 249:
        return "100-249"
    return "250+"


def validate(
    synthetic_df: pd.DataFrame,
    data: LoadedData,
    config: Config,
) -> ValidationReport:
    """Validate the synthetic population against all official targets.

    Args:
        synthetic_df: Generated firm-level frame (must include sic_code,
            annual_turnover_k, annual_input_k, employment, weight,
            vat_registered).
        data: Loaded source frames + derived targets.
        config: Run configuration (threshold).

    Returns:
        A :class:`ValidationReport` with per-dimension and overall accuracy.
    """
    threshold = config.vat_threshold
    df = synthetic_df.copy()
    df["hmrc_band"] = df["annual_turnover_k"].apply(lambda t: _hmrc_band_name(t, threshold))
    df["vat_liability_k"] = df["annual_turnover_k"] - df["annual_input_k"]
    df["sic_numeric"] = df["sic_code"].astype(int)
    df["weighted_liability_m"] = df["vat_liability_k"] * df["weight"] / 1000.0
    all_bands = df.groupby("hmrc_band")["weight"].sum()

    # --- HMRC turnover-band accuracy (excludes the ONS-based threshold band).
    hmrc_accs: List[float] = []
    for band_name, target in data.hmrc_bands.items():
        synth = float(all_bands.get(band_name, 0.0))
        if band_name == "£1_to_Threshold":
            continue  # ONS-based, not an HMRC calibration target
        hmrc_accs.append(_accuracy(synth, float(target)))
    hmrc_accuracy = float(np.mean(hmrc_accs)) if hmrc_accs else 0.0

    # --- ONS total population accuracy.
    total_weighted = float(df["weight"].sum())
    ons_accuracy = _accuracy(total_weighted, float(data.ons_total))

    # --- Employment-band accuracy.
    ons_emp_rows = data.ons_employment[
        ~data.ons_employment["Description"].str.contains("Total", na=False)
    ]
    emp_targets: Dict[str, float] = {
        band: float(ons_emp_rows[band].fillna(0).sum())
        if band in ons_emp_rows.columns
        else 0.0
        for band in EMPLOYMENT_BANDS
    }
    df["employment_band"] = df["employment"].apply(_employment_band_name)
    synth_emp = df.groupby("employment_band")["weight"].sum()
    emp_accs = [
        _accuracy(float(synth_emp.get(band, 0.0)), emp_targets[band])
        for band in EMPLOYMENT_BANDS
    ]
    employment_accuracy = float(np.mean(emp_accs)) if emp_accs else 0.0

    # --- Sector accuracy (VAT-registered firms only).
    sector_rows = data.hmrc_population_sector[
        data.hmrc_population_sector["Trade_Sector"] != "Total"
    ]
    vat_registered = df[df["vat_registered"]]
    synth_sector = vat_registered.groupby("sic_numeric")["weight"].sum()
    sector_accs = [
        _accuracy(
            float(synth_sector.get(int(r["Trade_Sector"]), 0.0)), float(r.iloc[-1])
        )
        for _, r in sector_rows.iterrows()
    ]
    sector_accuracy = float(np.mean(sector_accs)) if sector_accs else 0.0

    # --- VAT liability by sector (£m).
    liab_sector_rows = data.hmrc_liability_sector[
        data.hmrc_liability_sector["Trade_Sector"] != "Total"
    ]
    synth_liab_sector = df.groupby("sic_numeric")["weighted_liability_m"].sum()
    liab_sector_accs = [
        _accuracy(
            float(synth_liab_sector.get(int(r["Trade_Sector"]), 0.0)),
            float(r.iloc[-1]),
        )
        for _, r in liab_sector_rows.iterrows()
    ]
    vat_liability_sector_accuracy = (
        float(np.mean(liab_sector_accs)) if liab_sector_accs else 0.0
    )

    # --- VAT liability by turnover band (£m), VAT-registered firms only.
    synth_liab_band = vat_registered.groupby("hmrc_band")["weighted_liability_m"].sum()
    liab_band_accs = [
        _accuracy(
            float(synth_liab_band.get(band, 0.0)),
            float(data.vat_liability_bands[band]),
        )
        for band in VAT_LIABILITY_BANDS
    ]
    vat_liability_band_accuracy = (
        float(np.mean(liab_band_accs)) if liab_band_accs else 0.0
    )

    report = ValidationReport(
        hmrc_bands=hmrc_accuracy,
        ons_population=ons_accuracy,
        employment=employment_accuracy,
        sector=sector_accuracy,
        vat_liability_sector=vat_liability_sector_accuracy,
        vat_liability_band=vat_liability_band_accuracy,
        total_population=total_weighted,
    )

    logger.info("=== CALIBRATION SUMMARY (threshold £%.0fk) ===", threshold)
    logger.info("HMRC Turnover Bands:     %.1f%%", report.hmrc_bands * 100)
    logger.info("ONS Population:          %.1f%%", report.ons_population * 100)
    logger.info("Employment Bands:        %.1f%%", report.employment * 100)
    logger.info("Sector Distribution:     %.1f%%", report.sector * 100)
    logger.info("VAT Liability by Sector: %.1f%%", report.vat_liability_sector * 100)
    logger.info("VAT Liability by Band:   %.1f%%", report.vat_liability_band * 100)
    logger.info("Overall Accuracy:        %.1f%%", report.overall * 100)
    logger.info("Total Population:        %s firms", f"{total_weighted:,.0f}")

    return report
