"""Synthetic UK firm-population generator (orchestration).

A two-stage, multi-source microsimulation generator for the UK business
population, calibrated to official statistics:

    Stage 1 — Draw a base population from the **ONS Business Structure
    Database**. For each SIC sector and ONS turnover band, draw individual
    firms with realistic within-band turnover (uniform base + Gaussian noise
    smoothing). Draw input expenditure (Beta-distributed input/output ratios
    with sector-specific shifts) and employment (ONS employment-band shares).

    Stage 2 — Calibrate per-firm weights via multi-objective optimization so
    the *weighted* population simultaneously matches **HMRC VAT Annual
    Statistics** (firm counts by turnover band and by sector, net VAT liability
    by band and by sector) and ONS employment-band counts. Turnover bands are
    weighted ~5x; VAT-liability-by-band 2x. A symmetric-relative-error loss
    balances targets across scales. Below-threshold zero/negative-turnover
    firms are added manually from the HMRC Negative_or_Zero target.

VAT registration is then assigned: mandatory above the (single, configurable)
VAT threshold, plus voluntary below it at a rate calibrated from HMRC's
£1_to_Threshold count.

Output: ~2.94M weighted rows summing to ~2.0M firms, written to
``data/synthetic/synthetic_firms.csv`` with columns
``sic_code, annual_turnover_k, annual_input_k, vat_liability_k, employment,
weight, vat_registered``. (A ``productivity`` column is added downstream.)

Sources:
    * ONS Business Structure Database (firm counts by turnover & employment).
    * HMRC VAT Annual Statistics (VAT population & liability by band & sector).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .calibration import (
    EMPLOYMENT_BANDS,
    _employment_band_index,
    build_target_matrix,
    map_to_hmrc_bands,
    optimize_weights,
)
from .config import Config, DEFAULT_CONFIG
from .data_loader import LoadedData, load_data
from .validate import ValidationReport, validate

logger = logging.getLogger(__name__)

# ONS turnover-band parameters: (min, max, midpoint) in £k.
ONS_TURNOVER_BANDS: Dict[str, tuple] = {
    "0-49": (0, 49, 24.5),
    "50-99": (50, 99, 74.5),
    "100-249": (100, 249, 174.5),
    "250-499": (250, 499, 374.5),
    "500-999": (500, 999, 749.5),
    "1000-4999": (1000, 4999, 2999.5),
    "5000+": (5000, 50000, 15000),
}

# ONS employment-band parameters: (min, max, midpoint).
ONS_EMPLOYMENT_BANDS: Dict[str, tuple] = {
    "0-4": (1, 4, 2.5),
    "5-9": (5, 9, 7),
    "10-19": (10, 19, 14.5),
    "20-49": (20, 49, 34.5),
    "50-99": (50, 99, 74.5),
    "100-249": (100, 249, 174.5),
    "250+": (250, 2000, 400),
}

# Sectors that tend to have negative VAT liability (input ratio biased up).
_NEG_LIABILITY_SECTORS = {1, 3, 6, 7, 9, 10, 24, 30, 36, 37, 49, 50, 51, 60, 64, 79, 84}
# Sectors that tend to have high VAT liability (input ratio biased down).
_HIGH_LIABILITY_SECTORS = {11, 12, 69, 70, 78}


def _draw_band_turnover(
    count: int, min_t: float, max_t: float, device: str
) -> Tensor:
    """Draw within-band turnover values with Gaussian noise smoothing."""
    if count == 0:
        return torch.empty(0, device=device)
    band_width = max_t - min_t
    noise_std = max(25.0, band_width * 0.2)
    base = min_t + torch.rand(count, device=device) * band_width
    noise = torch.normal(0, noise_std, (count,), device=device)
    return torch.clamp(base + noise, min=0.1)


def generate_base_firms(
    ons_turnover: pd.DataFrame, device: str
) -> tuple[Tensor, Tensor]:
    """Draw the base firm population from ONS turnover structure.

    Args:
        ons_turnover: ONS turnover-band table (one row per SIC sector).
        device: Torch device.

    Returns:
        Tuple of (sic_codes int64 tensor, turnover float32 tensor).
    """
    logger.info("Generating base firms from ONS structure...")
    all_sic: list[int] = []
    all_turnover: list[float] = []

    for _, row in ons_turnover.iterrows():
        sic_code = row["SIC Code"]
        if pd.isna(sic_code) or str(sic_code) in ("", "Total"):
            continue
        sic_int = int(sic_code)
        for band, (min_t, max_t, _mid) in ONS_TURNOVER_BANDS.items():
            if band in row and pd.notna(row[band]) and row[band] > 0:
                count = int(row[band])
                if count <= 0:
                    continue
                turnovers = _draw_band_turnover(count, min_t, max_t, device)
                all_sic.extend([sic_int] * count)
                all_turnover.extend(turnovers.cpu().numpy())

    sic_tensor = torch.tensor(all_sic, dtype=torch.int64, device=device)
    turnover_tensor = torch.tensor(all_turnover, dtype=torch.float32, device=device)
    logger.info("Generated %s base firms", f"{len(all_sic):,}")
    return sic_tensor, turnover_tensor


def generate_input_values(
    turnover_values: Tensor, sic_codes: Tensor, device: str
) -> Tensor:
    """Draw per-firm input expenditure (£k) from Beta input/output ratios.

    Most firms get inputs at 60-95% of turnover (positive VAT liability);
    sector-specific shifts allow some firms inputs > turnover (negative
    liability), matching HMRC's negative net-liability sectors.

    Args:
        turnover_values: Per-firm turnover (£k).
        sic_codes: Per-firm SIC codes.
        device: Torch device.

    Returns:
        Per-firm input expenditure (£k).
    """
    logger.info("Generating input values...")
    n_firms = len(turnover_values)

    base_ratios = torch.distributions.Beta(4.0, 2.0).sample((n_firms,)).to(device)
    scaled = 0.3 + base_ratios * 1.0  # map [0,1] -> [0.3, 1.3]
    sector_noise = torch.randn(n_firms, device=device) * 0.15

    sic_np = sic_codes.cpu().numpy()
    neg_mask = np.isin(sic_np, list(_NEG_LIABILITY_SECTORS))
    high_mask = np.isin(sic_np, list(_HIGH_LIABILITY_SECTORS))
    neg_t = torch.tensor(neg_mask, device=device)
    high_t = torch.tensor(high_mask, device=device)

    scaled = scaled + neg_t.float() * (torch.rand(n_firms, device=device) * 0.3)
    scaled = scaled - high_t.float() * (torch.rand(n_firms, device=device) * 0.2)

    # Floor the input/output ratio at 0.6 (value-added <= 40% of turnover).
    # A lower floor lets a few firms get implausibly low input (VA up to 90%
    # of turnover), which both produces unrealistic per-firm VAT liabilities
    # and lets the weight optimiser concentrate huge weights on those outliers
    # to hit the liability targets — distorting the near-threshold density and
    # the static threshold sweep. 0.6 matches the documented 60-95% intent.
    final_ratios = torch.clamp(scaled + sector_noise, 0.6, 1.5)
    input_values = torch.where(
        turnover_values > 0, turnover_values * final_ratios, torch.zeros_like(turnover_values)
    )

    logger.info(
        "Input/output ratio: mean=%.2f std=%.2f; firms with negative VAT liability: %s",
        final_ratios.mean().item(),
        final_ratios.std().item(),
        f"{int((final_ratios > 1.0).sum().item()):,}",
    )
    return input_values


def assign_employment(
    num_firms: int, ons_employment: pd.DataFrame, device: str
) -> Tensor:
    """Assign employment counts using ONS employment-band shares.

    Args:
        num_firms: Number of firms to assign.
        ons_employment: ONS employment-band table.
        device: Torch device.

    Returns:
        Per-firm employment counts (float32).
    """
    logger.info("Assigning employment from ONS distribution...")
    sector_rows = ons_employment[
        ~ons_employment["Description"].str.contains("Total", na=False)
    ]
    band_counts = {
        band: int(sector_rows[band].fillna(0).sum()) if band in sector_rows.columns else 0
        for band in EMPLOYMENT_BANDS
    }
    total = sum(band_counts.values()) or 1

    values: list[float] = []
    for band in EMPLOYMENT_BANDS:
        target = int(round(num_firms * band_counts[band] / total))
        if target <= 0:
            continue
        min_v, max_v, midpoint = ONS_EMPLOYMENT_BANDS[band]
        if band == "0-4":
            v = torch.randint(1, 5, (target,), device=device).float()
        elif band == "250+":
            log_mean = torch.log(torch.tensor(float(midpoint), device=device))
            v = torch.normal(log_mean, 0.8, (target,), device=device).exp()
            v = torch.clamp(v, min_v, max_v).round()
        else:
            u = torch.rand(target, device=device)
            beta_approx = u.pow(0.5) * (1 - u).pow(2.0)
            v = (min_v + beta_approx * (max_v - min_v)).round()
        values.extend(v.cpu().numpy())

    np.random.shuffle(values)
    if len(values) < num_firms:
        values.extend([1] * (num_firms - len(values)))
    else:
        values = values[:num_firms]

    return torch.tensor(values, dtype=torch.float32, device=device)


def assign_vat_flags(
    turnover_values: Tensor, hmrc_bands: Dict[str, float], config: Config
) -> Tensor:
    """Assign VAT registration flags.

    Mandatory above the configurable threshold; voluntary below it at a rate
    calibrated to HMRC's £1_to_Threshold count.

    Args:
        turnover_values: Per-firm turnover (£k).
        hmrc_bands: HMRC band targets (provides the £1_to_Threshold count).
        config: Run configuration (threshold + device).

    Returns:
        Boolean tensor of VAT-registration status.
    """
    logger.info("Assigning VAT registration flags...")
    threshold = config.vat_threshold
    device = config.device

    below = (turnover_values > 0) & (turnover_values <= threshold)
    n_below = int(below.sum().item())
    target_below = float(hmrc_bands["£1_to_Threshold"])
    voluntary_rate = target_below / n_below if n_below > 0 else 0.15
    logger.info(
        "Voluntary VAT rate: %.3f (target %s / synthetic %s)",
        voluntary_rate,
        f"{target_below:,.0f}",
        f"{n_below:,}",
    )

    mandatory = turnover_values > threshold
    if n_below > 0:
        voluntary = below & (torch.rand(len(turnover_values), device=device) < voluntary_rate)
    else:
        voluntary = torch.zeros_like(below)
    vat_registered = mandatory | voluntary
    logger.info(
        "VAT: %d mandatory + %d voluntary = %d registered",
        int(mandatory.sum().item()),
        int(voluntary.sum().item()),
        int(vat_registered.sum().item()),
    )
    return vat_registered


def _add_zero_turnover_firms(
    sic_codes: Tensor,
    turnover: Tensor,
    input_values: Tensor,
    weights: Tensor,
    hmrc_bands: Dict[str, float],
    device: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Append HMRC Negative_or_Zero firms, allocated by sector share."""
    target = int(hmrc_bands["Negative_or_Zero"])
    if target <= 0:
        return sic_codes, turnover, input_values, weights

    logger.info("Adding %s zero/negative-turnover firms (HMRC)...", f"{target:,}")
    unique_sics, counts = torch.unique(sic_codes, return_counts=True)
    total = len(sic_codes)
    add_sics: list[int] = []
    for sic, count in zip(unique_sics, counts):
        n = int((target * count.float() / total).item())
        add_sics.extend([int(sic.item())] * n)

    if not add_sics:
        return sic_codes, turnover, input_values, weights

    n_add = len(add_sics)
    extra_sic = torch.tensor(add_sics, dtype=torch.int64, device=device)
    extra_zeros = torch.zeros(n_add, dtype=torch.float32, device=device)
    extra_weights = torch.ones(n_add, dtype=torch.float32, device=device)

    sic_codes = torch.cat([sic_codes, extra_sic])
    turnover = torch.cat([turnover, extra_zeros])
    input_values = torch.cat([input_values, extra_zeros])
    weights = torch.cat([weights, extra_weights])
    logger.info("Added %s zero-turnover firms", f"{n_add:,}")
    return sic_codes, turnover, input_values, weights


def generate(
    config: Optional[Config] = None,
    *,
    vintage: Optional[str] = None,
    threshold: Optional[float] = None,
    seed: Optional[int] = None,
    output: Optional[str] = None,
    write: bool = True,
    return_report: bool = False,
):
    """Generate the synthetic firm population.

    Orchestrates the full pipeline: load -> draw base firms -> draw inputs and
    employment -> build targets -> calibrate weights -> add zero-turnover firms
    -> assign VAT flags -> validate -> (optionally) write the CSV.

    Args:
        config: Base configuration. Defaults to :data:`config.DEFAULT_CONFIG`.
        threshold: Override the VAT threshold (£k) for this run.
        seed: Override the random seed for this run.
        output: Override the output CSV path (string).
        write: If True, write the CSV to disk.
        return_report: If True, return ``(df, ValidationReport)``.

    Returns:
        The synthetic ``pandas.DataFrame``, or ``(df, report)`` if
        ``return_report`` is True.
    """
    from dataclasses import replace

    from .config import VINTAGES

    cfg = config or DEFAULT_CONFIG
    overrides: dict = {}
    if vintage is not None:
        if vintage not in VINTAGES:
            raise ValueError(
                f"Unknown vintage {vintage!r}; choose from {sorted(VINTAGES)}"
            )
        overrides["data_vintage"] = vintage
        # Vintage pins its own threshold unless --threshold is given explicitly.
        if threshold is None:
            overrides["vat_threshold"] = VINTAGES[vintage]["threshold"]
    if threshold is not None:
        overrides["vat_threshold"] = float(threshold)
    if seed is not None:
        overrides["seed"] = int(seed)
    if overrides:
        # replace() re-runs Config.__post_init__, which re-derives
        # processed_dir from the (possibly new) data_vintage.
        cfg = replace(cfg, **overrides)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    logger.info(
        "Generating synthetic firms (threshold=£%.0fk, seed=%d, device=%s)",
        cfg.vat_threshold,
        cfg.seed,
        cfg.device,
    )

    data: LoadedData = load_data(cfg)

    base_sic, base_turnover = generate_base_firms(data.ons_turnover, cfg.device)
    base_input = generate_input_values(base_turnover, base_sic, cfg.device)

    # Temporary per-firm employment-band assignment for the target matrix.
    tmp_emp = assign_employment(len(base_sic), data.ons_employment, cfg.device)
    emp_band_idx = torch.tensor(
        [_employment_band_index(e.item()) for e in tmp_emp],
        dtype=torch.long,
        device=cfg.device,
    )

    target_matrix, target_values, spec = build_target_matrix(
        cfg,
        base_turnover,
        base_sic,
        base_input,
        emp_band_idx,
        data.hmrc_bands,
        data.hmrc_population_sector,
        data.ons_employment,
        data.hmrc_liability_sector,
        data.vat_liability_bands,
    )

    weights = optimize_weights(cfg, target_matrix, target_values, spec)

    final_sic, final_turnover, final_input, final_weights = _add_zero_turnover_firms(
        base_sic, base_turnover, base_input, weights, data.hmrc_bands, cfg.device
    )

    employment = assign_employment(len(final_sic), data.ons_employment, cfg.device)
    vat_flags = assign_vat_flags(final_turnover, data.hmrc_bands, cfg)

    logger.info("Assembling final DataFrame...")
    sic_np = final_sic.cpu().numpy().astype(int)
    turnover_np = final_turnover.cpu().numpy()
    input_np = final_input.cpu().numpy()

    synthetic_df = pd.DataFrame(
        {
            "sic_code": [str(s).zfill(5) for s in sic_np],
            "annual_turnover_k": turnover_np,
            "annual_input_k": input_np,
            "vat_liability_k": turnover_np - input_np,
            "employment": employment.cpu().numpy().astype(int),
            "weight": final_weights.cpu().numpy(),
            "vat_registered": vat_flags.cpu().numpy().astype(bool),
        }
    )
    logger.info(
        "Generated %s rows, weighted population %s",
        f"{len(synthetic_df):,}",
        f"{synthetic_df['weight'].sum():,.0f}",
    )

    report: ValidationReport = validate(synthetic_df, data, cfg)

    if write:
        out_path = cfg.synthetic_dir / output if output else cfg.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_df.to_csv(out_path, index=False)
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info("Wrote %s rows to %s (%.1f MB)", f"{len(synthetic_df):,}", out_path, size_mb)

    if return_report:
        return synthetic_df, report
    return synthetic_df
