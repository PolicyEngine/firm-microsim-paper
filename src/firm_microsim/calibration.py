"""Calibration: target-matrix construction and weight optimization.

This module ports the multi-objective calibration core of the original
monolithic generator:

    1. Map firms to HMRC turnover bands (edges driven by the single
       configurable VAT threshold).
    2. Build a sparse target matrix ``A`` where ``A[i, j]`` is firm ``j``'s
       contribution to target ``i`` (turnover bands, HMRC sectors, ONS
       employment bands, VAT-liability by sector, VAT-liability by band).
    3. Optimize per-firm log-weights with Adam under a *symmetric relative
       error* loss, so targets of very different scales are balanced. Turnover
       bands carry ~5x importance; VAT-liability-by-band carry 2x.

All band edges that previously hardcoded £90k now derive from
``config.vat_threshold`` so the package is genuinely single-version.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor

from .config import Config, STANDARD_VAT_RATE

logger = logging.getLogger(__name__)

# ONS employment-size bands (used for both targets and validation).
EMPLOYMENT_BANDS: List[str] = ["0-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250+"]

# VAT-liability turnover bands above the Negative_or_Zero band (validation
# reports all of these).
VAT_LIABILITY_BANDS: List[str] = [
    "£1_to_Threshold",
    "£Threshold_to_£150k",
    "£150k_to_£300k",
    "£300k_to_£500k",
    "£500k_to_£1m",
    "£1m_to_£10m",
    "Greater_than_£10m",
]

# Bands actually CALIBRATED for liability. The £1_to_Threshold band is
# excluded: its HMRC total is remitted by below-threshold VOLUNTARY
# registrants, whose input-reclaim-driven net remittances (~£2,150 average)
# the model's standard-rate-on-value-added liability does not represent.
# Imposing that total on the whole below-threshold population forces the
# optimiser to crush weights in the £50k-£85k region (the visible seam at the
# OBR window edge). It is reported as an informational diagnostic instead,
# like VAT liability by sector.
VAT_LIABILITY_BANDS_CALIBRATED: List[str] = VAT_LIABILITY_BANDS[1:]


def map_to_hmrc_bands(turnover_values: Tensor, threshold: float) -> Tensor:
    """Map turnover values (£k) to HMRC band indices 0-7.

    Bands: 0=Negative_or_Zero, 1=£1_to_Threshold, 2=£Threshold_to_£150k,
    3=£150k_to_£300k, 4=£300k_to_£500k, 5=£500k_to_£1m, 6=£1m_to_£10m,
    7=Greater_than_£10m. The £1_to_Threshold / £Threshold_to_£150k boundary
    is the configurable VAT threshold.

    Args:
        turnover_values: Per-firm turnover in £thousands.
        threshold: VAT threshold in £thousands.

    Returns:
        Long tensor of band indices, same length as ``turnover_values``.
    """
    band_indices = torch.full_like(turnover_values, 7, dtype=torch.long)
    band_indices = torch.where(turnover_values <= 0, 0, band_indices)
    band_indices = torch.where(
        (turnover_values > 0) & (turnover_values <= threshold), 1, band_indices
    )
    band_indices = torch.where(
        (turnover_values > threshold) & (turnover_values <= 150), 2, band_indices
    )
    band_indices = torch.where(
        (turnover_values > 150) & (turnover_values <= 300), 3, band_indices
    )
    band_indices = torch.where(
        (turnover_values > 300) & (turnover_values <= 500), 4, band_indices
    )
    band_indices = torch.where(
        (turnover_values > 500) & (turnover_values <= 1000), 5, band_indices
    )
    band_indices = torch.where(
        (turnover_values > 1000) & (turnover_values <= 10000), 6, band_indices
    )
    return band_indices


def _employment_band_index(emp_val: float) -> int:
    """Map an employment count to its ONS band index (0-6)."""
    if emp_val <= 4:
        return 0
    if emp_val <= 9:
        return 1
    if emp_val <= 19:
        return 2
    if emp_val <= 49:
        return 3
    if emp_val <= 99:
        return 4
    if emp_val <= 249:
        return 5
    return 6


def _band_membership_mask(
    turnover_values: Tensor, band_name: str, threshold: float
) -> Tensor:
    """Boolean mask of firms whose turnover falls in ``band_name``."""
    if band_name == "£1_to_Threshold":
        return (turnover_values > 0) & (turnover_values <= threshold)
    if band_name == "£Threshold_to_£150k":
        return (turnover_values > threshold) & (turnover_values <= 150)
    if band_name == "£150k_to_£300k":
        return (turnover_values > 150) & (turnover_values <= 300)
    if band_name == "£300k_to_£500k":
        return (turnover_values > 300) & (turnover_values <= 500)
    if band_name == "£500k_to_£1m":
        return (turnover_values > 500) & (turnover_values <= 1000)
    if band_name == "£1m_to_£10m":
        return (turnover_values > 1000) & (turnover_values <= 10000)
    return turnover_values > 10000  # Greater_than_£10m


class TargetSpec:
    """Bookkeeping for the calibration target layout.

    Holds the section sizes so importance weights can be applied to the
    correct rows of the loss without the brittle estimation used in the
    original script.
    """

    def __init__(self, n_sectors: int, n_vat_sectors: int, n_near: int = 0) -> None:
        self.n_turnover = 7
        self.n_sectors = n_sectors
        self.n_employment = len(EMPLOYMENT_BANDS)
        self.n_vat_sectors = n_vat_sectors
        self.n_vat_bands = len(VAT_LIABILITY_BANDS_CALIBRATED)
        self.n_near = n_near

        self.turnover_start = 0
        self.sector_start = self.n_turnover
        self.employment_start = self.sector_start + self.n_sectors
        self.vat_sector_start = self.employment_start + self.n_employment
        self.vat_band_start = self.vat_sector_start + self.n_vat_sectors
        self.near_start = self.vat_band_start + self.n_vat_bands
        self.n_targets = self.near_start + self.n_near


def build_target_matrix(
    config: Config,
    turnover_values: Tensor,
    sic_codes: Tensor,
    input_values: Tensor,
    employment_band_indices: Tensor,
    hmrc_bands: dict,
    hmrc_sector_df: pd.DataFrame,
    ons_employment_df: pd.DataFrame,
    vat_liability_sector_df: pd.DataFrame,
    vat_liability_bands: dict,
    near_threshold_bins: pd.DataFrame | None = None,
    base_weights: Tensor | None = None,
) -> Tuple[Tensor, Tensor, TargetSpec]:
    """Construct the calibration target matrix and target vector.

    Args:
        config: Run configuration (threshold + device).
        turnover_values: Per-firm turnover (£k).
        sic_codes: Per-firm integer SIC sector codes.
        input_values: Per-firm input expenditure (£k).
        employment_band_indices: Per-firm ONS employment band index (0-6).
        hmrc_bands: Latest HMRC VAT firm counts by turnover band.
        hmrc_sector_df: HMRC VAT population by sector.
        ons_employment_df: ONS employment-band table.
        vat_liability_sector_df: HMRC VAT liability by sector (£m).
        vat_liability_bands: Latest VAT liability by turnover band (£m).

    Returns:
        Tuple of (target_matrix [n_targets x n_firms], target_values, spec).
    """
    device = config.device
    threshold = config.vat_threshold
    n_firms = len(turnover_values)
    if base_weights is None:
        base_weights = torch.ones_like(turnover_values)

    sector_rows = hmrc_sector_df[hmrc_sector_df["Trade_Sector"] != "Total"].copy()
    if config.calibrate_vat_liability_sector:
        vat_liability_sector_rows = vat_liability_sector_df[
            vat_liability_sector_df["Trade_Sector"] != "Total"
        ].copy()
    else:
        # Excluded from calibration: empty -> contributes zero target rows.
        # (Still reported as an informational diagnostic by validate.py.)
        vat_liability_sector_rows = vat_liability_sector_df.iloc[0:0].copy()

    n_near = 0 if near_threshold_bins is None else len(near_threshold_bins)
    spec = TargetSpec(len(sector_rows), len(vat_liability_sector_rows), n_near)
    target_matrix = torch.zeros(spec.n_targets, n_firms, device=device)

    band_indices = map_to_hmrc_bands(turnover_values, threshold)

    # Rows 0-6: turnover bands (band index 1..7 -> rows 0..6).
    for row, band_idx in enumerate(range(1, 8)):
        target_matrix[row, band_indices == band_idx] = 1.0

    # Sector targets (VAT-registered firms, but membership is by SIC).
    for offset, (_, sector_row) in enumerate(sector_rows.iterrows()):
        sic_code = int(sector_row["Trade_Sector"])
        target_matrix[spec.sector_start + offset, sic_codes == sic_code] = 1.0

    # Employment-band targets.
    for band_idx in range(spec.n_employment):
        row = spec.employment_start + band_idx
        target_matrix[row, employment_band_indices == band_idx] = 1.0

    # Net VAT liability (£k) per firm = standard rate * value added
    # = STANDARD_VAT_RATE * (turnover - input). This matches the HMRC
    # net-VAT-liability targets, which are net of input reclaim.
    vat_liability_values = STANDARD_VAT_RATE * (turnover_values - input_values)

    # VAT-liability-by-sector targets (weight firms by their liability).
    for offset, (_, vat_row) in enumerate(vat_liability_sector_rows.iterrows()):
        row = spec.vat_sector_start + offset
        sic_code = int(vat_row["Trade_Sector"])
        mask = sic_codes == sic_code
        target_matrix[row, mask] = vat_liability_values[mask]

    # VAT-liability-by-band targets.
    for offset, band_name in enumerate(VAT_LIABILITY_BANDS_CALIBRATED):
        row = spec.vat_band_start + offset
        mask = _band_membership_mask(turnover_values, band_name, threshold)
        target_matrix[row, mask] = vat_liability_values[mask]

    # Near-threshold £1k-bin membership rows (bins are (lo, lo+1], matching
    # the coarse-band edge conventions). Target values are assembled below.
    near_targets: list[float] = []
    if n_near:
        # SHAPE-ONLY targets on BOTH sides of the threshold: each side takes
        # the OBR profile's within-side shape, scaled to the synthetic frame's
        # own base-weighted mass on that side. The OBR chart counts HMRC
        # traders (a different unit and, below the threshold, a different
        # universe than the ONS business frame), so its LEVELS are not
        # imported on either side; mixing direct counts on one side with
        # frame-scaled shape on the other inverted the cross-threshold
        # ordering (more mass just above than just below - economically
        # backwards for a liability notch). With side-consistent scaling the
        # cross-threshold ratio is the frame's own, and the OBR data supply
        # only the within-side profile (the rise into the threshold and the
        # decline beyond it).
        below = near_threshold_bins[near_threshold_bins["bin_lo_k"] < threshold]
        above = near_threshold_bins[near_threshold_bins["bin_lo_k"] >= threshold]
        below_lo = float(below["bin_lo_k"].min())
        above_hi = float(above["bin_lo_k"].max()) + 1.0
        below_mask = (turnover_values > below_lo) & (turnover_values <= threshold)
        above_mask = (turnover_values > threshold) & (turnover_values <= above_hi)
        below_rows = float(base_weights[below_mask].sum().item())
        above_rows = float(base_weights[above_mask].sum().item())
        below_total = float(below["count"].sum())
        above_total = float(above["count"].sum())
        for offset, (_, bin_row) in enumerate(near_threshold_bins.iterrows()):
            lo = float(bin_row["bin_lo_k"])
            row = spec.near_start + offset
            mask = (turnover_values > lo) & (turnover_values <= lo + 1.0)
            target_matrix[row, mask] = 1.0
            if lo < threshold:
                near_targets.append(
                    float(bin_row["count"]) / below_total * below_rows
                )
            else:
                near_targets.append(
                    float(bin_row["count"]) / above_total * above_rows
                )

    # ---- Target values --------------------------------------------------
    # £1_to_Threshold keeps the ONS structure (the base-weighted synthetic
    # count, which equals the row count on unsampled builds); all higher
    # bands match HMRC.
    ons_threshold_count = float(base_weights[band_indices == 1].sum().item())
    turnover_targets = [
        ons_threshold_count,
        hmrc_bands["£Threshold_to_£150k"],
        hmrc_bands["£150k_to_£300k"],
        hmrc_bands["£300k_to_£500k"],
        hmrc_bands["£500k_to_£1m"],
        hmrc_bands["£1m_to_£10m"],
        hmrc_bands["Greater_than_£10m"],
    ]

    # Value column is the (single) year column, always the last column —
    # year-agnostic so the 2023-24 / 2024-25 vintages both work.
    sector_targets = [float(r.iloc[-1]) for _, r in sector_rows.iterrows()]

    ons_emp_rows = ons_employment_df[
        ~ons_employment_df["Description"].str.contains("Total", na=False)
    ]
    employment_targets = [
        float(ons_emp_rows[band].fillna(0).sum()) if band in ons_emp_rows.columns else 0.0
        for band in EMPLOYMENT_BANDS
    ]

    # VAT liability targets are in £m in the source; convert to £k.
    # Value column is the (single) year column, always last — year-agnostic.
    vat_liability_sector_targets = [
        float(r.iloc[-1]) * 1000.0 for _, r in vat_liability_sector_rows.iterrows()
    ]
    vat_liability_band_targets = [
        float(vat_liability_bands[band]) * 1000.0 for band in VAT_LIABILITY_BANDS_CALIBRATED
    ]

    target_values_list = (
        turnover_targets
        + sector_targets
        + employment_targets
        + vat_liability_sector_targets
        + vat_liability_band_targets
        + near_targets
    )
    target_values = torch.tensor(target_values_list, dtype=torch.float32, device=device)

    logger.info("Target matrix shape: %s", tuple(target_matrix.shape))
    logger.info(
        "Targets: 7 turnover + %d sector + %d employment + %d VAT-liability sector "
        "+ %d VAT-liability band + %d near-threshold = %d",
        spec.n_sectors,
        spec.n_employment,
        spec.n_vat_sectors,
        spec.n_vat_bands,
        spec.n_near,
        spec.n_targets,
    )
    return target_matrix, target_values, spec


def _importance_weights(spec: TargetSpec, config: Config, device: str) -> Tensor:
    """Build the per-target importance-weight vector for the loss."""
    w = torch.ones(spec.n_targets, device=device)
    w[spec.turnover_start : spec.turnover_start + spec.n_turnover] = config.turnover_importance
    w[spec.sector_start : spec.sector_start + spec.n_sectors] = config.sector_importance
    w[spec.employment_start : spec.employment_start + spec.n_employment] = (
        config.employment_importance
    )
    w[spec.vat_sector_start : spec.vat_sector_start + spec.n_vat_sectors] = (
        config.vat_liability_sector_importance
    )
    w[spec.vat_band_start : spec.vat_band_start + spec.n_vat_bands] = (
        config.vat_liability_band_importance
    )
    w[spec.near_start : spec.near_start + spec.n_near] = (
        config.near_threshold_importance
    )
    return w


def optimize_weights(
    config: Config,
    target_matrix: Tensor,
    target_values: Tensor,
    spec: TargetSpec,
    base_weights: Tensor | None = None,
) -> Tensor:
    """Optimize per-firm weights to match all targets simultaneously.

    Minimizes a mean symmetric-relative-error loss with per-target importance
    weights, Adam, dropout regularization, a mean absolute log-weight penalty,
    gradient clipping, and early stopping. Weights are parameterized as
    ``exp(log_w)`` to remain strictly positive.

    Args:
        config: Run configuration (optimizer hyperparameters).
        target_matrix: ``A[i, j]`` contribution of firm ``j`` to target ``i``.
        target_values: Target vector to match.
        spec: Target layout used to apply importance weights to loss rows.

    Returns:
        Detached tensor of optimized per-firm weights.
    """
    logger.info("Starting multi-objective weight optimization...")
    device = config.device
    _, n_firms = target_matrix.shape

    # Under stratified sampling, weights start at (and the L1 penalty pulls
    # toward) the per-stratum base weights that carry the thinned mass, so a
    # sampled build is the same optimisation problem around a rescaled prior.
    if base_weights is None:
        base_log = torch.zeros(n_firms, device=device)
    else:
        base_log = torch.log(base_weights.to(device))
    log_weights = base_log.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([log_weights], lr=config.learning_rate)
    importance = _importance_weights(spec, config, device)

    best_loss = float("inf")
    patience_counter = 0
    epsilon = 1e-6

    for iteration in range(config.n_iterations):
        optimizer.zero_grad()
        weights = torch.exp(log_weights)

        # Dropout regularization during training.
        dropout_mask = torch.rand_like(weights) < config.dropout_keep_rate
        weights = weights * dropout_mask

        predictions = torch.matmul(target_matrix, weights)

        pred_adj = predictions + epsilon
        target_adj = target_values + epsilon
        error_1 = ((pred_adj / target_adj) - 1) ** 2
        error_2 = ((target_adj / pred_adj) - 1) ** 2
        sre_loss = torch.minimum(error_1, error_2)

        weighted_loss = sre_loss * importance
        total_loss = torch.mean(weighted_loss)
        # Keep the regularizer on the same normalized scale as the target loss.
        total_loss = total_loss + config.l1_reg_coef * torch.mean(
            torch.abs(log_weights - base_log)
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([log_weights], max_norm=config.grad_clip_norm)
        optimizer.step()

        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if iteration % 100 == 0:
            logger.info("Iteration %d: loss = %.6f", iteration, loss_val)

        if patience_counter > config.early_stopping_patience:
            logger.info("Early stopping at iteration %d", iteration)
            break

    final_weights = torch.exp(log_weights).detach()
    final_predictions = torch.matmul(target_matrix, final_weights)

    logger.info("Optimization complete. Turnover-band fit:")
    band_names = [
        "£1_to_Threshold",
        "£Threshold_to_£150k",
        "£150k_to_£300k",
        "£300k_to_£500k",
        "£500k_to_£1m",
        "£1m_to_£10m",
        "Greater_than_£10m",
    ]
    for i, name in enumerate(band_names):
        pred = final_predictions[i].item()
        target = target_values[i].item()
        if target > 0:
            accuracy = max(0.0, 1.0 - abs(pred - target) / target)
            logger.info("  %-22s %12.0f vs %12.0f (%.1f%%)", name, pred, target, accuracy * 100)

    return final_weights
