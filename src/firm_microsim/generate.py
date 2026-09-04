"""Synthetic UK firm-population generator (orchestration).

A two-stage, multi-source microsimulation generator for the UK business
population, calibrated to official statistics:

    Stage 1 — Draw a base population from the **ONS Business Structure
    Database**. For each SIC sector and ONS turnover band, draw individual
    firms with smooth within-band turnover draws that remain inside their
    source bands. Draw input expenditure (Beta-distributed input/output ratios
    with sector-specific shifts) and sector-conditional employment.

    Stage 2 — Calibrate per-firm weights via multi-objective optimization on
    TWO declared universes (issue #37): the ONS VAT/PAYE enterprise frame
    (population, employment bands, near-threshold shape) and the HMRC
    VAT-registered subset (trader counts by turnover band and by sector, net
    VAT liability by band), the latter entering through a per-band
    registration propensity. HMRC negative/zero-turnover traders are appended
    before calibration as an out-of-frame stratum. Turnover bands are weighted
    ~5x; VAT-liability-by-band 2x. A symmetric-relative-error loss balances
    targets across scales. VAT liability by sector is reported as an
    informational diagnostic rather than optimized by default because the
    input/output tax structure is not yet a calibrated target.

VAT scope and registration are then assigned by seeded weighted selection per
HMRC band so registered totals match HMRC to within one calibration weight
(see :func:`assign_vat_flags`).

Output: ~2.94M rows with calibration weights, written to
``data/synthetic/synthetic_firms.csv`` with columns
``sic_code, annual_turnover_k, annual_input_k, vat_liability_k, employment,
weight, vat_scope, vat_registered, in_frame``. ``in_frame`` marks ONS-frame
enterprises (False for the appended HMRC negative/zero-turnover traders);
``vat_scope`` marks firms in the VAT net (registered above the threshold, or
registrable below it); ``vat_registered`` marks HMRC-count-matched traders.

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
    optimize_weights,
)
from .config import Config, DEFAULT_CONFIG, STANDARD_VAT_RATE
from .data_loader import LoadedData, load_data
from .validate import ValidationReport, validate

logger = logging.getLogger(__name__)

# ONS turnover-band parameters: (inclusive lower bound, exclusive upper bound,
# midpoint) in £k.  The published integer labels are represented as contiguous
# continuous intervals; e.g. "50-99" means [50, 100), not [50, 99].
ONS_TURNOVER_BANDS: Dict[str, tuple] = {
    "0-49": (0, 50, 25),
    "50-99": (50, 100, 75),
    "100-249": (100, 250, 175),
    "250-499": (250, 500, 375),
    "500-999": (500, 1000, 750),
    "1000-4999": (1000, 5000, 3000),
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
    """Draw smooth turnover values inside a half-open source ONS band.

    A uniform draw is the maximum-entropy allocation given only band
    membership. Unlike a Beta(2,2) draw applied separately to every band, it
    does not force density to zero at each published band boundary.
    """
    if count == 0:
        return torch.empty(0, device=device)
    lower = max(float(min_t), 0.1)
    upper = float(max_t)
    unit = torch.rand(count, device=device)
    return lower + unit * (upper - lower)


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

    The input/output ratio is centred near 0.6 (value added ~40% of turnover,
    in line with the UK non-financial business economy) with sector-specific
    shifts, and clamped to [0.1, 0.95] so value added is always strictly
    positive. Net VAT liability is then the standard rate applied to value
    added (see :func:`generate_synthetic_firms`); the model is a standard-rate
    turnover-tax approximation and does not represent net-repayment positions.

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
    scaled = 0.2 + base_ratios * 0.6  # map [0,1] -> [0.2, 0.8]; Beta(4,2) mean -> 0.6
    sector_noise = torch.randn(n_firms, device=device) * 0.15

    sic_np = sic_codes.cpu().numpy()
    neg_mask = np.isin(sic_np, list(_NEG_LIABILITY_SECTORS))
    high_mask = np.isin(sic_np, list(_HIGH_LIABILITY_SECTORS))
    neg_t = torch.tensor(neg_mask, device=device)
    high_t = torch.tensor(high_mask, device=device)

    scaled = scaled + neg_t.float() * (torch.rand(n_firms, device=device) * 0.3)
    scaled = scaled - high_t.float() * (torch.rand(n_firms, device=device) * 0.2)

    # Clamp the input/output ratio to [0.1, 0.95]: value added is always
    # strictly positive (between 5% and 90% of turnover), so no firm has inputs
    # exceeding turnover. The upper bound 0.95 rules out negative net VAT
    # liability (the model is a standard-rate approximation that abstracts from
    # net-repayment positions); the lower bound 0.1 prevents implausibly high
    # value added that the weight optimiser could exploit with outlier weights.
    final_ratios = torch.clamp(scaled + sector_noise, 0.1, 0.95)
    input_values = torch.where(
        turnover_values > 0, turnover_values * final_ratios, torch.zeros_like(turnover_values)
    )

    logger.info(
        "Input/output ratio: mean=%.2f std=%.2f; mean value-added share=%.2f; "
        "firms with negative value added: %s",
        final_ratios.mean().item(),
        final_ratios.std().item(),
        (1.0 - final_ratios).mean().item(),
        f"{int((final_ratios > 1.0).sum().item()):,}",
    )
    return input_values


def assign_employment(
    sic_codes: Tensor, ons_employment: pd.DataFrame, device: str
) -> Tensor:
    """Assign employment conditional on the firm's ONS sector.

    Args:
        sic_codes: Per-firm SIC sector codes.
        ons_employment: ONS employment-band table.
        device: Torch device.

    Returns:
        Per-firm employment counts (float32).
    """
    logger.info("Assigning employment from sector-specific ONS distributions...")
    num_firms = len(sic_codes)
    sector_rows = ons_employment[
        ~ons_employment["Description"].str.contains("Total", na=False)
    ]
    national = torch.tensor(
        [float(sector_rows[b].fillna(0).sum()) for b in EMPLOYMENT_BANDS],
        dtype=torch.float32,
        device=device,
    )
    national = national / national.sum().clamp_min(1.0)
    row_by_sic = {
        int(row["SIC Code"]): row
        for _, row in sector_rows.iterrows()
        if pd.notna(row.get("SIC Code"))
    }

    result = torch.empty(num_firms, dtype=torch.float32, device=device)
    for sic in torch.unique(sic_codes).tolist():
        idx = torch.where(sic_codes == int(sic))[0]
        row = row_by_sic.get(int(sic))
        if row is None:
            probs = national
        else:
            counts = torch.tensor(
                [
                    float(row.get(b, 0)) if pd.notna(row.get(b, 0)) else 0.0
                    for b in EMPLOYMENT_BANDS
                ],
                dtype=torch.float32,
                device=device,
            )
            probs = counts / counts.sum().clamp_min(1.0)
            if float(counts.sum().item()) == 0:
                probs = national

        band_ids = torch.multinomial(probs, len(idx), replacement=True)
        for band_id, band in enumerate(EMPLOYMENT_BANDS):
            out_idx = idx[band_ids == band_id]
            if len(out_idx) == 0:
                continue
            min_v, max_v, midpoint = ONS_EMPLOYMENT_BANDS[band]
            if band == "0-4":
                values = torch.randint(1, 5, (len(out_idx),), device=device).float()
            elif band == "250+":
                log_mean = torch.log(torch.tensor(float(midpoint), device=device))
                values = torch.normal(log_mean, 0.8, (len(out_idx),), device=device).exp()
                values = torch.clamp(values, min_v, max_v).round()
            else:
                values = torch.randint(
                    int(min_v), int(max_v) + 1, (len(out_idx),), device=device
                ).float()
            result[out_idx] = values
    return result


def stratified_thin(
    turnover_values: Tensor, sic_codes: Tensor, config: Config
) -> tuple[Tensor, Tensor]:
    """Stratified thinning for fast iteration builds.

    Keeps rows inside the analysis window ``[sample_window_lo_k,
    sample_window_hi_k]`` at ``sample_window_fraction`` and rows outside it at
    ``sample_tail_fraction``, with at least ``sample_cell_floor`` retained rows
    per stratum. Strata are sector x HMRC band x window-side. Each retained
    row carries a base weight equal to its stratum's rows-drawn / rows-kept
    ratio, so base-weighted totals reproduce the full draw's totals exactly
    per stratum, and every calibration target remains a true total.

    Returns ``(keep_indices, base_weights_for_kept_rows)``.
    """
    import numpy as np
    from .calibration import map_to_hmrc_bands

    t = turnover_values.cpu().numpy()
    sic = sic_codes.cpu().numpy()
    band = map_to_hmrc_bands(turnover_values, config.vat_threshold).cpu().numpy()
    in_window = (
        (t >= config.sample_window_lo_k) & (t <= config.sample_window_hi_k)
    )
    frac = np.where(in_window, config.sample_window_fraction, config.sample_tail_fraction)

    rng = np.random.default_rng(config.seed + 1)
    u = rng.random(len(t))
    keep = u < frac

    # Per-stratum floor + exact ratio base weights.
    df = pd.DataFrame({"sic": sic, "band": band, "win": in_window, "keep": keep})
    df["order"] = u  # deterministic per-stratum top-up ordering
    grp = df.groupby(["sic", "band", "win"], sort=False)
    floor = config.sample_cell_floor
    kept_counts = grp["keep"].transform("sum")
    sizes = grp["keep"].transform("size")
    need_topup = (kept_counts < np.minimum(floor, sizes)) & (~df["keep"])
    if need_topup.any():
        # Keep the lowest-u unkept rows per deficient stratum up to the floor.
        deficit = (np.minimum(floor, sizes) - kept_counts).clip(lower=0)
        sel = df.index[need_topup]
        rank = (
            df.loc[sel]
            .groupby(["sic", "band", "win"], sort=False)["order"]
            .rank(method="first")
        )
        topup_idx = sel[rank.to_numpy() <= deficit.loc[sel].to_numpy()]
        df.loc[topup_idx, "keep"] = True

    kept_counts = df.groupby(["sic", "band", "win"], sort=False)["keep"].transform("sum")
    base = (sizes / kept_counts.replace(0, 1)).to_numpy()

    keep_np = df["keep"].to_numpy()
    keep_idx = torch.tensor(np.where(keep_np)[0], device=config.device)
    base_kept = torch.tensor(
        base[keep_np], dtype=turnover_values.dtype, device=config.device
    )
    logger.info(
        "Stratified thinning: kept %s of %s rows (%.1f%%); base weights "
        "1.0-%.1f; window [%.0fk, %.0fk] at %.0f%%, tails at %.0f%%",
        f"{int(keep_np.sum()):,}",
        f"{len(keep_np):,}",
        100.0 * keep_np.mean(),
        float(base_kept.max().item()),
        config.sample_window_lo_k,
        config.sample_window_hi_k,
        100.0 * config.sample_window_fraction,
        100.0 * config.sample_tail_fraction,
    )
    return keep_idx, base_kept


def _select_to_weighted_target(
    candidate_idx: Tensor, weights: Tensor, target: float, device: str
) -> Tensor:
    """Seeded random order of ``candidate_idx`` until cumulative weight >= target."""
    if len(candidate_idx) == 0 or target <= 0:
        return candidate_idx[:0]
    order = candidate_idx[torch.randperm(len(candidate_idx), device=device)]
    cumulative = torch.cumsum(weights[order], dim=0)
    cutoff = int(torch.searchsorted(cumulative, torch.tensor(float(target))).item())
    cutoff = min(cutoff, len(order) - 1)
    return order[: cutoff + 1]


def assign_vat_flags(
    turnover_values: Tensor,
    hmrc_bands: Dict[str, float],
    config: Config,
    calibration_weights: Tensor | None = None,
    frame_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Assign VAT scope and registration flags to match HMRC trader counts.

    Two universes (issue #37). Every ONS-frame row is an enterprise; only a
    subset are VAT traders. ``vat_scope`` marks firms in the VAT net:

    * above the threshold, a seeded weighted selection per HMRC turnover band
      whose cumulative calibration weight reaches that band's HMRC count (the
      remainder are PAYE-only or exempt-sector enterprises, out of scope);
    * below the threshold, every frame enterprise is treated as registrable
      (scope = True) — a maintained assumption, since HMRC publishes only the
      registered count there — and a seeded weighted selection reaching the
      HMRC ``£1_to_Threshold`` count is flagged registered (voluntary, or
      registered under the rolling test);
    * appended negative/zero-turnover rows (``frame_mask`` False) are HMRC
      traders by construction: in scope and registered.

    ``vat_registered`` = (scope and turnover > threshold) or selected below or
    appended. Weighted totals match HMRC to within one calibration weight per
    band.

    Returns ``(vat_scope, vat_registered)`` as boolean tensors.
    """
    logger.info("Assigning VAT scope and registration flags...")
    threshold = config.vat_threshold
    device = config.device
    n = len(turnover_values)
    weights = (
        calibration_weights
        if calibration_weights is not None
        else torch.ones_like(turnover_values)
    )
    if frame_mask is None:
        frame_mask = torch.ones(n, dtype=torch.bool, device=device)

    from .calibration import map_to_hmrc_bands

    band_indices = map_to_hmrc_bands(turnover_values, threshold)
    band_names = [
        "Negative_or_Zero", "£1_to_Threshold", "£Threshold_to_£150k",
        "£150k_to_£300k", "£300k_to_£500k", "£500k_to_£1m", "£1m_to_£10m",
        "Greater_than_£10m",
    ]

    scope = torch.zeros(n, dtype=torch.bool, device=device)
    registered = torch.zeros(n, dtype=torch.bool, device=device)

    # Appended HMRC traders: in scope, registered.
    scope[~frame_mask] = True
    registered[~frame_mask] = True

    # Below the threshold: all frame rows registrable; HMRC count registered.
    below = frame_mask & (band_indices == 1)
    scope[below] = True
    target_below = float(hmrc_bands.get("£1_to_Threshold", 0.0))
    chosen = _select_to_weighted_target(torch.where(below)[0], weights, target_below, device)
    registered[chosen] = True
    n_below = float(weights[below].sum().item())
    logger.info(
        "Below-threshold registered share: %.3f (HMRC %s / weighted frame %s)",
        target_below / n_below if n_below > 0 else float("nan"),
        f"{target_below:,.0f}", f"{n_below:,.0f}",
    )

    # Above the threshold: in-scope selection per band to the HMRC count.
    for b in range(2, 8):
        cand = frame_mask & (band_indices == b)
        target = float(hmrc_bands.get(band_names[b], 0.0))
        chosen = _select_to_weighted_target(torch.where(cand)[0], weights, target, device)
        scope[chosen] = True
        registered[chosen] = True
        mass = float(weights[cand].sum().item())
        logger.info(
            "  %-22s in-scope %s of %s weighted frame firms (%.3f)",
            band_names[b], f"{target:,.0f}", f"{mass:,.0f}",
            target / mass if mass > 0 else float("nan"),
        )

    logger.info(
        "VAT: %s in scope, %s registered (weighted %s)",
        f"{int(scope.sum().item()):,}",
        f"{int(registered.sum().item()):,}",
        f"{float(weights[registered].sum().item()):,.0f}",
    )
    return scope, registered


def _add_zero_turnover_firms(
    sic_codes: Tensor,
    turnover: Tensor,
    input_values: Tensor,
    weights: Tensor,
    hmrc_bands: Dict[str, float],
    device: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Append HMRC Negative_or_Zero traders, allocated by sector share.

    Appended BEFORE calibration (issue #37): the rows enter the optimiser with
    unit base weights, contribute to the HMRC registered-subset rows with
    propensity one, and are excluded from the ONS-frame rows (population,
    employment, near-threshold shape) through ``frame_mask``.
    """
    target = int(hmrc_bands["Negative_or_Zero"])
    if target <= 0:
        return sic_codes, turnover, input_values, weights

    logger.info("Adding %s zero/negative-turnover firms (HMRC)...", f"{target:,}")
    unique_sics, counts = torch.unique(sic_codes, return_counts=True)
    total = len(sic_codes)
    exact = target * counts.double() / total
    allocations = torch.floor(exact).long()
    remainder = target - int(allocations.sum().item())
    if remainder > 0:
        fractional = exact - allocations.double()
        top = torch.topk(fractional, k=remainder).indices
        allocations[top] += 1
    add_sics: list[int] = []
    for sic, n_alloc in zip(unique_sics, allocations):
        n = int(n_alloc.item())
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
    fast: bool = False,
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
    if fast:
        overrides.update(
            sample_window_fraction=0.30,
            sample_tail_fraction=0.05,
        )
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

    # Fast-iteration mode: stratified thinning with per-stratum base weights.
    if cfg.sample_tail_fraction < 1.0 or cfg.sample_window_fraction < 1.0:
        keep_idx, base_weights = stratified_thin(base_turnover, base_sic, cfg)
        base_sic = base_sic[keep_idx]
        base_turnover = base_turnover[keep_idx]
        base_input = base_input[keep_idx]
    else:
        base_weights = torch.ones_like(base_turnover)

    # Append the HMRC negative/zero-turnover traders BEFORE calibration so
    # every target row sees the same rows (issue #37). They are outside the
    # ONS frame (frame_mask False) and inside the HMRC registered subset.
    n_frame = len(base_turnover)
    final_sic, final_turnover, final_input, final_base_weights = _add_zero_turnover_firms(
        base_sic, base_turnover, base_input, base_weights, data.hmrc_bands, cfg.device
    )
    frame_mask = torch.zeros(len(final_turnover), dtype=torch.bool, device=cfg.device)
    frame_mask[:n_frame] = True

    # Per-firm employment assignment used by the target matrix AND retained in
    # the output. Earlier versions redrew it after optimisation, invalidating
    # the employment target rows.
    employment = assign_employment(final_sic, data.ons_employment, cfg.device)
    emp_band_idx = torch.tensor(
        [_employment_band_index(e.item()) for e in employment],
        dtype=torch.long,
        device=cfg.device,
    )

    target_matrix, target_values, spec = build_target_matrix(
        cfg,
        final_turnover,
        final_sic,
        final_input,
        emp_band_idx,
        data.hmrc_bands,
        data.ons_total,
        data.hmrc_population_sector,
        data.ons_employment,
        data.hmrc_liability_sector,
        data.vat_liability_bands,
        near_threshold_bins=getattr(data, "near_threshold_bins", None),
        base_weights=final_base_weights,
        frame_mask=frame_mask,
    )

    final_weights = optimize_weights(
        cfg, target_matrix, target_values, spec, base_weights=final_base_weights
    )

    vat_scope, vat_flags = assign_vat_flags(
        final_turnover, data.hmrc_bands, cfg,
        calibration_weights=final_weights, frame_mask=frame_mask,
    )

    logger.info("Assembling final DataFrame...")
    sic_np = final_sic.cpu().numpy().astype(int)
    turnover_np = final_turnover.cpu().numpy()
    input_np = final_input.cpu().numpy()

    synthetic_df = pd.DataFrame(
        {
            "sic_code": [str(s).zfill(5) for s in sic_np],
            "annual_turnover_k": turnover_np,
            "annual_input_k": input_np,
            "vat_liability_k": STANDARD_VAT_RATE * (turnover_np - input_np),
            "employment": employment.cpu().numpy().astype(int),
            "weight": final_weights.cpu().numpy(),
            "vat_scope": vat_scope.cpu().numpy().astype(bool),
            "vat_registered": vat_flags.cpu().numpy().astype(bool),
            "in_frame": frame_mask.cpu().numpy().astype(bool),
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
