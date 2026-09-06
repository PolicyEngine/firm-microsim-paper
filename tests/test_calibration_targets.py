"""Regression tests for the near-threshold calibration targets.

The OBR £1k-band data enter as SHAPE-ONLY targets, side-consistently: each
side of the threshold takes the chart's within-side shape, scaled to the
synthetic frame's own base-weighted mass on that side. Chart LEVELS must
never be imported on either side — importing them above the threshold while
frame-scaling below inverted the cross-threshold ordering (the defect this
guards against).
"""

from __future__ import annotations

import pytest
import torch

from firm_microsim.calibration import build_target_matrix
from firm_microsim.config import DEFAULT_CONFIG
from firm_microsim.data_loader import load_data

THRESHOLD = 85.0


@pytest.fixture(scope="module")
def frames():
    """Real loader frames for the £85k vintage (schema fidelity)."""
    cfg = DEFAULT_CONFIG  # 2023-24 defaults: threshold 85, OBR bins on
    data = load_data(cfg)
    if getattr(data, "near_threshold_bins", None) is None:
        pytest.skip("near-threshold bins not configured for this vintage")
    return cfg, data


def _tiny_firms():
    """Six firms straddling the threshold, with non-uniform base weights."""
    turnover = torch.tensor([66.5, 70.5, 84.5, 86.5, 89.5, 120.0])
    sic = torch.tensor([47, 47, 62, 62, 47, 62], dtype=torch.int64)
    inputs = turnover * 0.6
    emp = torch.zeros(len(turnover), dtype=torch.int64)
    base_weights = torch.tensor([2.0, 1.0, 1.0, 1.0, 3.0, 1.0])
    return turnover, sic, inputs, emp, base_weights


def test_near_targets_are_side_consistent_shapes(frames):
    cfg, data = frames
    turnover, sic, inputs, emp, base_weights = _tiny_firms()

    matrix, values, spec = build_target_matrix(
        cfg,
        turnover,
        sic,
        inputs,
        emp,
        data.hmrc_bands,
        data.ons_total,
        data.hmrc_population_sector,
        data.ons_employment,
        data.hmrc_liability_sector,
        data.vat_liability_bands,
        near_threshold_bins=data.near_threshold_bins,
        base_weights=base_weights,
    )

    bins = data.near_threshold_bins.reset_index(drop=True)
    below = bins[bins["bin_lo_k"] < THRESHOLD]
    above = bins[bins["bin_lo_k"] >= THRESHOLD]
    below_lo = float(below["bin_lo_k"].min())
    above_hi = float(above["bin_lo_k"].max()) + 1.0

    below_mask = (turnover > below_lo) & (turnover <= THRESHOLD)
    above_mask = (turnover > THRESHOLD) & (turnover <= above_hi)
    below_rows = float(base_weights[below_mask].sum())
    above_rows = float(base_weights[above_mask].sum())
    below_total = float(below["count"].sum())
    above_total = float(above["count"].sum())

    for offset, row in bins.iterrows():
        target = float(values[spec.near_start + offset])
        count = float(row["count"])
        if row["bin_lo_k"] < THRESHOLD:
            expected = count / below_total * below_rows
        else:
            expected = count / above_total * above_rows
        assert target == pytest.approx(expected, rel=1e-6), (
            f"bin {row['bin_lo_k']}: target {target} != side-consistent "
            f"shape value {expected}"
        )
        # The defect this test exists to catch: a direct chart-level import.
        assert target != pytest.approx(count, rel=1e-6) or expected == count

    # Each side's near-targets sum to that side's own frame mass — shape
    # targets redistribute, never add or remove cross-threshold mass.
    below_sum = sum(
        float(values[spec.near_start + i])
        for i, r in bins.iterrows()
        if r["bin_lo_k"] < THRESHOLD
    )
    above_sum = sum(
        float(values[spec.near_start + i])
        for i, r in bins.iterrows()
        if r["bin_lo_k"] >= THRESHOLD
    )
    assert below_sum == pytest.approx(below_rows, rel=1e-6)
    assert above_sum == pytest.approx(above_rows, rel=1e-6)


def test_near_target_rows_select_their_bins(frames):
    cfg, data = frames
    turnover, sic, inputs, emp, base_weights = _tiny_firms()
    matrix, values, spec = build_target_matrix(
        cfg,
        turnover,
        sic,
        inputs,
        emp,
        data.hmrc_bands,
        data.ons_total,
        data.hmrc_population_sector,
        data.ons_employment,
        data.hmrc_liability_sector,
        data.vat_liability_bands,
        near_threshold_bins=data.near_threshold_bins,
        base_weights=base_weights,
    )
    bins = data.near_threshold_bins.reset_index(drop=True)
    for offset, row in bins.iterrows():
        lo = float(row["bin_lo_k"])
        expected = ((turnover > lo) & (turnover <= lo + 1.0)).float()
        assert torch.equal(matrix[spec.near_start + offset], expected)


def test_population_and_employment_rows_are_frame_universe(frames):
    """Frame rows (population, employment) exclude appended HMRC traders and
    target the full ONS total; the employment targets partition that total
    (issue #37)."""
    cfg, data = frames
    turnover, sic, inputs, emp, base_weights = _tiny_firms()
    # Append two zero-turnover HMRC traders outside the frame.
    turnover = torch.cat([turnover, torch.zeros(2)])
    sic = torch.cat([sic, torch.tensor([47, 62])])
    inputs = torch.cat([inputs, torch.zeros(2)])
    emp = torch.cat([emp, torch.zeros(2, dtype=torch.int64)])
    base_weights = torch.cat([base_weights, torch.ones(2)])
    frame_mask = torch.tensor([True] * 6 + [False] * 2)
    matrix, values, spec = build_target_matrix(
        cfg, turnover, sic, inputs, emp, data.hmrc_bands, data.ons_total,
        data.hmrc_population_sector, data.ons_employment,
        data.hmrc_liability_sector, data.vat_liability_bands,
        near_threshold_bins=data.near_threshold_bins,
        base_weights=base_weights, frame_mask=frame_mask,
    )
    assert torch.equal(matrix[spec.population_start], frame_mask.float())
    assert float(values[spec.population_start]) == pytest.approx(data.ons_total)
    emp_rows = matrix[spec.employment_start: spec.employment_start + spec.n_employment]
    assert torch.equal(emp_rows.sum(0), frame_mask.float())
    emp_targets = values[spec.employment_start: spec.employment_start + spec.n_employment]
    assert float(emp_targets.sum()) == pytest.approx(data.ons_total, rel=2e-5)
    # Appended traders enter the HMRC rows with propensity one, and the
    # Negative_or_Zero band row targets the HMRC count.
    assert torch.equal(matrix[spec.turnover_start][6:], torch.ones(2))
    assert float(values[spec.turnover_start]) == pytest.approx(
        data.hmrc_bands["Negative_or_Zero"]
    )
    assert spec.propensity is not None
    assert float(spec.propensity.max()) <= 1.0 + 1e-6


def test_registration_propensity_below_one_on_real_frames(frames):
    """The ONS frame must contain the HMRC registered population in every band."""
    from firm_microsim.calibration import map_to_hmrc_bands, registration_propensity
    from firm_microsim.generate import generate_base_firms
    cfg, data = frames
    torch.manual_seed(0)
    sic, turnover = generate_base_firms(data.ons_turnover, "cpu")
    bands = map_to_hmrc_bands(turnover, cfg.vat_threshold)
    frame = torch.ones(len(turnover), dtype=torch.bool)
    prop = registration_propensity(bands, frame, torch.ones(len(turnover)), data.hmrc_bands)
    # Documented expectation per band (HMRC 2023-24 / frame mass with the
    # open 5000+ band drawn log-uniform on [£5m, £50m), issue #40):
    # below-threshold ~0.89; above, 0.5-0.94.
    expected = {1: 0.89, 2: 0.67, 3: 0.51, 4: 0.60, 5: 0.77, 6: 0.94, 7: 0.82}
    for b, e_p in expected.items():
        p = float(prop[bands == b].max())
        assert abs(p - e_p) < 0.03, f"band {b}: propensity {p:.3f} != ~{e_p}"
    # Expected registered count reproduces HMRC at unit weights (band 7 is
    # tight because the open 5000+ band is drawn uniformly to £50m).
    expected = float(prop.sum())
    hmrc = sum(v for k, v in data.hmrc_bands.items() if k not in ("Negative_or_Zero", "Unknown"))
    assert expected == pytest.approx(hmrc, rel=1e-3)


def test_inverse_dropout_scaling_is_unbiased() -> None:
    torch.manual_seed(123)
    keep = 0.8
    weights = torch.ones(1_000_000)
    mask = torch.rand_like(weights) < keep
    dropped = weights * mask / keep
    assert float(dropped.mean()) == pytest.approx(1.0, abs=0.003)
