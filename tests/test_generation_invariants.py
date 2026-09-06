from __future__ import annotations

import pandas as pd
import torch

from firm_microsim.config import Config
from firm_microsim.generate import (
    _add_zero_turnover_firms,
    _draw_band_turnover,
    assign_employment,
    assign_vat_flags,
)


def test_turnover_draws_stay_inside_source_band() -> None:
    torch.manual_seed(1)
    draws = _draw_band_turnover(20_000, 50.0, 100.0, "cpu")
    assert float(draws.min()) >= 50.0
    assert float(draws.max()) < 100.0


def test_adjacent_turnover_bands_leave_no_integer_boundary_gap() -> None:
    torch.manual_seed(2)
    lower_band = _draw_band_turnover(100_000, 50.0, 100.0, "cpu")
    upper_band = _draw_band_turnover(100_000, 100.0, 250.0, "cpu")
    assert bool(((lower_band >= 99.0) & (lower_band < 100.0)).any())
    assert bool(((upper_band >= 100.0) & (upper_band < 101.0)).any())


def test_employment_is_conditioned_on_sector() -> None:
    torch.manual_seed(2)
    source = pd.DataFrame(
        {
            "SIC Code": [1, 2],
            "Description": ["A", "B"],
            "0-4": [100, 0],
            "5-9": [0, 0],
            "10-19": [0, 100],
            "20-49": [0, 0],
            "50-99": [0, 0],
            "100-249": [0, 0],
            "250+": [0, 0],
        }
    )
    sic = torch.tensor([1] * 200 + [2] * 200)
    employment = assign_employment(sic, source, "cpu")
    assert bool((employment[:200] <= 4).all())
    assert bool(((employment[200:] >= 10) & (employment[200:] <= 19)).all())


def test_voluntary_registration_matches_weighted_target_within_one_weight() -> None:
    torch.manual_seed(3)
    turnover = torch.full((100,), 50.0)
    weights = torch.linspace(0.5, 2.0, 100)
    target = 42.0
    scope, flags = assign_vat_flags(
        turnover,
        {"£1_to_Threshold": target},
        Config(),
        calibration_weights=weights,
    )
    achieved = float(weights[flags].sum())
    assert achieved >= target
    assert achieved - target <= float(weights.max())
    assert bool(scope.all())  # every below-threshold frame firm is registrable


def test_above_threshold_scope_matches_hmrc_band_count_within_one_weight() -> None:
    torch.manual_seed(4)
    turnover = torch.full((200,), 120.0)  # £Threshold_to_£150k band at £85k
    weights = torch.linspace(0.5, 2.0, 200)
    target = 100.0
    scope, flags = assign_vat_flags(
        turnover,
        {"£1_to_Threshold": 0.0, "£Threshold_to_£150k": target},
        Config(),
        calibration_weights=weights,
    )
    assert torch.equal(scope, flags)  # above the threshold, in scope == registered
    achieved = float(weights[scope].sum())
    assert achieved >= target
    assert achieved - target <= float(weights.max())
    assert int(scope.sum()) < 200  # out-of-scope enterprises remain


def test_zero_turnover_allocation_hits_hmrc_target_exactly() -> None:
    sic = torch.tensor([1, 1, 1, 2, 2, 3])
    turnover = torch.ones(6)
    inputs = torch.zeros(6)
    weights = torch.ones(6)
    out = _add_zero_turnover_firms(
        sic, turnover, inputs, weights, {"Negative_or_Zero": 11}, "cpu"
    )
    assert len(out[0]) - len(sic) == 11
    assert int((out[1] == 0).sum()) == 11


def test_open_band_log_uniform_draw_stays_in_band_and_is_top_light() -> None:
    torch.manual_seed(5)
    draws = _draw_band_turnover(200_000, 5_000.0, 50_000.0, "cpu", log_uniform=True)
    assert float(draws.min()) >= 5_000.0 and float(draws.max()) < 50_000.0
    share_above_10m = float((draws >= 10_000.0).float().mean())
    assert abs(share_above_10m - 0.699) < 0.01  # ln(5)/ln(10)
