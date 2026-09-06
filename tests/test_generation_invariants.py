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


def test_unregistered_stratum_is_below_threshold_with_matching_mean() -> None:
    import pandas as pd
    from firm_microsim.generate import generate_unregistered_firms
    torch.manual_seed(6)
    bpe = pd.DataFrame({
        "SIC Code": ["47", "62", "69"],
        "unregistered_count": [50_000, 30_000, 20_000],
        "unregistered_turnover_m": [1_500.0, 1_800.0, float("nan")],  # means 30k, 60k, national
    })
    sic, t = generate_unregistered_firms(bpe, 85.0, "cpu")
    assert len(sic) == 100_000
    assert float(t.min()) >= 0.1 and float(t.max()) <= 500.0
    assert abs(float(t[sic == 47].mean()) - 30.0) < 1.0
    assert abs(float(t[sic == 62].mean()) - 60.0) < 1.5
    nat = (1_500.0 + 1_800.0) / 80_000 * 1000  # national mean over reported rows
    assert abs(float(t[sic == 69].mean()) - nat) < 1.5
    # Exponential tail: a minority sits above the threshold (exempt traders).
    share_above = float((t > 85.0).float().mean())
    assert 0.05 < share_above < 0.35


def test_power_law_fill_keeps_band_support_and_is_monotone() -> None:
    from firm_microsim.generate import _band_alphas, _draw_power_law
    torch.manual_seed(7)
    draws = _draw_power_law(200_000, 100.0, 250.0, 1.2, "cpu")
    assert float(draws.min()) >= 100.0 and float(draws.max()) < 250.0
    hist = torch.histc(draws, bins=15, min=100.0, max=250.0)
    assert bool((hist[1:] <= hist[:-1] * 1.02).all())  # non-increasing within band
    alphas = _band_alphas({"0-49": 388665, "50-99": 537540, "100-249": 874665,
                           "250-499": 384570, "500-999": 234660, "1000-4999": 226445})
    assert alphas["100-249"] > 0.5  # density falls steeply through the band
