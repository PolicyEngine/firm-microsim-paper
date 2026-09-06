"""Static model: one ageing convention, applied to turnover and liability."""

from __future__ import annotations

import pandas as pd
import pytest

from firm_microsim.static.model import FISCAL_YEARS, VINTAGE_BASE_GROWTH, StaticVATModel


def _model_with(df: pd.DataFrame, vintage: str = "2023-24") -> StaticVATModel:
    m = StaticVATModel.__new__(StaticVATModel)
    m.vintage = vintage
    m.data_threshold = 85_000.0
    m.firms = df
    return m


def _firms() -> pd.DataFrame:
    # A firm just below £85k that ageing carries across the threshold, one
    # above, one out of scope above, and one voluntary registrant below.
    return pd.DataFrame(
        {
            "annual_turnover_k": [84.0, 100.0, 100.0, 40.0],
            "vat_liability_k": [8.0, 10.0, 10.0, 4.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "vat_scope": [True, True, False, True],
            "vat_registered": [False, True, False, True],
            "voluntary": [False, False, False, True],
            "mandatory": [False, True, False, False],
        }
    )


def test_ageing_moves_turnover_and_liability_together() -> None:
    m = _model_with(_firms())
    aged = m._aged(1.05)
    assert aged["turnover"].iloc[0] == pytest.approx(84_000.0 * 1.05)
    assert aged["liab"].iloc[0] == pytest.approx(8_000.0 * 1.05)
    # The £84k firm crosses £85k once aged: membership is on aged turnover.
    assert bool(m._registered(aged, 85_000.0).iloc[0])
    assert not bool(m._registered(m._aged(1.0), 85_000.0).iloc[0])


def test_out_of_scope_and_voluntary_conventions() -> None:
    m = _model_with(_firms())
    df = m._aged(1.0)
    reg = m._registered(df, 85_000.0)
    assert list(reg) == [False, True, False, True]
    # Raising the threshold to £120k releases the in-scope firm only; the
    # voluntary registrant keeps remitting; out-of-scope never remits.
    assert m._revenue(df, 120_000.0) == pytest.approx(4_000.0)
    assert m._revenue(df, 85_000.0) == pytest.approx(14_000.0)
    # The reported base excludes below-threshold voluntary remittances.
    assert m._mandatory_base(df, 85_000.0) == pytest.approx(10_000.0)


def test_vintage_growth_is_relative_to_its_own_data_year() -> None:
    m23 = _model_with(_firms(), "2023-24")
    m24 = _model_with(_firms(), "2024-25")
    fy = {f["year"]: f["firm_growth"] for f in FISCAL_YEARS}
    assert m23._growth("2025-26") == pytest.approx(fy["2025-26"])
    assert m24._growth("2025-26") == pytest.approx(fy["2025-26"] / VINTAGE_BASE_GROWTH["2024-25"])
    assert m24._growth("2024-25") == pytest.approx(1.0)


def test_deregistration_gap_retains_top_of_released_band() -> None:
    m = _model_with(_firms())
    df = m._aged(1.0)
    # Raising the threshold to £101k: the £100k in-scope registrant sits
    # inside the £2k deregistration gap and stays registered ...
    assert bool(m._registered(df, 101_000.0).iloc[1])
    # ... but a raise to £103k releases it.
    assert not bool(m._registered(df, 103_000.0).iloc[1])
    # With no gap the naive whole-band release applies.
    assert not bool(m._registered(df, 101_000.0, gap=0.0).iloc[1])


def test_voluntary_registrant_aged_across_threshold_is_released_by_a_rise() -> None:
    """The documented convention: a data-year voluntary registrant that ageing
    carries above T0 is released by a rise (unless gap-protected); with
    retain_voluntary it keeps its registration."""
    m = _model_with(_firms())
    m.firms.loc[3, "annual_turnover_k"] = 84.0  # voluntary at £84k
    df = m._aged(1.05)  # -> £88.2k, above T0 = £85k
    assert bool(m._registered(df, 85_000.0).iloc[3])          # baseline: registered
    assert not bool(m._registered(df, 95_000.0).iloc[3])      # rise to £95k: released
    assert bool(m._registered(df, 95_000.0, retain_voluntary=True).iloc[3])
    assert bool(m._registered(df, 90_000.0).iloc[3])          # £88.2k >= 90k-2k: gap-protected
