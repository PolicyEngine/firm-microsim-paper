"""Static model conventions: data-year membership, aged liabilities, scope,
voluntary registrants and the deregistration gap."""

from __future__ import annotations

import pandas as pd
import pytest

from firm_microsim.static.model import (
    FISCAL_YEARS,
    STATUTORY_DEREGISTRATION_GAP,
    VINTAGE_BASE_GROWTH,
    StaticVATModel,
)


def _model_with(df: pd.DataFrame, vintage: str = "2023-24") -> StaticVATModel:
    m = StaticVATModel.__new__(StaticVATModel)
    m.vintage = vintage
    m.data_threshold = 85_000.0
    m.firms = df
    return m


def _firms() -> pd.DataFrame:
    # £84k in-scope unregistered; £100k in-scope registrant; £100k out of
    # scope; £40k voluntary registrant; £87k in-scope registrant.
    return pd.DataFrame(
        {
            "annual_turnover_k": [84.0, 100.0, 100.0, 40.0, 87.0],
            "vat_liability_k": [8.0, 10.0, 10.0, 4.0, 9.0],
            "weight": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "vat_scope": [True, True, False, True, True],
            "vat_registered": [False, True, False, True, True],
            "voluntary": [False, False, False, True, False],
            "mandatory": [False, True, False, False, True],
        }
    )


def test_membership_uses_data_year_turnover_and_liability_is_aged() -> None:
    m = _model_with(_firms())
    aged = m._aged(1.05)
    assert aged["turnover"].iloc[0] == pytest.approx(84_000.0)   # not aged
    assert aged["liab"].iloc[0] == pytest.approx(8_000.0 * 1.05)  # aged
    # The £84k firm never crosses £85k, whatever the growth factor.
    assert not bool(m._registered(aged, 85_000.0).iloc[0])


def test_scope_voluntary_and_release_conventions() -> None:
    m = _model_with(_firms())
    df = m._aged(1.0)
    reg = m._registered(df, 85_000.0)
    assert list(reg) == [False, True, False, True, True]
    # Raising to £90k releases the £87k registrant (whole band, gap 0); the
    # voluntary registrant keeps remitting; out-of-scope never remits.
    assert m._revenue(df, 90_000.0) == pytest.approx(14_000_000.0)
    assert m._revenue(df, 85_000.0) == pytest.approx(23_000_000.0)
    # With the statutory gap the £87k registrant sits below £88k and is
    # released; one at £89k would be gap-protected.
    assert not bool(m._registered(df, 90_000.0, gap=STATUTORY_DEREGISTRATION_GAP).iloc[4])
    df2 = df.copy()
    df2.loc[4, "turnover"] = 89_000.0
    assert bool(m._registered(df2, 90_000.0, gap=STATUTORY_DEREGISTRATION_GAP).iloc[4])
    # Reported base excludes below-threshold voluntary remittances.
    assert m._mandatory_base(df, 85_000.0) == pytest.approx(19_000_000.0)


def test_anchor_uses_data_year_band_and_retention_scales_losses_only() -> None:
    m = _model_with(_firms())
    full = m.anchor_reform()
    ret = m.anchor_reform(retention=0.43)
    # 2024-25: baseline 85k vs policy 90k releases the £87k firm: -9k * 1.031.
    assert full["policyengine_impact_m"].iloc[0] == pytest.approx(-9.0 * 1.0310, abs=0.06)
    assert ret["policyengine_impact_m"].iloc[0] == pytest.approx(-9.0 * 1.0310 * 0.57, abs=0.06)
    # 2028-29: baseline 92k > policy 90k: nobody in [90k, 92k) here -> 0.
    assert full["policyengine_impact_m"].iloc[4] == pytest.approx(0.0)


def test_vintage_growth_is_relative_to_its_own_data_year() -> None:
    m23 = _model_with(_firms(), "2023-24")
    m24 = _model_with(_firms(), "2024-25")
    fy = {f["year"]: f["firm_growth"] for f in FISCAL_YEARS}
    assert m23._growth("2025-26") == pytest.approx(fy["2025-26"])
    assert m24._growth("2025-26") == pytest.approx(fy["2025-26"] / VINTAGE_BASE_GROWTH["2024-25"])
    assert m24._growth("2024-25") == pytest.approx(1.0)
