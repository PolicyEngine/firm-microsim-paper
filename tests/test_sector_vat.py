"""Tests for the data-grounded sector VAT wedge module.

Fast: the override-equivalence test uses a tiny in-memory DataFrame (it does NOT
load the 150MB synthetic population); the ratio test reads two small CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from firm_microsim.dynamic.model import (
    build_reforms,
    reform_revenue,
)
from firm_microsim.dynamic.sector_vat import (
    RATIO_CAP,
    attach_sector_tau0,
    sector_vat_ratios,
)


def test_sector_vat_ratios_in_plausible_band():
    """Every sector wedge sits in [0, RATIO_CAP] and the fallback is sensible."""
    ratios = sector_vat_ratios("2023-24")
    assert len(ratios) > 50  # ~83 SIC divisions are present
    assert ratios.min() >= 0.0
    assert ratios.max() <= RATIO_CAP + 1e-12
    # Known sectors map to integer SIC divisions.
    assert ratios.index.is_unique
    assert 1 in ratios.index  # crop & animal production
    # Aggregate fallback is a plausible mid-single-digit-to-low-teens %.
    pop_mean = ratios.attrs["population_mean"]
    assert 0.0 < pop_mean <= RATIO_CAP


def test_attach_sector_tau0_uses_fallback_for_missing_sectors():
    ratios = pd.Series({10: 0.05, 20: 0.12})
    ratios.attrs["population_mean"] = 0.07
    df = pd.DataFrame({"sic_code": ["00010", "00020", "00099"]})
    tau0 = attach_sector_tau0(df, ratios)
    assert np.allclose(tau0, [0.05, 0.12, 0.07])


def test_tau0_override_reproduces_baseline_exactly():
    """The tau0_override path is non-breaking: passing the internally-computed
    ``liab / turnover`` array must reproduce the default-path result bit-for-bit.
    """
    rng = np.random.default_rng(0)
    n = 400
    # Spread firms across (and around) the behavioural band so the forward solve
    # actually engages for both the static and behavioural cases.
    turnover = rng.uniform(60_000.0, 140_000.0, size=n)
    liab = turnover * rng.uniform(0.005, 0.08, size=n)  # ~0.5%-8% net wedge
    df = pd.DataFrame(
        {
            "turnover": turnover,
            "liab": liab,
            "weight": rng.uniform(1.0, 5.0, size=n),
        }
    )
    # Exactly the array reform_revenue computes internally when override is None.
    tau0 = np.where(turnover > 0, liab / turnover, 0.0)

    sched, _label = build_reforms()["rate10"]
    for behavioural in (False, True):
        for e in (0.05, 0.17):
            base = reform_revenue(df, sched, e, behavioural=behavioural)
            over = reform_revenue(
                df, sched, e, behavioural=behavioural, tau0_override=tau0
            )
            for key in ("rev_baseline", "rev_reform", "d_rev", "n_moved"):
                assert base[key] == pytest.approx(over[key], rel=0, abs=0), (
                    f"{key} mismatch (behavioural={behavioural}, e={e})"
                )
