"""reform_revenue honours its threshold argument T in the baseline (review N8)."""

from __future__ import annotations

import pandas as pd
import pytest

from firm_microsim.dynamic.model import make_schedule_raise, reform_revenue


def _firms() -> pd.DataFrame:
    return pd.DataFrame(
        {"turnover": [88_000.0, 95_000.0], "liab": [1_000.0, 1_000.0], "weight": [1.0, 1.0]}
    )


def test_static_baseline_counts_only_firms_above_T() -> None:
    raise_to = make_schedule_raise(100_000.0)
    r85 = reform_revenue(_firms(), raise_to, 0.0, behavioural=False, T=85_000.0)
    r90 = reform_revenue(_firms(), raise_to, 0.0, behavioural=False, T=90_000.0)
    assert r85["rev_baseline"] == pytest.approx(2_000.0)
    assert r90["rev_baseline"] == pytest.approx(1_000.0)


def test_behavioural_baseline_uses_T() -> None:
    """At T = 90k the £88k firm is unregistered in the baseline and under the
    raise, so it must not move; the old code anchored it on the £85k notch."""
    r = reform_revenue(
        _firms(), make_schedule_raise(100_000.0), 0.5, behavioural=True, T=90_000.0
    )
    assert r["t_notch"][0] == pytest.approx(88_000.0)
    assert r["t_new"][0] == pytest.approx(88_000.0)
    assert r["t_new"][1] > 95_000.0
    assert r["rev_baseline"] == pytest.approx(1_000.0)
