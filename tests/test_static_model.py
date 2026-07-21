import pandas as pd

from firm_microsim.static.model import StaticVATModel


def test_threshold_sweep_has_correct_revenue_sign() -> None:
    model = StaticVATModel.__new__(StaticVATModel)
    model.firms = pd.DataFrame(
        {
            "annual_turnover_k": [80.0, 87.0, 95.0, 110.0],
            "vat_liability_k": [2.0, 3.0, 4.0, 5.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    result = model.threshold_sweep(
        year="2025-26", thresholds=[80_000, 90_000, 100_000], baseline=90_000
    ).set_index("threshold_k")

    assert result.loc[80.0, "revenue_change_m"] > 0
    assert result.loc[90.0, "revenue_change_m"] == 0
    assert result.loc[100.0, "revenue_change_m"] < 0
