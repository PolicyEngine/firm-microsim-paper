from __future__ import annotations

import pytest

from firm_microsim.validate import ValidationReport, _accuracy


def test__given_close_positive_target__then_accuracy_is_one_minus_relative_error():
    assert _accuracy(synthetic=90.0, target=100.0) == pytest.approx(0.9)


def test__given_large_miss__then_accuracy_clips_at_zero():
    assert _accuracy(synthetic=300.0, target=100.0) == 0.0
    assert _accuracy(synthetic=-50.0, target=100.0) == 0.0


def test__given_zero_target__then_accuracy_handles_zero_without_division():
    assert _accuracy(synthetic=0.0, target=0.0) == 1.0
    assert _accuracy(synthetic=1.0, target=0.0) == 0.0


def test__given_sector_liability_diagnostic__then_overall_excludes_it():
    report = ValidationReport(
        hmrc_bands=0.9,
        ons_population=0.8,
        employment=0.7,
        sector=0.6,
        vat_liability_sector=0.0,
        vat_liability_band=0.5,
        total_population=100.0,
    )

    assert report.overall == pytest.approx(0.7)
