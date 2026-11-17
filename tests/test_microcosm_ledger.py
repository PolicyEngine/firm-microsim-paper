from __future__ import annotations

import json
from pathlib import Path

import pytest  # compat: populace alias, remove after 1.1.0

from firm_microsim.microcosm_ledger import (
    ParitySummary,
    ParityTable,
    REFERENCE_MICROCOSM_LEDGER_2024_25,
    REFERENCE_PARITY,
    TargetSummary,  # compat: populace alias, remove after 1.1.0
    comparison_rows,
    format_comparison_report,
    provenance_payload,
)


def test_comparison_rows_include_truth_for_official_targets() -> None:
    rows = {row.label: row for row in comparison_rows()}

    assert rows["Weighted population"].truth == "2,734,615 ONS firms"
    assert rows["Weighted population"].comparability == "Direct"
    assert rows["VAT liability by band"].truth.endswith(
        "net VAT liability by turnover band"
    )
    assert rows["VAT liability by band"].comparability == "Direct"
    assert rows["Rows"].truth == "N/A, synthetic support size"
    assert rows["HMRC turnover bands"].comparability == "Not like-for-like"
    assert rows["Overall"].truth == "N/A, mean of calibrated accuracy scores"
    assert rows["Overall"].comparability == "Not like-for-like"


def test_reference_report_records_microcosm_calibration_tradeoff() -> None:
    report = format_comparison_report()

    assert "Paper 2024-25" in report
    assert "Populace/Ledger 2024-25" in report
    assert "pinned to merged Ledger and Populace snapshots" in report
    assert "Preliminary reference run" not in report
    assert (
        "| Overall | Not like-for-like | N/A, mean of calibrated accuracy scores | "
        "90.5% | 93.8% |"
    ) in report
    assert "not a like-for-like quality ranking" in report
    assert "0 mismatched" in report
    assert REFERENCE_PARITY.facts_sha256 in report
    assert "--reference-population" in report
    assert (
        "| VAT liability by sector diagnostic | Diagnostic | GBP 177.29bn net "
        "VAT liability by SIC sector | 44.5% | 42.2% |"
    ) in report


def test_checked_artifacts_match_reference_rendering() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (
        root / "results" / "populace_ledger_comparison.txt"
    ).read_text() == format_comparison_report()
    assert json.loads(
        (root / "results" / "populace_ledger_provenance.json").read_text()
    ) == provenance_payload()


def test_mismatched_parity_report_does_not_claim_exact_match() -> None:
    parity = ParitySummary(
        facts_count=1,
        facts_sha256="abc",
        tables=(
            ParityTable(
                name="hmrc_population_by_sic",
                rows=1,
                paper_rows_after_filter=1,
                same_keys=True,
                values_equal=False,
                max_abs_numeric_diff=1.0,
            ),
        ),
    )

    report = format_comparison_report(parity=parity)

    assert "1 mismatched" in report
    assert "do not exactly match" in report
    assert "The target surface is identical" not in report


def test_reference_provenance_records_pinned_pr_snapshot() -> None:
    payload = provenance_payload()

    assert payload["ledger_paper_parity"]["facts_count"] == 1_439
    assert payload["ledger_paper_parity"]["mismatched_tables"] == []
    assert payload["ledger_paper_parity"]["max_abs_numeric_diff"] == 0.0
    assert (
        payload["migration_snapshot"]["arch_data_commit"]
        == "cd98b5cb7b1604fbf7750689a429bbc356e5603a"
    )
    assert (
        payload["migration_snapshot"]["arch_data_merge_commit"]
        == "ac643afa0c1d45fc4abd0268dc5aa7c843440b38"
    )
    assert payload["migration_snapshot"]["arch_data_state_at_check"].startswith(
        "MERGED"
    )
    assert (
        payload["migration_snapshot"]["populace_commit"]
        == "fa20daf75ff023e5e88731a140f456f58e0b864e"
    )
    assert (
        payload["migration_snapshot"]["populace_merge_commit"]
        == "8271d767244161631253ad1d9ad792a82e2b96b4"
    )
    assert payload["migration_snapshot"]["populace_state_at_check"].startswith(
        "MERGED"
    )
    assert "--reference-population" in payload["migration_snapshot"][
        "comparison_command"
    ]


def test_reference_snapshot_uses_full_microcosm_run() -> None:
    assert REFERENCE_MICROCOSM_LEDGER_2024_25.rows == 2_946_015
    assert round(REFERENCE_MICROCOSM_LEDGER_2024_25.weighted_population) == 2_945_777
    assert round(REFERENCE_MICROCOSM_LEDGER_2024_25.hmrc_bands, 1) == 99.9


def test_populace_ledger_alias_still_works() -> None:  # compat: populace alias, remove after 1.1.0
    import importlib
    import sys
    import warnings

    sys.modules.pop("firm_microsim.populace_ledger", None)  # compat: populace alias, remove after 1.1.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = importlib.import_module("firm_microsim.populace_ledger")  # compat: populace alias, remove after 1.1.0
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert alias.REFERENCE_POPULACE_LEDGER_2024_25 is REFERENCE_MICROCOSM_LEDGER_2024_25  # compat: populace alias, remove after 1.1.0
    assert alias.format_comparison_report() == format_comparison_report()
    assert callable(alias.load_populace_source_data)  # compat: populace alias, remove after 1.1.0
    assert callable(alias.run_populace_from_ledger_facts)  # compat: populace alias, remove after 1.1.0
    row = comparison_rows()[0]
    assert row.populace_value == row.microcosm_value  # compat: populace alias, remove after 1.1.0
    assert alias.comparison_rows is comparison_rows  # compat: populace alias, remove after 1.1.0
    assert alias.TargetSummary is TargetSummary  # compat: populace alias, remove after 1.1.0


def test_old_keywords_still_work() -> None:  # compat: populace alias, remove after 1.1.0
    reference = REFERENCE_MICROCOSM_LEDGER_2024_25  # compat: populace alias, remove after 1.1.0
    with pytest.warns(DeprecationWarning):  # compat: populace alias, remove after 1.1.0
        rows = comparison_rows(populace=reference)  # compat: populace alias, remove after 1.1.0
    assert rows == comparison_rows(microcosm=reference)  # compat: populace alias, remove after 1.1.0
    with pytest.warns(DeprecationWarning):  # compat: populace alias, remove after 1.1.0
        report = format_comparison_report(populace=reference)  # compat: populace alias, remove after 1.1.0
    assert report == format_comparison_report()  # compat: populace alias, remove after 1.1.0
    with pytest.warns(DeprecationWarning):  # compat: populace alias, remove after 1.1.0
        payload = provenance_payload(populace=reference)  # compat: populace alias, remove after 1.1.0
    assert payload == provenance_payload()  # compat: populace alias, remove after 1.1.0
    with pytest.warns(DeprecationWarning):  # compat: populace alias, remove after 1.1.0
        row = TargetSummary("Rows", "Descriptive", "N/A", "1", populace_value="2")  # compat: populace alias, remove after 1.1.0
    assert row == TargetSummary("Rows", "Descriptive", "N/A", "1", "2")  # compat: populace alias, remove after 1.1.0


@pytest.mark.parametrize(  # compat: populace alias, remove after 1.1.0
    "call",  # compat: populace alias, remove after 1.1.0
    [  # compat: populace alias, remove after 1.1.0
        lambda ref: comparison_rows(ref, populace=ref),  # compat: populace alias, remove after 1.1.0
        lambda ref: comparison_rows(microcosm=ref, populace=ref),  # compat: populace alias, remove after 1.1.0
        lambda ref: format_comparison_report(ref, populace=ref),  # compat: populace alias, remove after 1.1.0
        lambda ref: format_comparison_report(microcosm=ref, populace=ref),  # compat: populace alias, remove after 1.1.0
        lambda ref: provenance_payload(ref, populace=ref),  # compat: populace alias, remove after 1.1.0
        lambda ref: provenance_payload(microcosm=ref, populace=ref),  # compat: populace alias, remove after 1.1.0
        lambda ref: TargetSummary("a", "b", "c", "d", "e", populace_value="f"),  # compat: populace alias, remove after 1.1.0
        lambda ref: TargetSummary(  # compat: populace alias, remove after 1.1.0
            "a", "b", "c", "d", microcosm_value="e", populace_value="f"  # compat: populace alias, remove after 1.1.0
        ),  # compat: populace alias, remove after 1.1.0
    ],  # compat: populace alias, remove after 1.1.0
)  # compat: populace alias, remove after 1.1.0
def test_old_and_new_keyword_together_raise(call) -> None:  # compat: populace alias, remove after 1.1.0
    with pytest.raises(TypeError, match="both"):  # compat: populace alias, remove after 1.1.0
        call(REFERENCE_MICROCOSM_LEDGER_2024_25)  # compat: populace alias, remove after 1.1.0
