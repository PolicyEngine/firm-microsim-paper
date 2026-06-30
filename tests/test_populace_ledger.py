from __future__ import annotations

import json
from pathlib import Path

from firm_microsim.populace_ledger import (
    ParitySummary,
    ParityTable,
    REFERENCE_POPULACE_LEDGER_2024_25,
    REFERENCE_PARITY,
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


def test_reference_report_records_populace_calibration_tradeoff() -> None:
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


def test_reference_snapshot_uses_full_populace_run() -> None:
    assert REFERENCE_POPULACE_LEDGER_2024_25.rows == 2_946_015
    assert round(REFERENCE_POPULACE_LEDGER_2024_25.weighted_population) == 2_945_777
    assert REFERENCE_POPULACE_LEDGER_2024_25.hmrc_bands == 99.9
