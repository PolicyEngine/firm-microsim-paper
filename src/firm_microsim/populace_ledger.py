"""Compare the paper population with the experimental Populace/Ledger run.

The firm generator is being migrated toward PolicyEngine's shared Populace stack,
with official targets flowing from Ledger. This module keeps that migration
auditable from the paper repository:

* without extra dependencies it prints the reference comparison from the
  verified 2024-25 Populace/Ledger run;
* with ``populace-build`` installed from the Populace source tree, it can
  recompute the comparison and Ledger-vs-paper target parity from Ledger
  consumer facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PROCESSED_DATA_DIR


@dataclass(frozen=True)
class CalibrationSnapshot:
    """One synthetic-population calibration summary."""

    rows: int
    weighted_population: float
    hmrc_bands: float
    ons_population: float
    employment: float
    sector: float
    vat_liability_band: float
    overall: float
    vat_liability_sector: float


@dataclass(frozen=True)
class TargetSummary:
    """Official target description for one reported metric."""

    label: str
    comparability: str
    truth: str
    paper_value: str
    populace_value: str


@dataclass(frozen=True)
class ParityTable:
    """One normalized Ledger-vs-paper source-table parity check."""

    name: str
    rows: int
    paper_rows_after_filter: int
    same_keys: bool
    values_equal: bool
    max_abs_numeric_diff: float | None


@dataclass(frozen=True)
class ParitySummary:
    """Source-table parity and facts-file provenance."""

    facts_count: int
    facts_sha256: str
    tables: tuple[ParityTable, ...]

    @property
    def mismatched_tables(self) -> tuple[str, ...]:
        return tuple(table.name for table in self.tables if not table.values_equal)

    @property
    def max_abs_numeric_diff(self) -> float:
        diffs = [
            table.max_abs_numeric_diff
            for table in self.tables
            if table.max_abs_numeric_diff is not None
        ]
        return max(diffs, default=0.0)


# Manually captured reference snapshots.
#
# PAPER_2024_25 is from the paper generator at seed 42. The 2024-25 synthetic
# population was regenerated on 2026-06-30 with:
#
#   firm-microsim --vintage 2024-25 --output synthetic_firms_2024-25.csv
#
# REFERENCE_POPULACE_LEDGER_2024_25 is from a separate full Populace optimizer
# run using the merged upstream snapshots recorded in REFERENCE_MIGRATION. These
# generated-population numbers are not recomputed in CI; CI verifies rendering,
# provenance, and Ledger-vs-paper source-table parity.
PAPER_2024_25 = CalibrationSnapshot(
    rows=2_945_974,
    weighted_population=2_577_078.0,
    hmrc_bands=92.7,
    ons_population=94.2,
    employment=89.7,
    sector=94.5,
    vat_liability_band=81.4,
    overall=90.5,
    vat_liability_sector=44.5,
)

REFERENCE_POPULACE_LEDGER_2024_25 = CalibrationSnapshot(
    rows=2_946_015,
    weighted_population=2_945_776.75,
    hmrc_bands=99.92894355826047,
    ons_population=92.27819089707326,
    employment=92.27463056446766,
    sector=85.0102525769766,
    vat_liability_band=99.45844774271694,
    overall=93.79009306789898,
    vat_liability_sector=42.17514113123149,
)

REFERENCE_PARITY = ParitySummary(
    facts_count=1_439,
    facts_sha256="58b6c2752adec5baa6a6260fe8cd9e9b85d0a78b5ba76ea28201f4c7986dce50",
    tables=(
        ParityTable("ons_turnover_by_sic_band", 88, 88, True, True, 0.0),
        ParityTable("ons_employment_by_sic_band", 88, 88, True, True, 0.0),
        ParityTable("hmrc_population_by_turnover_band", 1, 1, True, True, 0.0),
        ParityTable("hmrc_population_by_sic", 88, 88, True, True, 0.0),
        ParityTable("hmrc_liability_by_turnover_band", 1, 1, True, True, 0.0),
        ParityTable("hmrc_liability_by_sic", 88, 88, True, True, 0.0),
    ),
)

REFERENCE_SOURCE_TOTALS = {
    "source_facts": "1,439 Ledger consumer facts",
    "ons_population": "2,734,615 ONS firms",
    "hmrc_turnover_bands": "2,171,200 VAT-registered firms excluding Unknown",
    "hmrc_sector": "2,330,230 VAT-registered firms by SIC sector",
    "hmrc_liability_band": "GBP 177.17bn net VAT liability by turnover band",
    "hmrc_liability_sector": "GBP 177.29bn net VAT liability by SIC sector",
}

REFERENCE_MIGRATION = {
    "arch_data_pr": "https://github.com/PolicyEngine/ledger/pull/67",
    "arch_data_commit": "cd98b5cb7b1604fbf7750689a429bbc356e5603a",
    "arch_data_merge_commit": "ac643afa0c1d45fc4abd0268dc5aa7c843440b38",
    "arch_data_state_at_check": "MERGED on 2026-06-30",
    "populace_pr": "https://github.com/PolicyEngine/populace/pull/223",
    "populace_commit": "fa20daf75ff023e5e88731a140f456f58e0b864e",
    "populace_merge_commit": "8271d767244161631253ad1d9ad792a82e2b96b4",
    "populace_state_at_check": "MERGED on 2026-06-30",
    "source_packages": [
        "ons-uk-business-firm-targets-2025",
        "ons-uk-business-firm-sector-targets-2025",
        "hmrc-vat-firm-targets-2024-25",
        "hmrc-vat-firm-sector-targets-2024-25",
    ],
    "comparison_command": (
        "firm-microsim-populace-ledger --facts-jsonl "
        "/tmp/uk_firm_consumer_facts.jsonl --reference-population "
        "--output results/populace_ledger_comparison.txt "
        "--json-output results/populace_ledger_provenance.json"
    ),
}


def comparison_rows(
    populace: CalibrationSnapshot = REFERENCE_POPULACE_LEDGER_2024_25,
) -> list[TargetSummary]:
    """Return the 2024-25 paper-vs-Populace comparison table rows."""

    return [
        TargetSummary(
            "Rows",
            "Descriptive",
            "N/A, synthetic support size",
            _count(PAPER_2024_25.rows),
            _count(populace.rows),
        ),
        TargetSummary(
            "Weighted population",
            "Direct",
            REFERENCE_SOURCE_TOTALS["ons_population"],
            _count(PAPER_2024_25.weighted_population),
            _count(populace.weighted_population),
        ),
        TargetSummary(
            "HMRC turnover bands",
            "Not like-for-like",
            REFERENCE_SOURCE_TOTALS["hmrc_turnover_bands"],
            _pct(PAPER_2024_25.hmrc_bands),
            _pct(populace.hmrc_bands),
        ),
        TargetSummary(
            "ONS population",
            "Direct",
            REFERENCE_SOURCE_TOTALS["ons_population"],
            _pct(PAPER_2024_25.ons_population),
            _pct(populace.ons_population),
        ),
        TargetSummary(
            "Employment bands",
            "Direct",
            "ONS employment-band distribution, sum 2,734,615",
            _pct(PAPER_2024_25.employment),
            _pct(populace.employment),
        ),
        TargetSummary(
            "Sector distribution",
            "Project-specific",
            REFERENCE_SOURCE_TOTALS["hmrc_sector"],
            _pct(PAPER_2024_25.sector),
            _pct(populace.sector),
        ),
        TargetSummary(
            "VAT liability by band",
            "Direct",
            REFERENCE_SOURCE_TOTALS["hmrc_liability_band"],
            _pct(PAPER_2024_25.vat_liability_band),
            _pct(populace.vat_liability_band),
        ),
        TargetSummary(
            "Overall",
            "Not like-for-like",
            "N/A, mean of calibrated accuracy scores",
            _pct(PAPER_2024_25.overall),
            _pct(populace.overall),
        ),
        TargetSummary(
            "VAT liability by sector diagnostic",
            "Diagnostic",
            REFERENCE_SOURCE_TOTALS["hmrc_liability_sector"],
            _pct(PAPER_2024_25.vat_liability_sector),
            _pct(populace.vat_liability_sector),
        ),
    ]


def format_comparison_report(
    populace: CalibrationSnapshot = REFERENCE_POPULACE_LEDGER_2024_25,
    parity: ParitySummary = REFERENCE_PARITY,
    *,
    iterations: int = 1_000,
    seed: int = 42,
) -> str:
    """Render a Markdown comparison report."""

    mismatches = len(parity.mismatched_tables)
    parity_exact = mismatches == 0 and parity.max_abs_numeric_diff == 0.0
    if parity_exact:
        parity_sentence = (
            "The Ledger-backed targets match the paper's processed 2024-25 "
            "numeric input tables exactly after dropping presentation-only "
            "labels, totals, and the HMRC Unknown column that the generator "
            "does not calibrate."
        )
        interpretation_target_sentence = "The target surface is identical, "
    else:
        mismatch_list = ", ".join(parity.mismatched_tables) or "unknown tables"
        parity_sentence = (
            "The Ledger-backed targets do not exactly match the paper's "
            "processed 2024-25 numeric input tables under this parity check; "
            f"mismatched tables: {mismatch_list}."
        )
        interpretation_target_sentence = (
            "The target surface is not identical in this recompute, "
        )
    lines = [
        "# Populace/Ledger firm-generation comparison",
        "",
        "NOTE: the 'Paper 2024-25' column below is the pinned 2026-06-30",
        "PRE-CORRECTION build (see paper appendix on data), which predates both",
        "the net-liability correction and the two-universe calibration; the",
        "current build's scores are in results/calibration_accuracy.txt.",
        "",
        "Reference run:",
        "",
        "- Status: pinned to merged Ledger and Populace snapshots",
        "- Vintage: 2024-25",
        f"- Seed: {seed}",
        f"- Populace iterations: {_count(iterations)}",
        f"- Ledger input surface: {_count(parity.facts_count)} consumer facts",
        f"- Ledger facts SHA256: `{parity.facts_sha256}`",
        f"- Ledger target snapshot: {REFERENCE_MIGRATION['arch_data_pr']} at "
        f"`{REFERENCE_MIGRATION['arch_data_commit']}` "
        f"({REFERENCE_MIGRATION['arch_data_state_at_check']}, merge commit "
        f"`{REFERENCE_MIGRATION['arch_data_merge_commit']}`)",
        f"- Populace snapshot: {REFERENCE_MIGRATION['populace_pr']} at "
        f"`{REFERENCE_MIGRATION['populace_commit']}` "
        f"({REFERENCE_MIGRATION['populace_state_at_check']}, merge commit "
        f"`{REFERENCE_MIGRATION['populace_merge_commit']}`)",
        f"- Normalized source-table parity: {len(parity.tables)} tables checked, "
        f"{mismatches} mismatched, max absolute numeric difference "
        f"{parity.max_abs_numeric_diff:g}",
        "",
        parity_sentence,
        "",
        "| Metric | Comparability | Truth / target where it exists | Paper 2024-25 | Populace/Ledger 2024-25 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row.label} | {row.comparability} | {row.truth} | "
        f"{row.paper_value} | {row.populace_value} |"
        for row in comparison_rows(populace)
    )
    lines.extend(
        [
            "",
            "Interpretation: this is not a silent replacement for the paper's "
            f"published synthetic population. {interpretation_target_sentence}"
            "and the directly comparable rows are ONS population, employment "
            "bands, and VAT liability by turnover band. HMRC turnover-band "
            "accuracy, sector distribution, and overall accuracy are computed "
            "under project-specific definitions, so the overall scores are not "
            "a like-for-like quality ranking. VAT liability by sector remains "
            "an informational diagnostic, not a calibrated target.",
            "",
            "Recompute command:",
            "",
            "```bash",
            REFERENCE_MIGRATION["comparison_command"],
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def provenance_payload(
    populace: CalibrationSnapshot = REFERENCE_POPULACE_LEDGER_2024_25,
    parity: ParitySummary = REFERENCE_PARITY,
    *,
    iterations: int = 1_000,
    seed: int = 42,
) -> dict:
    """Return structured provenance for the checked comparison."""

    return {
        "vintage": "2024-25",
        "run_parameters": {
            "seed": seed,
            "populace_iterations": iterations,
        },
        "migration_snapshot": REFERENCE_MIGRATION,
        "paper_2024_25": asdict(PAPER_2024_25),
        "populace_ledger_2024_25": asdict(populace),
        "ledger_paper_parity": {
            "facts_count": parity.facts_count,
            "facts_sha256": parity.facts_sha256,
            "tables_checked": len(parity.tables),
            "mismatched_tables": list(parity.mismatched_tables),
            "max_abs_numeric_diff": parity.max_abs_numeric_diff,
            "tables": [asdict(table) for table in parity.tables],
        },
    }


def load_populace_source_data(facts_jsonl: Path):
    """Read Ledger facts and build Populace source data."""

    try:
        from populace.build.uk_runtime.firm_generation import (
            uk_firm_source_data_from_ledger_facts,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading Ledger firm facts requires the experimental Populace firm "
            "generator. Install the Populace source tree containing "
            "populace.build.uk_runtime.firm_generation in this environment."
        ) from exc

    facts = _read_facts(facts_jsonl)
    return uk_firm_source_data_from_ledger_facts(facts, data_vintage="2024-25")


def source_table_parity(
    facts_jsonl: Path,
    *,
    processed_dir: Path = PROCESSED_DATA_DIR / "2024-25",
) -> ParitySummary:
    """Compare Ledger-derived 2024-25 source tables with paper CSV inputs."""

    data = load_populace_source_data(facts_jsonl)
    tables = (
        _compare_by_key(
            "ons_turnover_by_sic_band",
            data.ons_turnover,
            processed_dir / "ons_firm_turnover.csv",
            ["SIC Code"],
            [
                "0-49",
                "50-99",
                "100-249",
                "250-499",
                "500-999",
                "1000-4999",
                "5000+",
                "Total",
            ],
            paper_filter_key="SIC Code",
        ),
        _compare_by_key(
            "ons_employment_by_sic_band",
            data.ons_employment,
            processed_dir / "ons_firm_employment.csv",
            ["SIC Code"],
            ["0-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250+", "Total"],
            paper_filter_key="SIC Code",
        ),
        _compare_by_key(
            "hmrc_population_by_turnover_band",
            data.hmrc_population_band,
            processed_dir / "hmrc_vat_population_by_turnover_band.csv",
            ["Financial_Year"],
            [
                "Negative_or_Zero",
                "£1_to_Threshold",
                "£Threshold_to_£150k",
                "£150k_to_£300k",
                "£300k_to_£500k",
                "£500k_to_£1m",
                "£1m_to_£10m",
                "Greater_than_£10m",
            ],
        ),
        _compare_by_key(
            "hmrc_population_by_sic",
            data.hmrc_population_sector,
            processed_dir / "hmrc_vat_population_by_sector.csv",
            ["Trade_Sector"],
            ["2024-25"],
        ),
        _compare_by_key(
            "hmrc_liability_by_turnover_band",
            data.hmrc_liability_band,
            processed_dir / "hmrc_vat_liability_by_turnover_band.csv",
            ["Financial_Year"],
            [
                "Negative_or_Zero",
                "£1_to_Threshold",
                "£Threshold_to_£150k",
                "£150k_to_£300k",
                "£300k_to_£500k",
                "£500k_to_£1m",
                "£1m_to_£10m",
                "Greater_than_£10m",
            ],
        ),
        _compare_by_key(
            "hmrc_liability_by_sic",
            data.hmrc_liability_sector,
            processed_dir / "hmrc_vat_liability_by_sector.csv",
            ["Trade_Sector"],
            ["2024-25"],
        ),
    )
    return ParitySummary(
        facts_count=len(_read_facts(facts_jsonl)),
        facts_sha256=_sha256(facts_jsonl),
        tables=tables,
    )


def run_populace_from_ledger_facts(
    facts_jsonl: Path,
    *,
    iterations: int = 1_000,
    seed: int = 42,
) -> CalibrationSnapshot:
    """Generate the Populace firm population from Ledger facts and score it."""

    try:
        from populace.build.uk_runtime.firm_generation import (
            UKFirmGenerationConfig,
            generate_uk_firm_population,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Recomputing the Populace/Ledger comparison requires the "
            "experimental Populace firm generator. Install the Populace source "
            "tree containing populace.build.uk_runtime.firm_generation in this "
            "environment, then rerun with --facts-jsonl."
        ) from exc

    data = load_populace_source_data(facts_jsonl)
    result = generate_uk_firm_population(
        data,
        UKFirmGenerationConfig(
            data_vintage="2024-25",
            n_iterations=iterations,
            seed=seed,
        ),
    )
    validation = result.validation
    return CalibrationSnapshot(
        rows=len(result.firms),
        weighted_population=float(result.firms["firm_weight"].sum()),
        hmrc_bands=validation.hmrc_bands * 100.0,
        ons_population=validation.ons_population * 100.0,
        employment=validation.employment * 100.0,
        sector=validation.sector * 100.0,
        vat_liability_band=validation.vat_liability_band * 100.0,
        overall=validation.overall * 100.0,
        vat_liability_sector=validation.vat_liability_sector * 100.0,
    )


def _compare_by_key(
    name: str,
    ledger_df,
    paper_csv: Path,
    key_cols: list[str],
    value_cols: list[str],
    *,
    paper_filter_key: str | None = None,
) -> ParityTable:
    import pandas as pd

    ledger = ledger_df[key_cols + value_cols].copy()
    paper = pd.read_csv(paper_csv)
    if paper_filter_key is not None:
        paper = paper[paper[paper_filter_key].notna()].copy()
    paper = paper[key_cols + value_cols].copy()

    ledger = _normalize_keys(ledger, key_cols)
    paper = _normalize_keys(paper, key_cols)
    for column in value_cols:
        ledger[column] = pd.to_numeric(ledger[column])
        paper[column] = pd.to_numeric(paper[column])

    ledger = ledger.sort_values(key_cols).reset_index(drop=True)
    paper = paper.sort_values(key_cols).reset_index(drop=True)
    same_shape = ledger.shape == paper.shape
    same_keys = same_shape and ledger[key_cols].equals(paper[key_cols])
    max_abs = None
    values_equal = False
    if same_shape and same_keys:
        diff = (ledger[value_cols].astype(float) - paper[value_cols].astype(float)).abs()
        max_abs = float(diff.to_numpy().max()) if diff.size else 0.0
        values_equal = max_abs < 1e-9

    return ParityTable(
        name=name,
        rows=len(ledger),
        paper_rows_after_filter=len(paper),
        same_keys=bool(same_keys),
        values_equal=bool(values_equal),
        max_abs_numeric_diff=max_abs,
    )


def _normalize_keys(df, key_cols: list[str]):
    import pandas as pd

    out = df.copy()
    for key in key_cols:
        converted = pd.to_numeric(out[key], errors="coerce")
        if converted.notna().all():
            out[key] = converted.astype(int)
        else:
            out[key] = out[key].astype(str)
    return out


def _read_facts(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count(value: float) -> str:
    return f"{value:,.0f}"


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--facts-jsonl",
        type=Path,
        default=None,
        help=(
            "Ledger consumer facts JSONL to recompute source parity and, by "
            "default, the Populace run. Omit to print the checked reference "
            "comparison."
        ),
    )
    parser.add_argument(
        "--paper-processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "2024-25",
        help="Paper processed CSV directory for Ledger parity checks.",
    )
    parser.add_argument(
        "--reference-population",
        action="store_true",
        help=(
            "When --facts-jsonl is supplied, compute Ledger/paper parity from "
            "facts but use the checked reference Populace population snapshot "
            "instead of rerunning the slow optimizer."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1_000,
        help="Populace calibration iterations when --facts-jsonl is supplied.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Populace generator seed when --facts-jsonl is supplied.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the Markdown report.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write structured JSON provenance.",
    )
    return parser


def main(argv: list[str] | None = None) -> str:
    """CLI implementation."""

    args = _build_parser().parse_args(argv)
    if args.facts_jsonl:
        parity = source_table_parity(
            args.facts_jsonl,
            processed_dir=args.paper_processed_dir,
        )
        populace = (
            REFERENCE_POPULACE_LEDGER_2024_25
            if args.reference_population
            else run_populace_from_ledger_facts(
                args.facts_jsonl,
                iterations=args.iterations,
                seed=args.seed,
            )
        )
    else:
        parity = REFERENCE_PARITY
        populace = REFERENCE_POPULACE_LEDGER_2024_25

    report_iterations = (
        1_000 if (not args.facts_jsonl or args.reference_population) else args.iterations
    )
    report_seed = 42 if (not args.facts_jsonl or args.reference_population) else args.seed
    report = format_comparison_report(
        populace,
        parity,
        iterations=report_iterations,
        seed=report_seed,
    )
    payload = provenance_payload(
        populace,
        parity,
        iterations=report_iterations,
        seed=report_seed,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(report)
    return report


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
