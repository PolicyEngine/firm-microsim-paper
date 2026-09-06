#!/usr/bin/env python3
"""ETL: ONS *UK Business: Activity, Size and Location* workbooks -> processed CSVs.

Rebuilds, for each vintage, the two ONS input tables the generator reads:

* ``ons_firm_turnover.csv``   from **Table 8**  — VAT and/or PAYE based
  *enterprises* within region by SIC division and turnover sizeband (£000s),
  United Kingdom columns.
* ``ons_firm_employment.csv`` from **Table 3**  — VAT and/or PAYE based
  *enterprises* within region by SIC division and employment sizeband,
  United Kingdom columns.

Both tables count the same statistical unit (enterprises), so their per-SIC
totals agree to ONS disclosure rounding. Table 18 (local units by employment
sizeband) is NOT used: a local unit is an individual site, and the 2023-24
employment CSV shipped before this script reproduced Table 18 by mistake
(issue #37).

Source layout (both editions): row 4 holds the geography header
(``K02000001 United Kingdom`` in column B), row 5 the sizeband labels, rows
6-93 the 88 SIC divisions ``"01 : Description"`` ... ``"99 : ..."`` and row 94
the ``Total`` row. Columns B-I are the UK sizebands plus ``Total``.

Usage::

    python scripts/etl_ons_tables.py            # rebuild both vintages
    python scripts/etl_ons_tables.py --check    # exit 1 if any CSV would change
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openpyxl

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "ons"
PROCESSED = REPO / "data" / "processed"

VINTAGES = {
    "2023-24": RAW / "ukbusinessworkbook2024.xlsx",
    "2024-25": RAW / "ukbusinessworkbook2025new.xlsx",
}
TABLES = {
    "ons_firm_turnover.csv": ("Table 8", ["0-49", "50-99", "100-249", "250-499", "500-999", "1000-4999", "5000+"]),
    "ons_firm_employment.csv": ("Table 3", ["0-4", "5-9", "10-19", "20-49", "50-99", "100-249", "250+"]),
}
UK_HEADER_ROW = 4
LABEL_ROW = 5
FIRST_DATA_ROW = 6


def extract(workbook: Path, sheet: str, bands: list[str]) -> list[list[str]]:
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    uk = str(rows[UK_HEADER_ROW - 1][1])
    if "United Kingdom" not in uk:
        raise ValueError(f"{workbook.name} {sheet}: expected UK header in B4, got {uk!r}")
    labels = [str(v).strip() for v in rows[LABEL_ROW - 1][1:9]]
    if labels != bands + ["Total"]:
        raise ValueError(f"{workbook.name} {sheet}: sizeband labels {labels} != {bands + ['Total']}")
    out = [["SIC Code", "Description"] + bands + ["Total"]]
    for row in rows[FIRST_DATA_ROW - 1:]:
        label = row[0]
        if label is None:
            continue
        label = str(label).strip()
        vals = [int(v) for v in row[1:9]]
        if label == "Total":
            out.append(["", "Total"] + [str(v) for v in vals])
            break
        code, _, desc = label.partition(" : ")
        if not code.isdigit():
            raise ValueError(f"{workbook.name} {sheet}: unexpected row label {label!r}")
        out.append([code.zfill(2), desc.replace(",", ";"), *[str(v) for v in vals]])
    if len(out) != 1 + 88 + 1:
        raise ValueError(f"{workbook.name} {sheet}: expected 88 SIC rows + Total, got {len(out) - 1}")
    return out


def write_csv(path: Path, table: list[list[str]]) -> str:
    text = "\n".join(",".join(r) for r in table) + "\n"
    path.write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 on any difference")
    args = ap.parse_args(argv)
    changed = 0
    for vintage, workbook in VINTAGES.items():
        for fname, (sheet, bands) in TABLES.items():
            table = extract(workbook, sheet, bands)
            text = "\n".join(",".join(r) for r in table) + "\n"
            target = PROCESSED / vintage / fname
            current = target.read_text() if target.exists() else None
            same = current == text
            if same:
                print(f"unchanged  {target.relative_to(REPO)}  ({sheet})")
                continue
            changed += 1
            if args.check:
                print(f"DIFFERS    {target.relative_to(REPO)}  ({sheet})")
            else:
                target.write_text(text)
                print(f"rewrote    {target.relative_to(REPO)}  ({sheet})")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
