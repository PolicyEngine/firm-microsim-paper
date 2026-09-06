#!/usr/bin/env python3
"""ETL: DBT *Business Population Estimates* detailed tables -> processed CSVs.

Extracts, for each vintage, the **unregistered** business stratum by SIC
division from Table 6 (UK divisions): the row "With no employees
(unregistered)" gives the number of businesses that are registered for neither
VAT nor PAYE and their total turnover (£ millions). These are the businesses
outside the ONS VAT/PAYE frame the generator is otherwise built on
(issue #25); all of them are below the VAT registration threshold by
construction.

Vintage mapping: BPE 2024 (start of 2024; turnover from VAT/SA returns for
the 12 months to end-2022/early-2023) feeds the 2023-24 vintage; BPE 2025
feeds 2024-25. Suppressed cells ("[c]") are left empty; the generator falls
back to the national mean turnover for those divisions.

Usage::

    python scripts/etl_bpe_tables.py            # rebuild both vintages
    python scripts/etl_bpe_tables.py --check    # exit 1 if any CSV would change
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import openpyxl

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "dbt"
PROCESSED = REPO / "data" / "processed"
VINTAGES = {
    "2023-24": RAW / "BPE_2024_detailed_tables.xlsx",
    "2024-25": RAW / "BPE_2025_detailed_tables.xlsx",
}
OUT_NAME = "bpe_unregistered_by_division.csv"
DIV_RE = re.compile(r"^(\d{2}) (.+)$")


def _cell(v):
    if v is None or (isinstance(v, str) and v.strip().startswith("[")):
        return ""
    return str(int(round(float(v))))


def extract(workbook: Path) -> list[list[str]]:
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    ws = wb["Table 6"]
    out = [["SIC Code", "Description", "unregistered_count", "unregistered_turnover_m",
            "all_count", "all_turnover_m"]]
    current = None
    for row in ws.iter_rows(values_only=True):
        label = row[0]
        if label is None:
            continue
        label = str(label).strip()
        m = DIV_RE.match(label)
        if m and row[1] is None and not m.group(2).startswith("to "):
            current = [m.group(1), m.group(2).replace(",", ";"), "", "", "", ""]
            out.append(current)
            continue
        if current is None:
            continue
        if label == "All businesses":
            current[4], current[5] = _cell(row[1]), _cell(row[3])
        elif label.startswith("With no employees (unr"):
            current[2], current[3] = _cell(row[1]), _cell(row[3])
    if len(out) - 1 < 80:
        raise ValueError(f"{workbook.name}: only {len(out) - 1} divisions parsed")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    changed = 0
    for vintage, workbook in VINTAGES.items():
        table = extract(workbook)
        text = "\n".join(",".join(r) for r in table) + "\n"
        target = PROCESSED / vintage / OUT_NAME
        if target.exists() and target.read_text() == text:
            print(f"unchanged  {target.relative_to(REPO)}")
            continue
        changed += 1
        if args.check:
            print(f"DIFFERS    {target.relative_to(REPO)}")
        else:
            target.write_text(text)
            print(f"rewrote    {target.relative_to(REPO)}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
