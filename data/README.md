# Data

This directory holds all input data for the `firm_microsim` package and paper.
It separates pristine official source files (`raw/`) from small machine-readable
derived tables (`processed/`) and from generated model output (`synthetic/`).

## Directory layout

```
data/
├── raw/                <- pristine official source files, copied verbatim, never edited
│   ├── ons/            <- ONS UK Business workbooks (.xlsx): 2024 + 2025 editions
│   └── hmrc/           <- HMRC Annual UK VAT Statistics: 2023-24 (.xls) + 2024-25 (.ods)
├── processed/          <- small derived CSV band tables, organised by data vintage
│   ├── 2023-24/        <- ONS 2024 + HMRC 2023-24  (VAT threshold £85k) — paper baseline
│   └── 2024-25/        <- ONS 2025 + HMRC 2024-25  (VAT threshold £90k) — latest gov data
├── synthetic/          <- generated synthetic firm population (NOT committed; see below)
└── README.md           <- this file
```

## Data vintages

The project keeps **two coherent official-data vintages** side by side. Each is a
self-contained set of the same six processed tables; they differ only in the data
year and the VAT registration threshold they embed:

| Vintage | ONS edition (turnover reference year) | HMRC edition | VAT threshold | Role |
| --- | --- | --- | --- | --- |
| `2023-24` | UK Business 2024 (FY2022/23) | Annual VAT 2023-24, provisional release of 6 Nov 2024 | **£85,000** (frozen since Apr 2017) | Baseline that matches the existing paper, bunching and notch model |
| `2024-25` | UK Business 2025 (FY2023/24) | Annual VAT 2024-25, release of 3 Dec 2025 | **£90,000** (raised 1 Apr 2024) | Latest HMRC data; forward-looking robustness |

Switching vintage in the generator is a one-liner — `firm-microsim --vintage
2024-25`, or `Config.for_vintage("2024-25")`, or set `DATA_VINTAGE` —
which selects the matching `processed/<vintage>/` directory **and** the right
threshold automatically. The default is `2023-24`.

> Do not mix vintages (e.g. a £85k-era ONS distribution with £90k-era VAT bands).
> Pick one threshold regime end-to-end; the bunching/notch design lives entirely
> at the threshold.

**ONS reference year.** Each ONS edition's turnover refers to the financial year
before its publication year: the 2024 edition to FY2022/23 and the 2025 edition
to FY2023/24 (note 4 of each workbook's notes sheet). The `2024-25` vintage's
frame turnover distribution therefore predates the £90,000 threshold of April
2024; only its HMRC tables are £90k-era. The ONS 2026 edition (turnover for
FY2024/25) is not used here.

**HMRC 2023-24 is the provisional release.** The `2023-24` HMRC tables come from
the release first published on 6 November 2024, which labels 2023-24
"Provisional" (2,178,950 traders, none of unknown turnover). The 2024-25 release
of 3 December 2025 revises 2023-24 to 2,319,000 traders, including 141,300 of
unknown turnover; the known turnover bands move by at most about 1%. The
`2023-24` vintage is kept on the provisional release so the paper's numbers
reproduce.

## Official sources

| Source | Publication | Vintage | URL |
| --- | --- | --- | --- |
| ONS | UK Business: Activity, Size and Location, 2024 | 2023-24 | https://www.ons.gov.uk/businessindustryandtrade/business/activitysizeandlocation/datasets/ukbusinessactivitysizeandlocation |
| ONS | UK Business: Activity, Size and Location, 2025 | 2024-25 | https://www.ons.gov.uk/businessindustryandtrade/business/activitysizeandlocation/datasets/ukbusinessactivitysizeandlocation |
| HMRC | Annual UK VAT Statistics 2023 to 2024 | 2023-24 | https://www.gov.uk/government/statistics/value-added-tax-vat-annual-statistics |
| HMRC | Annual UK VAT Statistics 2024 to 2025 | 2024-25 | https://www.gov.uk/government/statistics/value-added-tax-vat-annual-statistics |

The raw workbooks are the authoritative inputs. Everything in `processed/` is a
faithful extract of tables inside those workbooks (no modelling applied, except
the documented fills and totals noted under each table below), kept as
CSV so the package can read them without an Excel engine. The two ONS tables per
vintage are produced by `scripts/etl_ons_tables.py` (run `--check` to verify the
checked CSVs against the workbooks); the HMRC tables are manual extracts of the
named sheets described below.

### Statistical units and universes

Both ONS tables count **enterprises** (Table 8: turnover sizebands; Table 3:
employment sizebands), so their per-SIC totals agree to ONS disclosure rounding.
An earlier 2023-24 employment CSV reproduced Table 18 — **local units**, i.e.
individual sites (3,173,660 against 2,724,775 enterprises) — and was replaced in
issue #37. The generator treats the ONS frame as the modelled population and the
HMRC trader tables as its VAT-registered subset (see the top-level README).

### 2024-25 band re-aggregation note

The HMRC 2024-25 release **splits** the old `£1m–£10m` turnover band into three
(£1m–£1.35m, £1.35m–£1.6m, £1.6m–£10m). To keep the `2024-25/` tables
schema-identical to `2023-24/` (so the generator reads both unchanged), those
three sub-bands are **summed back** into a single `£1m_to_£10m` column. The
`£1_to_Threshold` / `£Threshold_to_£150k` columns in the 2024-25 tables use the
**£90k** threshold.

---

## `raw/`

### `raw/ons/ukbusinessworkbook2024.xlsx` · `raw/ons/ukbusinessworkbook2025new.xlsx`

ONS "UK Business: Activity, Size and Location" workbooks — the **2024** edition
(feeds the `2023-24/` vintage) and the **2025** edition (feeds `2024-25/`). Full
multi-sheet Excel releases. The `processed/<vintage>/ons_*` CSVs are extracted
from their size-band tables (counts of enterprises by SIC division and by
turnover band / employment band). Use these workbooks for any table not already
extracted into `processed/`.

### `raw/hmrc/Annual_UK_VAT_Statistics_2023-24.xls` · `raw/hmrc/Annual_UK_VAT_Statistics_2024-25.ods`

HMRC "Annual UK VAT Statistics" releases — **2023-24** (.xls; £85k threshold) and
**2024-25** (.ods; £90k threshold). Each contains the VAT trader population and
net VAT liability tables, broken down by annual turnover band, by trade sector,
and by financial year. The `processed/<vintage>/hmrc_*` CSVs are extracts of
these tables (see the re-aggregation note above for the 2024-25 £1m+ bands).
The HMRC CSVs are manual extracts; unlike the ONS tables there is no extract
script or `--check`.

The 2023-24 `.xls` is a format conversion of the published `.ods`, not the
publisher's file. It is cell-identical in Tables 2, 4a/4b and 5a/5b to both
published versions, which gov.uk no longer serves (both asset URLs now return
the 2024-25 file); archived copies on the Wayback Machine:

| Published 2023-24 file | sha256 |
| --- | --- |
| Original, 6 Nov 2024 (`.ods`) | `630a741db1a231d9a7d674bbc266df471dfae763736c79faa6fa9eb7d813781b` |
| Reissue, 23 Apr 2025 (`.ods`; these tables unchanged) | `96fb80936050f90fc8059ba0bf33ac9f470890584a775284df46c98de849a63a` |

sha256 of the raw files as committed:

| File | sha256 |
| --- | --- |
| `raw/hmrc/Annual_UK_VAT_Statistics_2023-24.xls` | `7de85d149d1e23aeb79f4f9764f59630753283c086adf2cc88726a29a3d7c784` |
| `raw/hmrc/Annual_UK_VAT_Statistics_2024-25.ods` | `03593a96d41bc91a7471a96ec2253efa34b21ed080484bb21a4b876391c93f06` |
| `raw/ons/ukbusinessworkbook2024.xlsx` | `c61def5fbfa460efa892af0744d4cc269243c4990170a1f596b20ccda144b7bf` |
| `raw/ons/ukbusinessworkbook2025new.xlsx` | `5e51666eb97e6170051154aa09181fc785558712fa66816d2590aef998a09b6b` |

---

## `processed/`

The **same six files** below exist under each vintage directory
(`processed/2023-24/` and `processed/2024-25/`) with **identical schemas** — only
the data year, the embedded VAT threshold, and the sector value-column header
(`2023-24` vs `2024-25`) differ. The generator reads whichever directory the
active vintage selects. Paths shown below are relative to a vintage directory.

All counts are rounded by the publishers, which is why some columns sum
slightly off the printed totals. ONS counts are control-rounded to base 5 for
disclosure. The HMRC 2023-24 release rounds to the nearest 10; the 2024-25
release rounds to the nearest 10 below 10,000 and to the nearest 100 at or
above it (counts and £ million alike). Negative liability cells are net
repayment positions.

### `ons_firm_turnover.csv`

Count of enterprises by SIC division and **turnover band (£ thousands)**, from
**Table 8** (United Kingdom columns) of the ONS UK Business workbook for the
vintage.

- Columns: `SIC Code`, `Description`, then one count column per turnover band,
  then `Total`.
- Turnover bands (units: £ thousand annual turnover):
  - `0-49`
  - `50-99`
  - `100-249`
  - `250-499`
  - `500-999`
  - `1000-4999`
  - `5000+`
- Rows: one per SIC division (SIC codes 01–99), with a final `Total` row.

### `ons_firm_employment.csv`

Count of enterprises by SIC division and **employment size band (number of
employees)**, from **Table 3** (United Kingdom columns) of the ONS UK Business
workbook for the vintage.

- Columns: `SIC Code`, `Description`, then one count column per employment band,
  then `Total`.
- Employment bands (units: number of employees):
  - `0-4`
  - `5-9`
  - `10-19`
  - `20-49`
  - `50-99`
  - `100-249`
  - `250+`
- Rows: one per SIC division (SIC codes 01–99), with a final `Total` row.

### `hmrc_vat_population_by_turnover_band.csv`

Number of VAT-registered traders by **annual turnover band**, from HMRC Annual
VAT Statistics. The `2023-24/` file has one row per financial year (2004-05
through 2023-24); the `2024-25/` file has a single 2024-25 row (the source's
revised 2004-05 to 2024-25 series is not extracted).

- Columns: `Financial_Year`, then one count column per turnover band, then
  `Total`.
- Turnover bands:
  - `Negative_or_Zero`
  - `£1_to_Threshold` (£1 up to the VAT registration threshold)
  - `£Threshold_to_£150k`
  - `£150k_to_£300k`
  - `£300k_to_£500k`
  - `£500k_to_£1m`
  - `£1m_to_£10m`
  - `Greater_than_£10m`
  - `Unknown`
- Note: the band edges are defined relative to the VAT registration
  **threshold** in force for that year (the threshold has been £85,000 for
  recent years and rose to £90,000 in 2024-25), so the `£1_to_Threshold` and
  `£Threshold_to_£150k` boundaries are year-dependent.
- `Total`: in `2023-24/` the published HMRC total (2,178,950, against a band sum
  of 2,178,960); in `2024-25/` the sum of the rounded bands (2,330,500), not the
  published total (2,330,400).

### `hmrc_vat_liability_by_turnover_band.csv`

Net VAT declared/liability (£ million) by **annual turnover band**, from HMRC
Annual VAT Statistics, with the same year coverage as the population table
(2004-05 through 2023-24 in `2023-24/`; a single 2024-25 row in `2024-25/`).

- Columns: `Financial_Year`, then one £m column per turnover band, then `Total`.
- Turnover bands (same definitions as the population table, minus `Unknown`):
  - `Negative_or_Zero`
  - `£1_to_Threshold`
  - `£Threshold_to_£150k`
  - `£150k_to_£300k`
  - `£300k_to_£500k`
  - `£500k_to_£1m`
  - `£1m_to_£10m`
  - `Greater_than_£10m`
- Values are in £ million and may be negative (net repayment positions, e.g. the
  `Negative_or_Zero` band).
- `Total`: in `2023-24/` the published total (£173,380m, against a band sum of
  £173,390m); in `2024-25/` the sum of the rounded bands (£177,170m), not the
  published total (£177,100m).

### `hmrc_vat_population_by_sector.csv`

Number of VAT-registered traders by **trade sector** for the vintage's year
(`2023-24` or `2024-25`), from HMRC Annual VAT Statistics.

- Columns: `Trade_Sector` (5-digit HMRC trade-sector code, e.g. `00001`),
  `Trade_Sub_Sector` (sector description), and a single year column named after
  the vintage (`2023-24` or `2024-25`) holding the trader count.
- Rows: one per trade sub-sector (codes 00001–00099). The `2023-24/` file ends
  with the published `Total` row (2,178,950); the `2024-25/` file has no `Total`
  row (published total 2,330,400).
- HMRC's `Blank/Unknown` sector row is not extracted (250 traders in 2023-24,
  360 in 2024-25), so the sector rows sum to slightly less than the published
  total (2023-24: 2,178,750 against 2,178,950).
- The package reads this value column positionally (last column), so it is
  robust to the year-named header.

### `hmrc_vat_liability_by_sector.csv`

Net VAT declared/liability (£ million) by **trade sector** for the vintage's year
(`2023-24` or `2024-25`), from HMRC Annual VAT Statistics.

- Columns: `Trade_Sector` (5-digit HMRC trade-sector code), `Trade_Sub_Sector`
  (sector description), and a single year column (`2023-24` or `2024-25`) holding
  net VAT liability in £ million.
- Rows: one per trade sub-sector (codes 00001–00099). Values may be negative
  (net repayment sectors).
- Four 2024-25 cells are suppressed in the source and are **not** source values:
  00007 Mining of metal ores, 00037 Sewerage and 00060 Programming and
  broadcasting (`<30`), and 00012 Tobacco (`SUPP`). They carry the 2023-24 values
  (−10, −10, −20 and 1,700). This table feeds only the sector-liability
  diagnostic, which is not a calibration target.

---

## `synthetic/`

Holds the **generated** synthetic firm population CSV produced by the
`firm_microsim` package (the model is calibrated to the `processed/` band
tables). This output is **not committed** to the repository — it is regenerated
by running the package. Only `.gitkeep` is tracked so the directory exists on a
fresh clone. See the top-level README / package docs for the generation command.

The synthetic population CSVs are not tracked: `.gitignore` excludes
`data/synthetic/*.csv` (no Git LFS involved), and the pipeline regenerates
them from the committed raw workbooks.

`data/processed/obr_vat_bunching.csv` holds the OBR's £1,000-band business
counts from the Economic and Fiscal Outlook, March 2023, Box 3.3 ("The impact of
the frozen VAT registration threshold"), Chart 3.C, taken from sheet C3.C of the
chapter-3 charts-and-tables workbook (OGL v3.0; workbook sha256
`bbd93969eb77460be10dd5cee5d0442f3484002b222940a1e986a4135abc2fcd`, fetched
2026-11-22). The values equal the workbook cells (rounded to 3 dp), extracted
as documented in `scripts/plot_obr_bunching.py`. Units are thousands of
businesses per £1,000 turnover band; the source does not state its universe or
bin convention (the loader assumes `(lo, lo+1]`). The 2025-26 column is a
projection under a frozen £85,000 threshold, which was not realised (the
threshold rose to £90,000 in April 2024). The 2023-24 calibration interpolates
the 2019-20 outturn and the 2025-26 projection columns
(`src/firm_microsim/data_loader.py`).
