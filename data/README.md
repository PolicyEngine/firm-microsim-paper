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

| Vintage | ONS edition | HMRC edition | VAT threshold | Role |
| --- | --- | --- | --- | --- |
| `2023-24` | UK Business 2024 | Annual VAT 2023-24 | **£85,000** (frozen since Apr 2017) | Baseline that matches the existing paper, bunching and notch model |
| `2024-25` | UK Business 2025 | Annual VAT 2024-25 | **£90,000** (raised 1 Apr 2024) | Latest available gov data; forward-looking robustness |

Switching vintage in the generator is a one-liner — `firm-microsim --vintage
2024-25`, or `Config.for_vintage("2024-25")`, or set `DATA_VINTAGE` —
which selects the matching `processed/<vintage>/` directory **and** the right
threshold automatically. The default is `2023-24`.

> Do not mix vintages (e.g. a £85k-era ONS distribution with £90k-era VAT bands).
> Pick one threshold regime end-to-end; the bunching/notch design lives entirely
> at the threshold.

## Official sources

| Source | Publication | Vintage | URL |
| --- | --- | --- | --- |
| ONS | UK Business: Activity, Size and Location, 2024 | 2023-24 | https://www.ons.gov.uk/businessindustryandtrade/business/activitysizeandlocation/datasets/ukbusinessactivitysizeandlocation |
| ONS | UK Business: Activity, Size and Location, 2025 | 2024-25 | https://www.ons.gov.uk/businessindustryandtrade/business/activitysizeandlocation/datasets/ukbusinessactivitysizeandlocation |
| HMRC | Annual UK VAT Statistics 2023 to 2024 | 2023-24 | https://www.gov.uk/government/statistics/value-added-tax-vat-annual-statistics |
| HMRC | Annual UK VAT Statistics 2024 to 2025 | 2024-25 | https://www.gov.uk/government/statistics/value-added-tax-vat-annual-statistics |

The raw workbooks are the authoritative inputs. Everything in `processed/` is a
faithful extract of tables inside those workbooks (no modelling applied), kept as
CSV so the package can read them without an Excel engine.

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

---

## `processed/`

The **same six files** below exist under each vintage directory
(`processed/2023-24/` and `processed/2024-25/`) with **identical schemas** — only
the data year, the embedded VAT threshold, and the sector value-column header
(`2023-24` vs `2024-25`) differ. The generator reads whichever directory the
active vintage selects. Paths shown below are relative to a vintage directory.

All counts are rounded by the publishers (ONS/HMRC round to the nearest 5 for
disclosure control; HMRC VAT tables are typically rounded to the nearest 10),
which is why some columns sum slightly off the printed totals and why a few
liability cells are negative (net repayment positions).

### `ons_firm_turnover.csv`

Count of enterprises by SIC sector and **turnover band (£ thousands)**, from the
ONS UK Business 2024 workbook.

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

Count of enterprises by SIC sector and **employment size band (number of
employees)**, from the ONS UK Business 2024 workbook.

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

Number of VAT-registered traders by **annual turnover band**, one row per
financial year (2004-05 through 2023-24), from HMRC Annual VAT Statistics.

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

### `hmrc_vat_liability_by_turnover_band.csv`

Net VAT declared/liability (£ million) by **annual turnover band**, one row per
financial year (2004-05 through 2023-24), from HMRC Annual VAT Statistics.

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

### `hmrc_vat_population_by_sector.csv`

Number of VAT-registered traders by **trade sector** for the vintage's year
(`2023-24` or `2024-25`), from HMRC Annual VAT Statistics.

- Columns: `Trade_Sector` (5-digit HMRC trade-sector code, e.g. `00001`),
  `Trade_Sub_Sector` (sector description), and a single year column named after
  the vintage (`2023-24` or `2024-25`) holding the trader count.
- Rows: one per trade sub-sector (codes 00001–00099), with a final `Total` row.
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

---

## `synthetic/`

Holds the **generated** synthetic firm population CSV produced by the
`firm_microsim` package (the model is calibrated to the `processed/` band
tables). This output is **not committed** to the repository — it is regenerated
by running the package. Only `.gitkeep` is tracked so the directory exists on a
fresh clone. See the top-level README / package docs for the generation command.

Because the synthetic population file can be large, it is configured for Git LFS
in `.gitattributes` for the case where someone does choose to track it; by
default `.gitignore` excludes `data/synthetic/*.csv`.
