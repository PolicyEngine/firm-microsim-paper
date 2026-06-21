# firm-microsim-paper

Synthetic firm-level microsimulation of the UK VAT registration threshold.

This repository builds an open, firm-level synthetic population of UK businesses
— calibrated to official ONS and HMRC aggregates — that resolves the turnover
distribution at the individual-firm level around the VAT registration threshold,
where published statistics only report coarse bands. It supports bunching
estimation, static revenue costings, and dynamic firm re-optimisation around the
threshold notch.

## Layout

```
firm-microsim-paper/
├── data/                 # official inputs + generated output (see data/README.md)
│   ├── raw/              # pristine ONS + HMRC source workbooks
│   ├── processed/        # derived band tables, by vintage (2023-24, 2024-25)
│   └── synthetic/        # generated synthetic population (regenerated, not committed)
├── firm_microsim/        # the data-generation package + figures
│   ├── config.py         # single source of truth: vintage, VAT threshold, paths, hyperparams
│   ├── data_loader.py    # load processed ONS + HMRC tables -> calibration targets
│   ├── calibration.py    # target matrix + torch weight optimisation
│   ├── generate.py       # full pipeline orchestrator
│   ├── validate.py       # calibration-accuracy report vs official targets
│   ├── figures.py        # house-style paper figures -> results/
│   └── __main__.py       # CLI: python -m firm_microsim
├── results/              # generated figures (snake_case PNGs, both vintages)
└── requirements.txt
```

## Method (data generation)

A two-stage synthetic-population pipeline, parameterised by a single VAT
threshold:

1. **Draw base firms** from the ONS business structure — sample continuous
   within-band turnover, employment, and intermediate inputs for individual firms
   so the population has firm-level resolution the official bands lack.
2. **Calibrate firm weights** by multi-objective optimisation (Adam, symmetric
   relative-error loss) so weighted totals reproduce the official targets — HMRC
   VAT-registered counts by turnover band and by sector, ONS employment-band
   totals, and HMRC VAT-liability totals — with turnover bands weighted most
   heavily. VAT registration is then assigned: mandatory above the threshold,
   voluntary below at the HMRC-calibrated rate.

The result is ~2.94M firm rows weighted to ~2.0M UK firms. Because the population
is calibrated **to** the HMRC aggregates, agreement with them is an internal
consistency check, not external validation.

## Data vintages — single version, one-line switch

The pipeline is **single-version**: there is one `VAT_THRESHOLD`, not separate
85k/90k scripts. Two coherent official-data vintages are available and selected
with a single switch (see `data/README.md`):

| Vintage | Data | Threshold | Role |
| --- | --- | --- | --- |
| `2023-24` (default) | ONS 2024 + HMRC 2023-24 | £85,000 | Paper baseline |
| `2024-25` | ONS 2025 + HMRC 2024-25 | £90,000 | Latest gov data |

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate the synthetic population (default: 2023-24 / £85k baseline)
python -m firm_microsim

# Latest vintage (2024-25 / £90k) — switches data dir AND threshold together
python -m firm_microsim --vintage 2024-25

# Other overrides
python -m firm_microsim --threshold 88 --seed 7 --output my_run.csv
```

```python
import firm_microsim
df = firm_microsim.generate()                      # baseline
df = firm_microsim.generate(vintage="2024-25")     # latest
df, report = firm_microsim.generate(return_report=True)
```

Output is written to `data/synthetic/synthetic_firms.csv`
(`sic_code, annual_turnover_k, annual_input_k, vat_liability_k, employment,
weight, vat_registered`).

## Calibration accuracy

The population is calibrated to **five** official ONS + HMRC targets; the
validator scores each dimension as `accuracy = 1 − relative error` (0–1).
Reproduce with:

```bash
python scripts/report_calibration.py
```

| Calibrated dimension | 85k (2023-24) | 90k (2024-25) |
| --- | ---: | ---: |
| HMRC turnover bands | 93.0% | 92.6% |
| ONS population | 91.0% | 93.9% |
| Employment bands | 78.2% | 80.3% |
| Sector distribution | 92.5% | 94.4% |
| VAT liability by band | 94.5% | 81.3% |
| **Overall (5 calibrated dimensions)** | **89.9%** | **88.5%** |

**VAT liability by *sector*** is **not** a calibration target — it is reported as
an informational diagnostic only (47.7% / 18.8%). The model fixes firm inputs
and sets liability = turnover − input but does not yet calibrate the
**input/output tax structure**, so per-sector net liability is structurally
unhittable and, while targeted, competed with the dimensions above (it scored
43.9% / −121.1% and dragged the naive mean down). It is gated off via
`Config.calibrate_vat_liability_sector = False`. Restoring it after input/output
calibration is tracked in issues
[#1](https://github.com/PolicyEngine/firm-microsim-paper/issues/1) and
[#2](https://github.com/PolicyEngine/firm-microsim-paper/issues/2).

## Figures

Figures follow the project house style: single clean panels (no embedded titles,
source notes, or logos — captions and side-by-side layouts are composed in
LaTeX), teal palette, saved as snake_case PNGs to `results/` at 300 dpi. They are
produced by the in-package `firm_microsim.figures` module and generated for
**both vintages** (two full sets, suffixed `_85k` / `_90k`):

```bash
python -m firm_microsim.figures          # regenerate every figure, both vintages
```

`results/` then contains:

| Figure | 85k (2023-24) | 90k (2024-25) | Source |
| --- | --- | --- | --- |
| All UK firms by turnover band | `firms_by_turnover_band_85k.png` | `firms_by_turnover_band_90k.png` | ONS |
| VAT-registered firms by turnover band | `vat_firms_by_turnover_band_85k.png` | `vat_firms_by_turnover_band_90k.png` | HMRC |
| Full-range turnover distribution | `turnover_distribution_85k.png` | `turnover_distribution_90k.png` | synthetic |

The turnover-distribution figures require the matching synthetic CSV
(`data/synthetic/synthetic_firms_<vintage>.csv`); generate it first with
`python -m firm_microsim --vintage <vintage> --output synthetic_firms_<vintage>.csv`.

> **Note on ONS counts:** firm-count figures sum the per-SIC rows only — the ONS
> band tables include a `Total` summary row that must be excluded, or every firm
> is counted twice (a bug present in earlier drafts that doubled the ONS panel).
