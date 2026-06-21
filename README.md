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
├── firm_microsim/        # the data-generation package
│   ├── config.py         # single source of truth: vintage, VAT threshold, paths, hyperparams
│   ├── data_loader.py    # load processed ONS + HMRC tables -> calibration targets
│   ├── calibration.py    # target matrix + torch weight optimisation
│   ├── generate.py       # full pipeline orchestrator
│   ├── validate.py       # calibration-accuracy report vs official targets
│   └── __main__.py       # CLI: python -m firm_microsim
├── figures/              # standalone figure scripts
│   ├── plot_band_distributions.py    # two-panel ONS + HMRC firm counts by band
│   └── plot_turnover_distribution.py # full-range weighted turnover histogram
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

## Figures

```bash
# Firm counts by turnover band (two panels: ONS + HMRC). Defaults to 2024-25/£90k.
python figures/plot_band_distributions.py                 # -> figures/band_distributions.png
python figures/plot_band_distributions.py --vintage 2023-24

# Full-range weighted turnover distribution (needs the synthetic CSV).
python figures/plot_turnover_distribution.py              # -> figures/turnover_distribution_full.png
```

> **Note on ONS counts:** firm-count figures sum the per-SIC rows only — the ONS
> band tables include a `Total` summary row that must be excluded, or every firm
> is counted twice (a bug present in earlier drafts that doubled the ONS panel).
