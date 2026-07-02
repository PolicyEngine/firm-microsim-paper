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
├── src/firm_microsim/    # installable package
│   ├── config.py         # vintage, VAT threshold, paths, hyperparameters
│   ├── generate.py       # synthetic-population generator
│   ├── static/           # static threshold-reform results
│   ├── bunching/         # reduced-form bunching estimator
│   ├── notch/            # structural notch diagnostics
│   ├── dynamic/          # conditional behavioural costing
│   └── analysis/         # paper table and diagnostic scripts
├── pyproject.toml        # package metadata, dependencies, console scripts
├── paper/                # Quarto/LaTeX manuscript sources and checked PDF
├── results/              # generated figures + calibration_accuracy.txt
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

The result is ~2.94M firm rows weighted to ~2.5M UK firms. Because the population
is calibrated **to** the HMRC aggregates, agreement with them is an internal
consistency check, not external validation.

The official target surface is also being mirrored into PolicyEngine Ledger and
Populace. This repository keeps the paper's archived CSV inputs and generator for
reproducibility, and includes a Populace/Ledger comparison command so the pinned
migration snapshot can be audited without silently changing the published paper
population.

## Data vintages — single version, one-line switch

The pipeline is **single-version**: there is one `VAT_THRESHOLD`, not separate
85k/90k scripts. Two coherent official-data vintages are available and selected
with a single switch (see `data/README.md`):

| Vintage | Data | Threshold | Role |
| --- | --- | --- | --- |
| `2023-24` (default) | ONS 2024 + HMRC 2023-24 | £85,000 | Paper baseline |
| `2024-25` | ONS 2025 + HMRC 2024-25 | £90,000 | Latest gov data |

## Usage

Install the package in editable mode, then run the package entry points:

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"

firm-microsim          # ALL DATA: every vintage + calibration report + figures
firm-microsim-static   # ALL STATIC RESULTS: threshold-reform figures
```

`firm-microsim` with no arguments runs the full data build — it
generates `synthetic_firms_<vintage>.csv` for every vintage, writes
`results/calibration_accuracy.txt`, and renders the descriptive figures. Single
steps are still available:

```bash
firm-microsim --vintage 2024-25             # one vintage only (£90k)
firm-microsim --threshold 88 --seed 7 --output my_run.csv
firm-microsim-report                        # calibration report only
firm-microsim-figures                       # descriptive figures only
firm-microsim-populace-ledger               # Populace/Ledger comparison
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

## Paper

The manuscript is rendered with Quarto, while preserving the existing LaTeX
section files and template:

```bash
cd paper
quarto render index.qmd --to pdf
```

The PolicyBench-style web paper is generated from the same LaTeX sections and
rendered with PolicyEngine design tokens:

```bash
cd paper
python3 build_site.py
```

The deployable static site is written to `paper/site/`. Quarto writes PDF render
artifacts to `paper/out/`; the checked-in submission PDF is `paper/main.pdf`.
The legacy direct-LaTeX entrypoint `paper/main.tex` remains available for
debugging and uses the same shared frontmatter/body inputs.

The current Vercel deployment is
<https://firm-microsim-paper.vercel.app/>.

## Calibration accuracy

The population is calibrated to **five** official ONS + HMRC target groups; the
validator scores each dimension as
`accuracy = max(0, 1 − |synthetic − target| / |target|)`. The displayed error is
the clipped complement of that score, not a signed relative error. **Overall**
is the simple mean over the five calibrated dimensions below.
Reproduce with:

```bash
firm-microsim-report
```

| Calibrated dimension | 85k (2023-24) | 90k (2024-25) |
| --- | ---: | ---: |
| HMRC turnover bands | 93.8% | 93.1% |
| ONS population | 90.3% | 92.6% |
| Employment bands | 77.9% | 92.4% |
| Sector distribution | 92.7% | 94.2% |
| VAT liability by band (6 calibrated bands) | 91.3% | 92.0% |
| **Overall (5 calibrated dimensions)** | **89.2%** | **92.8%** |

**VAT liability by *sector*** is **not** a calibration target — it is reported as
an informational diagnostic only, and neither is the **below-threshold
(£1-to-Threshold) liability band**: its HMRC total is remitted by voluntary
registrants (input-reclaim traders averaging ~£2,150 net) whom the
standard-rate-on-value-added liability model does not represent, so calibrating
it against the whole below-threshold population distorts near-threshold
weights (see the paper's data section). The model draws per-firm input
shares (mean value-added share ≈ 40%) and sets net liability
`v = 0.20 × (turnover − input)` — the standard rate applied to value added — but
does not yet calibrate the **input/output tax structure by sector**, so
per-sector net liability is structurally unhittable and is gated off via
`Config.calibrate_vat_liability_sector = False`. Restoring it after input/output
calibration is tracked in issues
[#1](https://github.com/PolicyEngine/firm-microsim-paper/issues/1) and
[#2](https://github.com/PolicyEngine/firm-microsim-paper/issues/2). An earlier
build set `v = turnover − input` (no 0.20 factor); the correction and its
consequences are documented in
[#15](https://github.com/PolicyEngine/firm-microsim-paper/issues/15) and the
paper's Section 5.

## Fast iteration builds

`firm-microsim --fast` runs the full pipeline on a stratified sample (~20% of
rows: 30% inside the £15k–£150k analysis window, 5% outside, per-stratum
floors), carrying the thinned mass as base weights so every calibration target
remains a true total. A vintage builds in ~15 seconds instead of ~13 minutes;
headline aggregates reproduce the full build within ~0.3% and local bunching
statistics within ~5%. Use for development only — release artifacts are
full-size. Generator-seed sensitivity of the full build is recorded in
`results/seed_sensitivity.txt` (E ±2%, raise ±£2.1m / taper ±£1.5m across
seeds; reproduce with `scripts/seed_sensitivity.py`).

## Populace/Ledger migration check

`firm-microsim-populace-ledger` reports the current migration comparison. The
checked reference run used the 2024-25 Ledger target surface from
[PolicyEngine/ledger#67](https://github.com/PolicyEngine/ledger/pull/67)
at `cd98b5cb7b1604fbf7750689a429bbc356e5603a` and Populace's experimental UK
firm generator from
[PolicyEngine/populace#223](https://github.com/PolicyEngine/populace/pull/223)
at `fa20daf75ff023e5e88731a140f456f58e0b864e`. Both upstream PRs merged on
June 30, 2026: Ledger at merge commit
`ac643afa0c1d45fc4abd0268dc5aa7c843440b38`, and Populace at merge commit
`8271d767244161631253ad1d9ad792a82e2b96b4`. The reference population uses
1,000 calibration iterations:

```bash
firm-microsim-populace-ledger \
  --output results/populace_ledger_comparison.txt \
  --json-output results/populace_ledger_provenance.json
```

When `populace-build` is installed from the Populace source tree, the same command
can recompute the table and paper-CSV parity from Ledger consumer facts:

```bash
firm-microsim-populace-ledger \
  --facts-jsonl /path/to/uk_firm_consumer_facts.jsonl \
  --iterations 1000 \
  --output results/populace_ledger_comparison.txt \
  --json-output results/populace_ledger_provenance.json
```

The current reference comparison shows exact parity between the Ledger-backed
targets and the paper's processed 2024-25 numeric inputs: six normalized source
tables checked, zero mismatches, max numeric difference 0. It does **not** exactly
replicate the paper's generated synthetic population: Populace's shared optimizer
landed at 93.8% overall accuracy under its own validator versus the paper's
then-90.5% (2024-25 scores 92.8% on the corrected build,
`results/calibration_accuracy.txt`), but that overall pair is **not
like-for-like**: HMRC turnover-band accuracy uses different band sets, and sector
distribution reflects different calibration-target definitions. The directly
comparable rows are ONS population, employment bands, and VAT liability by
turnover band. **Note:** the pinned Populace snapshot predates this repo's
net-liability correction (issue #15) and inherits the same `v = turnover − input`
mis-scaling; the target-parity result is unaffected (it concerns input tables,
not generated rows), but the Populace generator needs the same fix upstream. The
Populace/Ledger path remains a migration check rather than a silent replacement
for the paper's archived generator/results.

## Figures

Figures follow the project house style: single clean panels (no embedded titles,
source notes, or logos — captions and side-by-side layouts are composed in
LaTeX), teal palette, saved as snake_case PNGs to `results/` at 300 dpi. They are
produced by the in-package `firm_microsim.figures` module and generated for
**both vintages** (two full sets, suffixed `_85k` / `_90k`):

```bash
firm-microsim-figures          # regenerate every figure, both vintages
```

`results/` then contains:

| Figure | 85k (2023-24) | 90k (2024-25) | Source |
| --- | --- | --- | --- |
| All UK firms by turnover band | `firms_by_turnover_band_85k.png` | `firms_by_turnover_band_90k.png` | ONS |
| VAT-registered firms by turnover band | `vat_firms_by_turnover_band_85k.png` | `vat_firms_by_turnover_band_90k.png` | HMRC |
| Full-range turnover distribution | `turnover_distribution_85k.png` | `turnover_distribution_90k.png` | synthetic |

The turnover-distribution figures require the matching synthetic CSV
(`data/synthetic/synthetic_firms_<vintage>.csv`); generate it first with
`firm-microsim --vintage <vintage> --output synthetic_firms_<vintage>.csv`.

> **Note on ONS counts:** firm-count figures sum the per-SIC rows only — the ONS
> band tables include a `Total` summary row that must be excluded, or every firm
> is counted twice (a bug present in earlier drafts that doubled the ONS panel).

## Static threshold reform results

The `firm_microsim.static` module costs VAT-threshold reforms mechanically (turnover held
fixed; only registration status changes), reproducing the paper's static
results. Run:

```bash
firm-microsim-static          # -> results/{vat_threshold_revenue_impact,revenue_impact_2025_26,firms_impact_2025_26}.png
```

- `vat_threshold_revenue_impact.png` — the £85k→£90k anchor reform vs HMRC's
  published costing, by fiscal year. **Built on the £85k / 2023-24 vintage** —
  the pre-reform basis HMRC actually had at the 6 March 2024 costing (the
  threshold was still £85k until 1 April 2024). Full-deregistration model
  −317/−323/−195/−66/+103 vs HMRC −150/−185/−125/−50/+65 £m (43% voluntary
  retention: −180/−184/−111/−38/+59); both turn positive by 2028-29. Released
  band holds ≈ 48k registered firms, next to HMRC's published 28,000 expected
  first-year deregistrations. See `results/static_sweep.txt`.
- `revenue_impact_2025_26.png` / `firms_impact_2025_26.png` — the forward static
  sweep of registration thresholds (£70k–£120k) vs the current £90k baseline,
  **on the £90k / 2024-25 vintage**.

**Two vintages, two exercises.** The anchor reform uses the £85k vintage, where
the affected `[85,90)k` band sits *above* the £85k registration threshold and is
cleanly populated with registered firms — so a **simple band-sum** suffices (no
de-bunching). The forward sweep uses the current £90k vintage; there the
`[85,90)k` firms are *below* threshold and the calibration concentrates weight
on them, so the sweep instead fits the clean above-threshold firm/liability
profile and extrapolates it across the threshold
(`StaticVATModel._counterfactual_bins`, unaged turnover scaled to the fiscal
year by a nominal-growth factor). Revenue and the anchor reform match the paper
and HMRC closely; the forward-sweep firm-count magnitudes run low because the
regenerated population has a lower near-threshold VAT-paying-firm density.
