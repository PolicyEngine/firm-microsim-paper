# Changelog of substantive corrections

Recorded so the manuscript can stay short; each entry names the issue or pull
request that diagnosed and fixed it.

| Date | Change | Where |
| --- | --- | --- |
| 2026-06 | Net VAT liability was set to value added rather than the standard rate on value added, inflating per-firm liabilities about fivefold and producing a spurious £85k excess mass (E ≈ 8,712). Corrected to `v = 0.20 (y − x)`. | #15, #21 |
| 2026-07 | The below-threshold HMRC liability total (voluntary registrants) was calibrated against the whole below-threshold population, draining weights near the threshold and, on one build, producing a large specification-robust artifact at the £90k band edge. Now an informational diagnostic. | #23, #26 |
| 2026-07 | Near-threshold £1k-band shape targets from OBR Chart C added for 2023-24, applied side-consistently as shape (not level) targets. | #23 |
| 2026-07 | Calibration dropout was not inverse-scaled, biasing every total upward by ~5%; best-loss iterate now restored. | #32, #33 |
| 2026-07 | The average-rate taper over £85k–£105k created a smoothly dominated interval; replaced by a monotone marginal-rate taper with band top T/(1−2τ). | #32, #33 |
| 2026-07 | The behavioural solver's damped fixed-point iteration had no fixed point for abilities straddling a notch; replaced by the closed-form region-confined solve. | #34 |
| 2026-09 | The 2023-24 ONS employment table was local units (Table 18), not enterprises (Table 3); population and employment targets were on incompatible universes. Two-universe calibration (ONS frame + HMRC registered subset via registration propensity); ONS tables rebuilt by an executable ETL. | #37, #46 |
| 2026-09 | Static model: one ageing convention, deregistration-threshold gap, explicit registration rule; anchor release decomposed by data-year status. | #39, #46 |
| 2026-09 | Bunching estimator reports gross and net excess and flags a censored y_R; open ONS 5000+ band drawn log-uniform; constant-50% marginal taper added. | #38, #40 |
| 2026-09 | DBT unregistered stratum added as a third universe (frozen exponential shape); OBR £1k-bin counts applied as levels on the all-business universe; closed ONS bands filled with a log-log power law (no band-edge steps); static model evaluates membership on data-year turnover with whole-band release, gap and retention as sensitivities. | #25, #50 |
