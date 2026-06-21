"""Static VAT-threshold revenue model — mechanical, no behavioural response.

Ports the static-costing core of the original `static_revenue.py` /
`vat_threshold_2025_26.py`. Firm turnover is held fixed; only registration
status flips when the threshold moves, so revenue and firm-count changes are
purely the reclassification of firms into / out of the VAT net.

The model treats the calibrated per-firm ``vat_liability_k`` as the net VAT
contribution (the synthetic weights are calibrated so the weighted sum
reproduces HMRC's net VAT-liability totals), and ages the microdata to a given
fiscal year with a cumulative nominal-growth factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm_microsim.config import SYNTHETIC_DATA_DIR

# Fiscal-year aging factors (cumulative from the base-year microdata) plus the
# April-2024 anchor reform: a frozen £85k baseline, RPI-uprated with a two-year
# lag, against the £90k policy, alongside HMRC's published costing (£m).
FISCAL_YEARS = [
    {"year": "2024-25", "baseline": 85000, "policy": 90000, "hmrc_impact": -150, "firm_growth": 1.0310},
    {"year": "2025-26", "baseline": 85000, "policy": 90000, "hmrc_impact": -185, "firm_growth": 1.0516},
    {"year": "2026-27", "baseline": 87000, "policy": 90000, "hmrc_impact": -125, "firm_growth": 1.0779},
    {"year": "2027-28", "baseline": 89000, "policy": 90000, "hmrc_impact": -50, "firm_growth": 1.1102},
    {"year": "2028-29", "baseline": 92000, "policy": 90000, "hmrc_impact": 65, "firm_growth": 1.1424},
]

# Current statutory threshold (£) — the baseline for the threshold sweep.
POLICY_THRESHOLD = 90000

# Threshold sweep grid: £70k … £120k in £5k steps (matches the paper).
SWEEP_THRESHOLDS = list(range(70000, 120001, 5000))


def _fiscal_year(year: str) -> dict:
    """Look up a fiscal-year record by its label (e.g. ``"2025-26"``)."""
    try:
        return next(fy for fy in FISCAL_YEARS if fy["year"] == year)
    except StopIteration as exc:  # pragma: no cover - guard
        years = ", ".join(fy["year"] for fy in FISCAL_YEARS)
        raise ValueError(f"Unknown fiscal year {year!r}; choose from {years}") from exc


class StaticVATModel:
    """Static threshold-reform costing over the synthetic firm population."""

    def __init__(self, vintage: str = "2024-25") -> None:
        """Load the synthetic population for ``vintage`` (default the £90k year)."""
        self.vintage = vintage
        path = SYNTHETIC_DATA_DIR / f"synthetic_firms_{vintage}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — generate it first with:\n"
                f"  python -m firm_microsim --vintage {vintage} "
                f"--output synthetic_firms_{vintage}.csv"
            )
        self.firms = pd.read_csv(
            path, usecols=["annual_turnover_k", "vat_liability_k", "weight"]
        )

    # -- core mechanics ----------------------------------------------------
    def _aged(self, growth: float) -> pd.DataFrame:
        """Return turnover (£) and net VAT liability (£), aged by ``growth``."""
        df = self.firms
        return pd.DataFrame(
            {
                "turnover": df["annual_turnover_k"] * 1000.0 * growth,
                "liab": df["vat_liability_k"] * 1000.0 * growth,
                "weight": df["weight"],
            }
        )

    @staticmethod
    def _revenue(df: pd.DataFrame, threshold: float) -> float:
        """Total weighted net VAT (£) from firms registered at ``threshold``."""
        registered = df["turnover"] >= threshold
        return float((df["liab"].where(registered, 0.0) * df["weight"]).sum())

    @staticmethod
    def _vat_paying_firms(df: pd.DataFrame, threshold: float) -> float:
        """Weighted count of VAT-paying firms (registered & net-positive)."""
        mask = (df["turnover"] >= threshold) & (df["liab"] > 0)
        return float(df.loc[mask, "weight"].sum())

    # -- smooth counterfactual density ------------------------------------
    # The synthetic population carries a registration step at the threshold:
    # the VAT-liability calibration concentrates firm weight on high-liability
    # firms below the threshold, so the observed below-threshold density and
    # per-firm liability are inflated (a behavioural/bunching + calibration
    # artifact). The above-threshold region is clean. For the *static*
    # counterfactual we therefore fit the smooth above-threshold profile of
    # per-£1k-bin VAT-paying firms and liability and extrapolate it across the
    # threshold, matching the paper's mechanical sweep on a smooth distribution.
    def _counterfactual_bins(
        self,
        baseline: float = POLICY_THRESHOLD,
        bin_k: float = 1.0,
        lo_k: float = 60.0,
        hi_k: float = 160.0,
        fit_pad_k: float = 2.0,
        fit_top_k: float = 150.0,
        degree: int = 1,
    ):
        """Return (£k bin centres, cf VAT-paying firms, cf liability £/bin).

        Computed on the UNAGED population so the registration step sits cleanly
        at the threshold. ``firms`` and ``liability`` per bin are fitted on the
        clean above-threshold region ``[baseline + fit_pad_k, fit_top_k]`` and
        evaluated over the whole ``[lo_k, hi_k]`` range, so the below-threshold
        inflation is replaced by a smooth extrapolation of the above-threshold
        trend. Callers scale liability to the fiscal year separately.
        """
        tk = self.firms["annual_turnover_k"].to_numpy()
        liab = self.firms["vat_liability_k"].to_numpy() * 1000.0  # £
        w = self.firms["weight"].to_numpy()
        edges = np.arange(lo_k, hi_k + bin_k, bin_k)
        centres = (edges[:-1] + edges[1:]) / 2.0

        paying = liab > 0
        firms, _ = np.histogram(tk[paying], bins=edges, weights=w[paying])
        liab_bin, _ = np.histogram(tk, bins=edges, weights=liab * w)

        base_k = baseline / 1000.0
        fit = (centres >= base_k + fit_pad_k) & (centres <= fit_top_k)
        cf_firms = np.polyval(np.polyfit(centres[fit], firms[fit], degree), centres)
        cf_liab = np.polyval(np.polyfit(centres[fit], liab_bin[fit], degree), centres)
        # clamp tiny negatives from extrapolation
        cf_firms = np.clip(cf_firms, 0.0, None)
        return centres, cf_firms, cf_liab

    # -- public results ----------------------------------------------------
    def threshold_sweep(
        self,
        year: str = "2025-26",
        thresholds: list | None = None,
        baseline: int = POLICY_THRESHOLD,
    ) -> pd.DataFrame:
        """Revenue (£m) and VAT-paying-firm (000s) changes vs ``baseline``.

        Each row costs moving the threshold to a new location, holding turnover
        fixed. Lowering the threshold draws firms in (positive); raising it
        loses them (negative). Liability is scaled to the fiscal year by the
        nominal-growth factor; firm counts are year-invariant.
        """
        thresholds = thresholds or SWEEP_THRESHOLDS
        growth = _fiscal_year(year)["firm_growth"]
        centres, cf_firms, cf_liab = self._counterfactual_bins(baseline)
        cf_liab = cf_liab * growth
        base_k = baseline / 1000.0

        rows = []
        for t in thresholds:
            t_k = t / 1000.0
            lo_k, hi_k = sorted((t_k, base_k))
            # bins reclassified between the new threshold and the baseline
            band = (centres >= lo_k) & (centres < hi_k)
            sign = 1.0 if t_k < base_k else -1.0  # lowering adds firms/revenue
            rows.append(
                {
                    "threshold_k": t_k,
                    "revenue_change_m": sign * cf_liab[band].sum() / 1e6,
                    "firms_change_k": sign * cf_firms[band].sum() / 1000.0,
                }
            )
        return pd.DataFrame(rows)

    def anchor_reform(self) -> pd.DataFrame:
        """£85k→£90k anchor-reform impact (£m) per year: model vs HMRC.

        For each fiscal year, the affected firms lie in the band between the
        (uprated, frozen) baseline and the £90k policy. When baseline < policy
        the reform removes them from the net (revenue loss, negative); when
        fiscal drag lifts the baseline above £90k it adds them (gain, positive).
        """
        centres, _cf_firms, cf_liab = self._counterfactual_bins(POLICY_THRESHOLD)
        rows = []
        for fy in FISCAL_YEARS:
            lo_k, hi_k = sorted((fy["baseline"] / 1000.0, fy["policy"] / 1000.0))
            band = (centres >= lo_k) & (centres < hi_k)
            mass_m = float((cf_liab[band] * fy["firm_growth"]).sum()) / 1e6
            pe_impact = mass_m if fy["baseline"] > fy["policy"] else -mass_m
            rows.append(
                {
                    "year": fy["year"],
                    "hmrc_impact_m": float(fy["hmrc_impact"]),
                    "policyengine_impact_m": round(pe_impact, 1),
                }
            )
        return pd.DataFrame(rows)

    def total_revenue_bn(self, year: str = "2025-26", threshold: int = POLICY_THRESHOLD) -> float:
        """Total VAT revenue (£bn) at ``threshold`` in ``year`` (sanity check)."""
        df = self._aged(_fiscal_year(year)["firm_growth"])
        return self._revenue(df, threshold) / 1e9
