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

from firm_microsim.config import SYNTHETIC_DATA_DIR, VINTAGES

# Fiscal-year ageing factors (cumulative nominal growth from the 2023-24 data
# year) plus the April-2024 anchor reform: a frozen £85k baseline, RPI-uprated
# with a two-year lag, against the £90k policy, alongside HMRC's published
# costing (£m). Both turnover and liability are aged by the same factor against
# fixed nominal thresholds (the fiscal-drag convention); see ``_growth``. The
# factors are an ASSUMED nominal-turnover path (about 3.1%, 2.0%, 2.5%, 3.0%,
# 2.9% a year), of the order of the OBR's March 2024 nominal GDP forecast; they
# are not taken from a published series and are reported as assumptions in the
# paper (issue #47). The HMRC costing profile is from the TIIN "Increasing the
# VAT registration threshold" (2024): 28,000 fewer registrants in 2024-25 and
# 14,000 fewer on average over 2024-25 to 2028-29.
FISCAL_YEARS = [
    {"year": "2024-25", "baseline": 85000, "policy": 90000, "hmrc_impact": -150, "firm_growth": 1.0310},
    {"year": "2025-26", "baseline": 85000, "policy": 90000, "hmrc_impact": -185, "firm_growth": 1.0516},
    {"year": "2026-27", "baseline": 87000, "policy": 90000, "hmrc_impact": -125, "firm_growth": 1.0779},
    {"year": "2027-28", "baseline": 89000, "policy": 90000, "hmrc_impact": -50, "firm_growth": 1.1102},
    {"year": "2028-29", "baseline": 92000, "policy": 90000, "hmrc_impact": 65, "firm_growth": 1.1424},
]
# Cumulative factor already embodied in each vintage's data year, so a
# 2024-25 build is not aged by the 2023-24 -> 2024-25 step a second time.
VINTAGE_BASE_GROWTH = {"2023-24": 1.0, "2024-25": 1.0310}

# The deregistration threshold sits £2,000 below the registration threshold
# (£83k/£85k to March 2024; £88k/£90k from April 2024). A registered firm may
# deregister only if its turnover falls below the deregistration threshold, so
# when the threshold rises the mechanically "released" band is
# [old threshold, new threshold - gap): firms in the top £2,000 of the raised
# band stay registered.
DEREGISTRATION_GAP = 2_000.0

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
        if vintage not in VINTAGES:
            raise ValueError(f"Unknown vintage {vintage!r}; choose from {sorted(VINTAGES)}")
        self.data_threshold = float(VINTAGES[vintage]["threshold"]) * 1000.0
        path = SYNTHETIC_DATA_DIR / f"synthetic_firms_{vintage}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — generate it first with:\n"
                f"  python -m firm_microsim --vintage {vintage} "
                f"--output synthetic_firms_{vintage}.csv"
            )
        cols = ["annual_turnover_k", "vat_liability_k", "weight", "vat_scope", "vat_registered"]
        self.firms = pd.read_csv(path, usecols=cols)
        # Baseline voluntary registrants: registered with data-year turnover at
        # or below the data-year threshold. Their status is held fixed under
        # every counterfactual threshold (registration is not modelled).
        t_k = self.firms["annual_turnover_k"]
        registered = self.firms["vat_registered"].astype(bool)
        self.firms["voluntary"] = registered & (t_k * 1000.0 <= self.data_threshold)
        # Mandatory registrants at the data-year threshold: the population that
        # a threshold rise can release, subject to the deregistration threshold.
        self.firms["mandatory"] = registered & (t_k * 1000.0 > self.data_threshold)

    # -- core mechanics ----------------------------------------------------
    def _growth(self, year: str) -> float:
        """Nominal-growth factor from this vintage's data year to ``year``."""
        return _fiscal_year(year)["firm_growth"] / VINTAGE_BASE_GROWTH[self.vintage]

    def _aged(self, growth: float) -> pd.DataFrame:
        """Return turnover (£) and net VAT liability (£), both aged by ``growth``.

        Turnover and liability grow together against fixed nominal thresholds
        (the fiscal-drag convention the paper's institutional section relies
        on), so band membership in a later fiscal year is evaluated on aged
        turnover.
        """
        df = self.firms
        return pd.DataFrame(
            {
                "turnover": df["annual_turnover_k"] * 1000.0 * growth,
                "liab": df["vat_liability_k"] * 1000.0 * growth,
                "weight": df["weight"],
                "scope": df["vat_scope"].astype(bool),
                "voluntary": df["voluntary"].astype(bool),
                "mandatory": df["mandatory"].astype(bool),
            }
        )

    def _registered(
        self,
        df: pd.DataFrame,
        threshold: float,
        gap: float = DEREGISTRATION_GAP,
        retain_voluntary: bool = False,
    ) -> pd.Series:
        """Registration under a counterfactual ``threshold`` (£).

        With aged turnover ``y`` and the data-year threshold ``T0``, an
        in-scope firm is registered if

        * ``y >= threshold`` (required under the counterfactual); or
        * it was registered at the data year and ``y >= threshold - gap``
          (cannot deregister: turnover is above the deregistration
          threshold); or
        * it is a baseline voluntary registrant with ``y < T0`` (never
          required under either regime, so its status is unchanged); or
        * ``retain_voluntary`` and it is a baseline voluntary registrant
          (fixed-preference convention: a firm that chose registration below
          ``T0`` keeps it wherever the threshold moves).

        Out-of-scope (PAYE-only / exempt-sector) enterprises never remit. At
        ``threshold == T0`` the rule reproduces the baseline registered set.
        Voluntary registrants that ageing carries into a band whose
        requirement the move removes are therefore RELEASED by default; the
        ONS frame's high below-threshold registration share (about 89%)
        partly reflects that small firms enter the frame through VAT
        registration, so treating it as revealed preference
        (``retain_voluntary=True``) is reported as a sensitivity, not the
        headline.
        """
        y = df["turnover"]
        registered0 = df["mandatory"] | df["voluntary"]
        reg = (y >= threshold) | (registered0 & (y >= threshold - gap))
        reg = reg | (df["voluntary"] & (y < self.data_threshold))
        if retain_voluntary:
            reg = reg | df["voluntary"]
        return df["scope"] & reg

    def _revenue(self, df: pd.DataFrame, threshold: float, **kw) -> float:
        """Total weighted net VAT (£) from firms registered at ``threshold``."""
        registered = self._registered(df, threshold, **kw)
        return float((df["liab"].where(registered, 0.0) * df["weight"]).sum())

    @staticmethod
    def _mandatory_base(df: pd.DataFrame, threshold: float) -> float:
        """Weighted net VAT (£) of in-scope firms at or above ``threshold``.

        Excludes below-threshold voluntary remittances, whose model liability
        is not calibrated, so the figure is comparable to HMRC's
        above-threshold liability bands.
        """
        mask = df["scope"] & (df["turnover"] >= threshold)
        return float((df["liab"].where(mask, 0.0) * df["weight"]).sum())

    def _vat_paying_firms(self, df: pd.DataFrame, threshold: float, **kw) -> float:
        """Weighted count of VAT-paying firms (registered & net-positive)."""
        mask = self._registered(df, threshold, **kw) & (df["liab"] > 0)
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
        # Polynomial extrapolation can become negative below the fit range;
        # neither a firm count nor aggregate net liability used for a costing
        # can be negative.
        cf_firms = np.clip(cf_firms, 0.0, None)
        cf_liab = np.clip(cf_liab, 0.0, None)
        return centres, cf_firms, cf_liab

    # -- public results ----------------------------------------------------
    def threshold_sweep(
        self,
        year: str = "2025-26",
        thresholds: list | None = None,
        baseline: int = POLICY_THRESHOLD,
    ) -> pd.DataFrame:
        """Revenue (£m) and VAT-paying-firm (000s) changes vs ``baseline``.

        Each row costs moving the threshold to a new location. Lowering the
        threshold draws in-scope, not-yet-registered firms in (positive);
        raising it releases in-scope registered firms (negative). Turnover and
        liability are aged to the fiscal year by the same nominal-growth
        factor, so membership is evaluated on aged turnover.
        """
        thresholds = thresholds or SWEEP_THRESHOLDS
        df = self._aged(self._growth(year))
        base_revenue = self._revenue(df, baseline)
        base_firms = self._vat_paying_firms(df, baseline)

        rows = []
        for t in thresholds:
            rows.append(
                {
                    "threshold_k": t / 1000.0,
                    "revenue_change_m": (self._revenue(df, t) - base_revenue) / 1e6,
                    "firms_change_k": (self._vat_paying_firms(df, t) - base_firms) / 1000.0,
                }
            )
        return pd.DataFrame(rows)

    def anchor_reform(
        self,
        gap: float = DEREGISTRATION_GAP,
        retention: float = 0.0,
        retain_voluntary: bool = False,
    ) -> pd.DataFrame:
        """£85k→£90k anchor-reform impact (£m) per year: model vs HMRC.

        ``gap`` is the registration-minus-deregistration threshold distance
        (£2,000 by statute; pass 0 for the naive whole-band release).
        ``retention`` scales the released liability by ``1 - retention`` (a
        share of released firms assumed to stay registered voluntarily, e.g.
        the Liu et al. 43%). ``retain_voluntary`` applies the fixed-preference
        convention of :meth:`_registered`.

        Simple band-sum on the loaded vintage. Use the £85k (2023-24) vintage —
        the basis HMRC actually had at the 6 March 2024 costing (the threshold
        was still £85k until 1 April 2024). There the affected [baseline, £90k)
        firms sit ABOVE the £85k registration threshold, so the band is cleanly
        populated with in-scope registered firms. Each year's impact is the
        revenue under the £90k policy minus revenue under that year's
        counterfactual baseline threshold, both evaluated on turnover and
        liability aged to the year with the same registration rule.
        When fiscal drag lifts the baseline above £90k (2028-29) the band flips
        and the reform adds firms (a revenue gain).
        """
        rows = []
        for fy in FISCAL_YEARS:
            df = self._aged(self._growth(fy["year"]))
            base_t, pol_t = float(fy["baseline"]), float(fy["policy"])
            # Revenue under each regime from the same registration rule:
            # in-scope firms at/above the regime's threshold, data-year
            # registrants retained down to its deregistration threshold,
            # voluntary registrants throughout. Differencing the two regimes
            # handles both the release years and the 2028-29 sign flip.
            liab_w = df["liab"] * df["weight"]
            kw = dict(gap=gap, retain_voluntary=retain_voluntary)
            r_policy = float(liab_w[self._registered(df, pol_t, **kw)].sum())
            r_base = float(liab_w[self._registered(df, base_t, **kw)].sum())
            pe_impact = (r_policy - r_base) / 1e6
            if pe_impact < 0:
                pe_impact *= 1.0 - retention
            rows.append(
                {
                    "year": fy["year"],
                    "hmrc_impact_m": float(fy["hmrc_impact"]),
                    "policyengine_impact_m": round(pe_impact, 1),
                }
            )
        return pd.DataFrame(rows)

    def total_revenue_bn(self, year: str = "2025-26", threshold: int = POLICY_THRESHOLD) -> float:
        """In-scope above-threshold VAT base (£bn) at ``threshold`` in ``year``.

        Comparable to HMRC's above-threshold liability bands; excludes the
        uncalibrated below-threshold voluntary remittances.
        """
        df = self._aged(self._growth(year))
        return self._mandatory_base(df, threshold) / 1e9
