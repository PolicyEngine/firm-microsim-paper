"""Data-grounded sector VAT wedge for the iso-elastic reform costings.

Motivation
----------
The iso-elastic simulator applies each firm's *effective* VAT wedge
``tau0 = liab / turnover`` (net VAT remitted as a share of turnover, i.e. the
rate that actually bites on value added, output VAT minus input credits). In the
synthetic population that wedge is driven by per-firm input shares drawn from a
``Beta(4, 2)`` distribution with hand-set sector shifts in
:mod:`firm_microsim.generate` -- they are *not* grounded in real deductible-cost
data, and they wash out to a near-uniform ~3% wedge across the population.

This module replaces that synthetic wedge with one estimated from official
sector aggregates: HMRC net VAT by trade sector divided by an ONS-derived
estimate of sector turnover. The result is a per-sector net-VAT-to-turnover
ratio that we attach to each firm by its SIC division, and feed to
:func:`firm_microsim.dynamic.model.reform_revenue` via its ``tau0_override``
hook. We then re-run the flat reforms and compare the costs against the
firm-level (synthetic) baseline.

UNITS (read carefully -- this is where the bugs hide)
-----------------------------------------------------
* ``hmrc_vat_liability_by_sector.csv``: column ``Trade_Sector`` (int-able SIC
  division) and a year column (e.g. ``2023-24``) holding **net VAT in £m** per
  sector. (Confirmed against ``validate.py``, which compares it to the synthetic
  ``vat_liability_k * weight / 1000`` = £m.) Net VAT can be **negative** for
  net-repayment sectors (zero-rated outputs: agriculture, food, etc.).
* ``ons_firm_turnover.csv``: ``SIC Code`` (int-able SIC division), then **firm
  COUNTS** by turnover band (band labels are turnover ranges in £k), and a
  ``Total`` column that is the **firm count**, NOT a turnover value. There is no
  ready-made sector-turnover total in the processed inputs, so we estimate
  sector turnover as ``sum_band(count_band * midpoint_band)`` with band
  midpoints in £k, then convert to £m (``/1000``). The open-ended top band
  ``5000+`` (£5m+) has no upper bound; we use a £10m midpoint -- this is the one
  genuinely crude assumption and it understates turnover for a handful of
  capital-intensive / excise sectors dominated by a few very large firms.

Because that top-band approximation is unreliable for concentrated sectors
(e.g. petroleum refining comes out absurdly high), and because net-repayment
sectors give a negative wedge that has no meaning as a *rate applied to value
added*, we clip the final per-sector ratio to the plausible band
``[0, RATIO_CAP]`` with ``RATIO_CAP = 0.30`` (1.5x the 20% standard rate). The
cap binds only for a few outlier sectors; the bulk of the sector spread (0%–~14%)
survives, which is exactly the heterogeneity the uniform ~3% synthetic wedge
hides. Sectors with missing or zero estimated turnover fall back to the
population aggregate ratio (total net VAT / total estimated turnover, clipped).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from firm_microsim.config import PROCESSED_DATA_DIR, RESULTS_DIR, SYNTHETIC_DATA_DIR
from firm_microsim.dynamic.model import (
    ELASTICITIES,
    build_reforms,
    load_reform_data,
    reform_revenue,
)

# ONS turnover-band midpoints in £k (bands are turnover ranges). The open-ended
# top band "5000+" (£5m+) is assigned a £10m midpoint -- see module docstring.
_ONS_BAND_MIDPOINTS_K = {
    "0-49": 25.0,
    "50-99": 75.0,
    "100-249": 175.0,
    "250-499": 375.0,
    "500-999": 750.0,
    "1000-4999": 3000.0,
    "5000+": 10000.0,
}

# Plausibility cap on the effective wedge (rate applied to value added). 1.5x the
# 20% standard rate; net-repayment sectors are floored at 0.
RATIO_CAP = 0.30

# Three flat reforms to cost (subset of build_reforms()).
FLAT_REFORMS = ("raise100k", "rate10", "rate15")


def sector_vat_ratios(vintage: str = "2023-24") -> pd.Series:
    """Empirical net-VAT-to-turnover wedge by integer SIC division.

    Loads HMRC net VAT by sector (£m) and ONS firm counts by turnover band,
    estimates sector turnover from band counts x band midpoints (£k -> £m), and
    returns ``ratio = net_vat_m / turnover_m`` clipped to ``[0, RATIO_CAP]``.

    Returns a :class:`pandas.Series` indexed by integer SIC division. Its
    ``.attrs["population_mean"]`` holds the aggregate fallback ratio (total net
    VAT / total estimated turnover, clipped) used for missing/zero sectors.
    """
    base = PROCESSED_DATA_DIR / vintage
    liab = pd.read_csv(base / "hmrc_vat_liability_by_sector.csv")
    ons = pd.read_csv(base / "ons_firm_turnover.csv")

    # --- HMRC net VAT (£m) by SIC division. Last column is the year column.
    liab = liab[liab["Trade_Sector"] != "Total"].copy()
    liab["sic"] = liab["Trade_Sector"].astype(int)
    year_col = liab.columns[-2]  # year column sits before the appended "sic"
    liab["net_vat_m"] = liab[year_col].astype(float)

    # --- ONS estimated sector turnover (£m) from firm counts x band midpoints.
    sic_col = ons["SIC Code"]
    ons = ons[sic_col.notna() & (sic_col.astype(str).str.strip() != "")].copy()
    ons["sic"] = ons["SIC Code"].astype(int)
    turnover_k = sum(
        ons[band].fillna(0.0).astype(float) * mid
        for band, mid in _ONS_BAND_MIDPOINTS_K.items()
    )
    ons["turnover_m"] = turnover_k / 1000.0  # £k -> £m

    merged = liab[["sic", "net_vat_m"]].merge(
        ons[["sic", "turnover_m"]], on="sic", how="inner"
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(
            merged["turnover_m"] > 0,
            merged["net_vat_m"] / merged["turnover_m"],
            np.nan,
        )
    ratio = pd.Series(
        np.clip(raw, 0.0, RATIO_CAP), index=merged["sic"].to_numpy()
    )

    # Population aggregate fallback (clipped), for missing / zero-turnover SICs.
    total_vat = float(merged["net_vat_m"].sum())
    total_turnover = float(merged["turnover_m"].sum())
    pop_mean = float(np.clip(total_vat / total_turnover, 0.0, RATIO_CAP))
    ratio = ratio.dropna()
    ratio.attrs["population_mean"] = pop_mean
    return ratio


def attach_sector_tau0(
    df: pd.DataFrame, ratios: pd.Series, fallback: float | None = None
) -> np.ndarray:
    """Map each firm's ``int(sic_code)`` to its sector wedge -> per-firm array.

    Firms whose SIC division is absent from ``ratios`` receive ``fallback``
    (default: the population aggregate stored in
    ``ratios.attrs["population_mean"]``, or the mean of ``ratios`` if unset).
    """
    if fallback is None:
        fallback = ratios.attrs.get("population_mean", float(ratios.mean()))
    sic = df["sic_code"].astype(int).to_numpy()
    lookup = dict(zip(ratios.index.tolist(), ratios.to_numpy(dtype=float)))
    return np.array([lookup.get(int(s), fallback) for s in sic], dtype=float)


def _load_population_with_sic(vintage: str = "2023-24") -> pd.DataFrame:
    """Load the reform-costing population and attach ``sic_code`` (same order).

    ``load_reform_data`` deliberately does not read ``sic_code``; we read it from
    the same CSV in identical row order and attach it by position.
    """
    df = load_reform_data()
    sic = pd.read_csv(
        SYNTHETIC_DATA_DIR / f"synthetic_firms_{vintage}.csv",
        usecols=["sic_code"],
        dtype={"sic_code": str},
    )["sic_code"]
    if len(sic) != len(df):
        raise RuntimeError(
            f"sic_code rows ({len(sic)}) != population rows ({len(df)})"
        )
    df = df.reset_index(drop=True)
    df["sic_code"] = sic.reset_index(drop=True)
    return df


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def run_comparison(vintage: str = "2023-24") -> dict:
    """Compute baseline vs sector-grounded wedges and reform costs.

    Returns a dict with the wedge summary and a list of per-(reform, e) rows.
    """
    df = _load_population_with_sic(vintage)
    ratios = sector_vat_ratios(vintage)
    sector_tau0 = attach_sector_tau0(df, ratios)

    t_obs = df["turnover"].to_numpy(dtype=float)
    liab = df["liab"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        firm_tau0 = np.where(t_obs > 0, liab / t_obs, 0.0)

    pos = t_obs > 0
    wedge = {
        "baseline_mean": _weighted_mean(firm_tau0[pos], w[pos]),
        "sector_mean": _weighted_mean(sector_tau0[pos], w[pos]),
        "sector_ratio_min": float(ratios.min()),
        "sector_ratio_max": float(ratios.max()),
        "sector_ratio_median": float(ratios.median()),
        "n_sectors": int(ratios.shape[0]),
        "population_mean_fallback": float(ratios.attrs["population_mean"]),
    }

    reforms = build_reforms()
    rows = []
    for rname in FLAT_REFORMS:
        sched, label = reforms[rname]
        for e in ELASTICITIES:
            base = reform_revenue(df, sched, e, behavioural=True)
            sect = reform_revenue(
                df, sched, e, behavioural=True, tau0_override=sector_tau0
            )
            rows.append(
                {
                    "reform": rname,
                    "label": label,
                    "e": e,
                    "d_rev_baseline": base["d_rev"],
                    "d_rev_sector": sect["d_rev"],
                    "delta": sect["d_rev"] - base["d_rev"],
                }
            )
    return {"wedge": wedge, "rows": rows}


def _write_table(result: dict, path) -> None:
    w = result["wedge"]
    lines = []
    lines.append("Data-grounded sector VAT wedge: comparison")
    lines.append("=" * 64)
    lines.append("")
    lines.append("Effective wedge (net VAT / turnover, population-weighted mean)")
    lines.append(f"  firm-level synthetic baseline : {w['baseline_mean']:7.4f}")
    lines.append(f"  sector-grounded (this module) : {w['sector_mean']:7.4f}")
    lines.append(
        f"  sector ratio spread           : "
        f"min {w['sector_ratio_min']:.4f}, median {w['sector_ratio_median']:.4f}, "
        f"max {w['sector_ratio_max']:.4f}  (n={w['n_sectors']} sectors)"
    )
    lines.append(
        f"  population-mean fallback       : {w['population_mean_fallback']:.4f}"
    )
    lines.append("")
    lines.append("Reform revenue change vs £85k notch (£bn, behavioural)")
    lines.append(
        f"  {'reform':<10}{'e':>6}{'baseline':>14}{'sector':>14}{'delta':>14}"
    )
    for r in result["rows"]:
        lines.append(
            f"  {r['reform']:<10}{r['e']:>6.2f}"
            f"{r['d_rev_baseline'] / 1e9:>14.4f}"
            f"{r['d_rev_sector'] / 1e9:>14.4f}"
            f"{r['delta'] / 1e9:>14.4f}"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _write_figure(result: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = result["wedge"]
    rows = result["rows"]
    e_head = 0.17
    head = [r for r in rows if abs(r["e"] - e_head) < 1e-9]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel 0: mean effective wedge (baseline vs sector), with sector spread.
    ax0.bar(
        ["firm-level\nsynthetic", "sector-\ngrounded"],
        [w["baseline_mean"], w["sector_mean"]],
        color=["#9aa0a6", "#1a73e8"],
    )
    ax0.errorbar(
        1,
        w["sector_mean"],
        yerr=[[w["sector_mean"] - w["sector_ratio_min"]],
              [w["sector_ratio_max"] - w["sector_mean"]]],
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    ax0.set_ylabel("effective VAT wedge (net VAT / turnover)")
    ax0.set_title("Mean effective wedge\n(bar = pop. mean; whisker = sector min/max)")

    # Panel 1: reform cost (£bn) baseline vs sector at headline e=0.17.
    labels = [r["reform"] for r in head]
    x = np.arange(len(labels))
    bw = 0.38
    ax1.bar(x - bw / 2, [r["d_rev_baseline"] / 1e9 for r in head], bw,
            label="firm-level baseline", color="#9aa0a6")
    ax1.bar(x + bw / 2, [r["d_rev_sector"] / 1e9 for r in head], bw,
            label="sector-grounded", color="#1a73e8")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("revenue change vs £85k notch (£bn)")
    ax1.set_title(f"Reform cost (behavioural, e={e_head})")
    ax1.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(vintage: str = "2023-24") -> dict:
    """Run the full comparison and write the txt table + png figure."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_comparison(vintage)

    txt_path = RESULTS_DIR / "sector_vat_comparison.txt"
    png_path = RESULTS_DIR / "sector_vat_comparison.png"
    _write_table(result, txt_path)
    _write_figure(result, png_path)

    w = result["wedge"]
    print(
        f"mean effective wedge: baseline={w['baseline_mean']:.4f}  "
        f"sector-grounded={w['sector_mean']:.4f}  "
        f"(sector spread {w['sector_ratio_min']:.4f}..{w['sector_ratio_max']:.4f}, "
        f"n={w['n_sectors']})"
    )
    for r in result["rows"]:
        print(
            f"  {r['reform']:<10} e={r['e']:.2f}  "
            f"baseline={r['d_rev_baseline'] / 1e9:+.3f}bn  "
            f"sector={r['d_rev_sector'] / 1e9:+.3f}bn  "
            f"delta={r['delta'] / 1e9:+.3f}bn"
        )
    print(f"wrote {txt_path}")
    print(f"wrote {png_path}")
    return result


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare iso-elastic VAT reform costs under the synthetic firm-level "
            "effective wedge vs a data-grounded sector net-VAT-to-turnover wedge."
        )
    )
    parser.add_argument(
        "--vintage", default="2023-24", help="data vintage (default: 2023-24)"
    )
    args = parser.parse_args()
    main(args.vintage)


if __name__ == "__main__":  # pragma: no cover
    cli()
