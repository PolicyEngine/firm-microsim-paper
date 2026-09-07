#!/usr/bin/env python3
"""Claim manifest: every headline number the manuscript quotes, traced to its
artifact (issue #40).

``python scripts/claims.py``          rewrites ``results/claims.json`` from the
                                      checked ``results/*.txt`` artifacts.
``python scripts/claims.py --check``  additionally asserts that each claim's
                                      manuscript rendering occurs in the named
                                      LaTeX source, exiting 1 on any miss.

Each claim records the artifact file, the regex that extracts the value, the
extracted value, the rendering the manuscript must contain, and the .tex file
that must contain it. ``tests/test_claims_manifest.py`` runs the check in CI.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PAPER = REPO / "paper"
OUT = RESULTS / "claims.json"


def _read(name: str) -> str:
    return (RESULTS / name).read_text()


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _tex_int(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def _series(vals: list[float]) -> str:
    return ", ".join(("$+$" if v >= 0 else "$-$") + f"{abs(v):.0f}" for v in vals)


def build_claims() -> list[dict]:
    C: list[dict] = []

    def add(key, artifact, pattern, value, tex, tex_file, note=""):
        C.append({
            "key": key, "artifact": artifact, "pattern": pattern,
            "value": value, "tex": tex, "tex_file": tex_file, "note": note,
        })

    # --- calibration -----------------------------------------------------
    cal = _read("calibration_accuracy.txt")
    for v, tag in (("2023-24", "2324"), ("2024-25", "2425")):
        block = cal.split(f"Vintage {v}")[1].split("Vintage")[0]
        m = re.search(r"Overall \(5 calibrated dims\)\s+([\d.]+)%", block)
        add(f"overall_{tag}", "calibration_accuracy.txt", m.re.pattern, m.group(1),
            m.group(1) + r"\%", "Sections/data.tex")
        m = re.search(r"effective sample size: ([\d,]+)", block)
        add(f"ess_{tag}", "calibration_accuracy.txt", m.re.pattern, m.group(1),
            m.group(1).replace(",", "{,}"), "Sections/data.tex")
        m = re.search(r"max: [\d.]+ / [\d.]+ / [\d.]+ / [\d.]+ / ([\d.]+)", block)
        add(f"maxw_{tag}", "calibration_accuracy.txt", m.re.pattern, m.group(1),
            _tex_int(_num(m.group(1))), "Sections/data.tex")
        for lab, key in (("Sector Distribution", "sector"), ("VAT Liability by Band", "liab"),
                         ("ONS Population", "pop"), ("Employment Bands", "emp"),
                         ("VAT Liability below Thresh.", "liab_below")):
            m = re.search(re.escape(lab) + r"\s+([\d.]+)%", block)
            add(f"{key}_{tag}", "calibration_accuracy.txt", m.re.pattern, m.group(1),
                m.group(1) + r"\%", "Appendix/a_data.tex")

    # --- static sweep / anchor -------------------------------------------
    sw = _read("static_sweep.txt")
    anchor = [float(a[2]) for a in re.findall(r"^(20\d\d-\d\d)\s+(-?[\d.]+)\s+(-?[\d.]+)$", sw, re.M)]
    add("anchor_series", "static_sweep.txt", "anchor table", anchor, _series(anchor), "Sections/static.tex")
    ret = [float(x) for x in re.findall(r"retention-adjusted ([+-][\d.]+)m", sw)]
    add("anchor_retention_series", "static_sweep.txt", "retention-adjusted", ret, _series(ret), "Sections/static.tex")
    gapd = [float(x) for x in re.findall(r"gap-protected release ([+-][\d.]+)m", sw)]
    add("anchor_gap_series", "static_sweep.txt", "gap-protected release", gapd, _series(gapd), "Sections/static.tex")
    add("anchor_2526_abs", "static_sweep.txt", "anchor 2025-26", anchor[1],
        f"$-\\pounds{abs(anchor[1]):.0f}$m", "Sections/conclusion.tex")
    add("anchor_ret_2526_abs", "static_sweep.txt", "retention 2025-26", ret[1],
        f"$-\\pounds{abs(ret[1]):.0f}$m", "Sections/conclusion.tex")
    m = re.search(r"Total VAT revenue at GBP 90k, 2025-26: ([\d.]+)bn", sw)
    add("base_2526", "static_sweep.txt", m.re.pattern, m.group(1), f"\\pounds{m.group(1)}bn", "Sections/static.tex")
    m = re.search(r"2023-24: in-scope base .*?= ([\d.]+)bn", sw)
    add("ubase_2324", "static_sweep.txt", m.re.pattern, m.group(1), f"\\pounds{m.group(1)}bn", "Sections/static.tex")
    rows = re.findall(r"^\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)$", sw.split("Threshold sweep")[1], re.M)
    for t, rev, firms in rows:
        if float(t) == 90.0:
            continue
        sg = "+" if float(rev) > 0 else "-"
        add(f"sweep_{int(float(t))}k", "static_sweep.txt", "sweep table", float(rev),
            f"${sg}{abs(float(rev)):,.1f}$".replace(",", "{,}"), "Sections/static.tex")

    # --- reform menu ------------------------------------------------------
    menu = _read("reform_menu_common_base.txt")
    for key, label in (("raise100k", "Raise threshold to GBP100,000"), ("taper", "Graduated taper [85k,141.7k]"),
                       ("taper_flat50", "Flat 50% marginal taper [85k,141.7k]"),
                       ("rate10", "Reduced rate 10% [85k,105k]"), ("rate15", "Reduced rate 15% [85k,105k]")):
        m = re.search(re.escape(label) + r"\s+\S.*?(-?\d+)\s*$", menu, re.M)
        val = int(m.group(1))
        add(f"menu_{key}", "reform_menu_common_base.txt", label, val,
            f"$-{_tex_int(abs(val))}$", "Sections/static.tex")

    m = re.search(r"\(A\) DIRECT band-sum \[85k,100k\) :\s+(-?[\d.]+) m\s+firms (-?[\d.]+)", menu)
    add("rel100_text", "reform_menu_common_base.txt", "direct band-sum", m.group(1),
        f"\\(-\\)\\pounds{abs(float(m.group(1))):.0f}m", "Sections/static.tex")
    add("rel100_firms_text", "reform_menu_common_base.txt", "direct band-sum firms", m.group(2),
        _tex_int(round(abs(float(m.group(2))) * 1000, -2)), "Sections/static.tex")

    # --- dynamic -----------------------------------------------------------
    dyn = _read("dynamic_reform_results.txt")
    for key, label in (("rate10", "Reduced rate 10%"), ("rate15", "Reduced rate 15%")):
        ln = [line for line in dyn.splitlines() if line.strip().startswith(label)][0]
        vals = [int(x) for x in re.findall(r"£\s*([+-]?\d+)m", ln)]
        add(f"dyn_{key}_e017", "dynamic_reform_results.txt", label, vals[2],
            f"$-\\pounds{abs(vals[2])}$m", "Sections/behavioural.tex")
    for key, label in (("raise100k", "Raise threshold to £100k"), ("rate10", "Reduced rate 10%")):
        ln = [line for line in dyn.splitlines() if line.strip().startswith(label)][0]
        firms = re.findall(r"m\s+([\d,]+)\s+[+-]", ln)[0]
        add(f"dyn_{key}_firms", "dynamic_reform_results.txt", label, firms,
            "$" + firms.replace(",", "{,}") + "$", "Sections/behavioural.tex")
    for key, label in (("sector_diag_2324", "2023-24"), ("sector_diag_2425", "2024-25")):
        block = cal.split(f"Vintage {label}")[1].split("Vintage")[0]
        m = re.search(r"VAT Liability by Sector\s+([\d.]+)%", block)
        add(key, "calibration_accuracy.txt", m.re.pattern, m.group(1), m.group(1) + r"\%", "Appendix/a_data.tex")
    m = re.search(r"INPUT-VAT formulation.*?e=0\.17\s+n_H\(delta=0\.0\)=£\s*([\d.]+)k\s+n_H\(delta=0\.2\)=£\s*([\d.]+)k\s+n_H\(delta=0\.4\)=£\s*([\d.]+)k", dyn, re.S)
    for i, key in ((1, "nH_iv_e017_d0"), (2, "nH_iv_e017_d02"), (3, "nH_iv_e017_d04")):
        add(key, "dynamic_reform_results.txt", m.re.pattern, m.group(i),
            f"\\pounds{_num(m.group(i))*1000:,.0f}".replace(",", "{,}"), "Sections/behavioural.tex")

    # --- bunching -----------------------------------------------------------
    bun = _read("bunching_inference.txt")
    for v, tag in (("2023-24", "2324"), ("2024-25", "2425")):
        m = re.search(rf"Vintage {v}.*?E = ([\d,]+) \(gross\).*?b_llat = ([\d.]+)", bun, re.S)
        add(f"E_{tag}", "bunching_inference.txt", "headline E", m.group(1),
            m.group(1).replace(",", "{,}"), "Sections/bunching.tex")
        if tag == "2324":
            add("E_2324_summary", "bunching_inference.txt", "headline E", m.group(1),
                m.group(1).replace(",", "{,}"), "Sections/conclusion.tex")
            add("bllat_2324_summary", "bunching_inference.txt", "headline b_llat", m.group(2),
                f"{float(m.group(2)):.3f}", "Sections/conclusion.tex")
        add(f"bllat_{tag}", "bunching_inference.txt", "headline b_llat", m.group(2),
            f"{float(m.group(2)):.3f}", "Sections/bunching.tex")

    # --- dominated region ---------------------------------------------------
    dom = _read("dominated_region_mass.txt")
    m = re.search(r"20% \(baseline notch\)\s+[\d,]+\s+\[[^\]]+\)\s+([\d,]+)", dom)
    base = _num(m.group(1))
    add("dominated_base_obs", "dominated_region_mass.txt", m.re.pattern, base,
        _tex_int(round(base, -2)), "Sections/model.tex")

    # --- dominated region under input VAT --------------------------------------
    m = re.search(r"At delta_s = s\* delta: share of near-threshold firms with a > 0 \.\. ([\d.]+)", dom)
    add("a_delta_share_pos", "dominated_region_mass.txt", m.re.pattern, m.group(1),
        f"{float(m.group(1))*100:.0f}\\%", "Sections/model.tex")
    m = re.search(r"Standard-rated share of inputs s\* reconciling the two: ([\d.]+)", dom)
    add("s_star", "dominated_region_mass.txt", m.re.pattern, m.group(1),
        f"s^{{*}}={float(m.group(1)):.2f}", "Sections/model.tex")
    m = re.search(r"preferring registration with delta_s = delta \(all inputs standard-rated\): ([\d.]+)", dom)
    add("implied_voluntary_share_s1", "dominated_region_mass.txt", m.re.pattern, m.group(1),
        f"{float(m.group(1))*100:.0f}\\%", "Sections/model.tex")
    block = dom.split("At delta_s = s* delta")[1]
    m = re.search(r"mean width among firms with a positive width \.+ GBP ([\d,]+)", block)
    add("a_delta_mean_pos", "dominated_region_mass.txt", m.re.pattern, m.group(1),
        "\\pounds" + m.group(1).replace(",", "{,}"), "Sections/model.tex")
    m = re.search(r"mean width across near-threshold firms \.+ GBP ([\d,]+)", block)
    add("a_delta_mean_all", "dominated_region_mass.txt", m.re.pattern, m.group(1),
        "\\pounds" + m.group(1).replace(",", "{,}"), "Sections/model.tex")
    m = re.search(r"in-scope firms inside their own region \.+ ([\d,]+)", block)
    add("a_delta_own_count", "dominated_region_mass.txt", m.re.pattern, m.group(1),
        _tex_int(round(_num(m.group(1)), -3)), "Sections/model.tex")
    m = re.search(r"share of near-threshold firms with a\(delta_s\) = 0: ([\d.]+)", dom)
    add("a_delta_share_created", "dominated_region_mass.txt", m.re.pattern, m.group(1),
        f"{float(m.group(1))*100:.0f}\\%", "Sections/model.tex")
    m = re.search(r"Constant marginal rate m\* = ([\d.]+) costs the same as the raise to GBP100k \(([-+\d.]+) m\): band top U\(m\*\) = GBP ([\d,]+), affected firms ([\d.]+)", menu)
    add("taper_match_m", "reform_menu_common_base.txt", m.re.pattern, m.group(1), f"{float(m.group(1))*100:.1f}\\%", "Sections/static.tex")
    add("taper_match_top", "reform_menu_common_base.txt", m.re.pattern, m.group(3), "\\pounds" + f"{round(_num(m.group(3)), -2):,.0f}".replace(",", "{,}"), "Sections/static.tex")
    add("taper_match_firms", "reform_menu_common_base.txt", m.re.pattern, m.group(4), _tex_int(_num(m.group(4)) * 1000), "Sections/static.tex")
    m = re.search(r"\[OBR-chart universe[^\]]*\]\] E = ([\d,]+) \(gross\) \| E_net = ([-\d,]+) \| Delta_R = ([\d,]+) \| b_llat = ([\d.]+) \| b = [-\d.]+ \| y_R = ([\d.]+)", bun)
    add("chart_E_net_2324", "bunching_inference.txt", m.re.pattern, m.group(2), m.group(2).replace(",", "{,}"), "Sections/bunching.tex")
    add("chart_DR_2324", "bunching_inference.txt", m.re.pattern, m.group(3), m.group(3).replace(",", "{,}"), "Sections/bunching.tex")
    add("chart_yR_2324", "bunching_inference.txt", m.re.pattern, m.group(5), f"\\pounds{round(float(m.group(5))*1000, -2):,.0f}".replace(",", "{,}"), "Sections/bunching.tex")
    m = re.search(r"\[in-scope VAT firms\] E = ([\d,]+)", bun)
    add("scope_E_2324", "bunching_inference.txt", m.re.pattern, m.group(1), m.group(1).replace(",", "{,}"), "Sections/bunching.tex")
    sens = _read("stratum_shape_sensitivity.txt")
    for lab, key in (("exponential (headline)", "exp"), ("lognormal (sigma=1.0)", "logn")):
        blk = sens.split(f"[{lab}]")[1]
        m = re.search(r"above GBP85k \(exempt, out of scope\): ([\d,]+)", blk)
        add(f"stratum_above_{key}", "stratum_shape_sensitivity.txt", m.re.pattern, m.group(1), _tex_int(round(_num(m.group(1)), -3)), "Appendix/a_data.tex")
        m = re.search(r"cut to 70k from 85k \(2025-26\): \+([\d,.]+) m, \+([\d.]+)k firms", blk)
        add(f"cut70_{key}", "stratum_shape_sensitivity.txt", m.re.pattern, m.group(1), f"\\pounds{_num(m.group(1)):,.0f}m".replace(",", "{,}"), "Appendix/a_data.tex")
    aged = [float(x) for x in re.findall(r"aged-membership ([+-][\d.]+)m", sw)]
    add("anchor_aged_series", "static_sweep.txt", "aged-membership", aged, _series(aged), "Sections/static.tex")
    m = re.search(r"70k: newly registered ([\d,]+); standard-rate \+([\d,.]+)m; at GBP [\d,]+/firm \+([\d,.]+)m", sw)
    add("cut70_per_firm_variant", "static_sweep.txt", m.re.pattern, m.group(3),
        f"\\pounds{_num(m.group(3)):.0f}m", "Sections/static.tex")

    # --- seeds ---------------------------------------------------------------
    seeds = _read("seed_sensitivity.txt")
    m = re.search(r"half-range across seeds: E ±(\d+) \| b_llat ±([\d.]+) \| raise ±£([\d.]+)m \| taper ±£([\d.]+)m \| base ±£([\d.]+)bn", seeds)
    add("seed_E", "seed_sensitivity.txt", m.re.pattern, m.group(1), f"\\pm{m.group(1)}", "Appendix/a_inference.tex")
    add("seed_base", "seed_sensitivity.txt", m.re.pattern, m.group(5), f"\\pm\\pounds{m.group(5)}", "Appendix/a_inference.tex")
    return C


def check(claims: list[dict]) -> list[str]:
    misses = []
    for c in claims:
        tex = re.sub(r"\s+", " ", (PAPER / c["tex_file"]).read_text())
        if re.sub(r"\s+", " ", c["tex"]) not in tex:
            misses.append(f"{c['key']}: '{c['tex']}' not found in {c['tex_file']} (artifact {c['artifact']})")
    return misses


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    claims = build_claims()
    OUT.write_text(json.dumps(claims, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)} ({len(claims)} claims)")
    if args.check:
        misses = check(claims)
        for m in misses:
            print("MISS", m)
        print(f"{len(claims) - len(misses)}/{len(claims)} claims found in the manuscript")
        return 1 if misses else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
