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
        for lab, key in (("Sector Distribution", "sector"), ("VAT Liability by Band", "liab")):
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
    m = re.search(r"e=0\.17\s+n_H\(delta=0\)=£\s*([\d.]+)k.*?n_H\(delta=0\.6\)=£\s*([\d.]+)k", dyn)
    add("nH_e017_d06", "dynamic_reform_results.txt", m.re.pattern, m.group(2),
        f"\\pounds{_num(m.group(2))*1000:,.0f}".replace(",", "{,}"), "Sections/behavioural.tex")

    # --- bunching -----------------------------------------------------------
    bun = _read("bunching_inference.txt")
    for v, tag in (("2023-24", "2324"), ("2024-25", "2425")):
        m = re.search(rf"Vintage {v}.*?E = ([\d,]+) \(gross\).*?b_llat = ([\d.]+)", bun, re.S)
        add(f"E_{tag}", "bunching_inference.txt", "headline E", m.group(1),
            m.group(1).replace(",", "{,}"), "Sections/bunching.tex")
        add(f"bllat_{tag}", "bunching_inference.txt", "headline b_llat", m.group(2),
            f"{float(m.group(2)):.3f}", "Sections/bunching.tex")

    # --- dominated region ---------------------------------------------------
    dom = _read("dominated_region_mass.txt")
    m = re.search(r"20% \(baseline notch\)\s+[\d,]+\s+\[[^\]]+\)\s+([\d,]+)", dom)
    base = _num(m.group(1))
    add("dominated_base_obs", "dominated_region_mass.txt", m.re.pattern, base,
        _tex_int(round(base, -2)), "Sections/model.tex")

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
