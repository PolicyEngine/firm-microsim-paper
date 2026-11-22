#!/usr/bin/env python3
"""Claim manifest: every headline number the manuscript quotes, traced to its
artifact (issue #40).

``python scripts/claims.py``          rewrites ``results/claims.json`` from the
                                      checked ``results/*.txt`` artifacts.
``python scripts/claims.py --check``  rebuilds the claims in memory (writing
                                      nothing), fails if they differ from the
                                      committed ``results/claims.json``, and
                                      asserts that each claim's rendering occurs
                                      in the named source file as a delimited
                                      token (not inside a longer number),
                                      exiting 1 on any miss.

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
ROOT = REPO
CLAIMS_JSON = Path("results") / "claims.json"


def _read(name: str) -> str:
    return (ROOT / "results" / name).read_text()


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def _tex_int(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def _signed(x: float, fmt: str = ",.0f") -> str:
    """``$+1{,}234$`` / ``$-1{,}234$``: the sign comes from the value."""
    body = format(abs(x), fmt).replace(",", "{,}")
    return f"${'+' if x > 0 else '-'}{body}$" if x != 0 else f"${body}$"


def _series(vals: list[float]) -> str:
    return ", ".join(("$+$" if v >= 0 else "$-$") + f"{abs(v):.0f}" for v in vals)


def build_claims() -> list[dict]:
    C: list[dict] = []

    def add(key, artifact, pattern, value, tex, tex_file, note="",
            allow_short=False, allow_zero=False):
        if not tex_file.startswith(("paper/", "README")):
            tex_file = "paper/" + tex_file
        if not allow_short and len(re.findall(r"\d", tex)) < 3:
            raise ValueError(f"claim {key}: rendering {tex!r} has fewer than three "
                             "digits; pass allow_short=True if intended")
        vals = value if isinstance(value, list) else [value]
        if not allow_zero and any(_num(str(v)) == 0 for v in vals):
            raise ValueError(f"claim {key}: extracted value is zero; "
                             "pass allow_zero=True if intended")
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

    blocks = {v: cal.split(f"Vintage {v}")[1].split("Vintage")[0] for v in ("2023-24", "2024-25")}

    def _cal(v: str, lab: str) -> str:
        return re.search(re.escape(lab) + r"\s+([\d.]+)%", blocks[v]).group(1)

    for key, lab, tex_lab, readme_lab in (
        ("hmrc_bands", "HMRC Turnover Bands", "HMRC turnover bands (registered)",
         "HMRC turnover bands (registered rows; hold by construction)"),
        ("ons_pop", "ONS Population", "ONS enterprise total (frame)", "ONS population (frame rows)"),
        ("employment", "Employment Bands", "ONS employment bands (frame)", "Employment bands (frame rows)"),
        ("sector", "Sector Distribution", "HMRC sector counts (registered)",
         "Sector distribution (registered rows)"),
        ("liab", "VAT Liability by Band", "HMRC net liability by band (registered)",
         "VAT liability by band (6 calibrated bands, registered rows)"),
    ):
        a, b = _cal("2023-24", lab), _cal("2024-25", lab)
        add(f"caltable_{key}", "calibration_accuracy.txt", lab, [a, b],
            f"{tex_lab} & {a}\\% & {b}\\% \\\\", "Appendix/a_data.tex")
        add(f"readme_{key}", "calibration_accuracy.txt", lab, [a, b],
            f"| {readme_lab} | {a}% | {b}% |", "README.md")
    a, b = (re.search(r"Overall \(5 calibrated dims\)\s+([\d.]+)%", blocks[v]).group(1)
            for v in ("2023-24", "2024-25"))
    add("readme_overall", "calibration_accuracy.txt", "Overall", [a, b],
        f"| **Overall (5 calibrated dimensions)** | **{a}%** | **{b}%** |", "README.md")

    # --- static sweep / anchor -------------------------------------------
    sw = _read("static_sweep.txt")
    anchor = [float(a[2]) for a in re.findall(r"^(20\d\d-\d\d)\s+(-?[\d.]+)\s+(-?[\d.]+)$", sw, re.M)]
    add("anchor_series", "static_sweep.txt", "anchor table", anchor, _series(anchor), "Sections/static.tex")
    ret = [float(x) for x in re.findall(r"retention-adjusted ([+-][\d.]+)m", sw)]
    add("anchor_retention_series", "static_sweep.txt", "retention-adjusted", ret, _series(ret), "Sections/static.tex")
    nogap = [float(x) for x in re.findall(r"whole-band release ([+-][\d.]+)m", sw)]
    add("anchor_nogap_series", "static_sweep.txt", "whole-band release", nogap, _series(nogap), "Sections/static.tex")
    fixed = [float(x) for x in re.findall(r"fixed-preference ([+-][\d.]+)m", sw)]
    add("anchor_fixed_series", "static_sweep.txt", "fixed-preference", fixed, _series(fixed), "Sections/static.tex")
    add("anchor_2526_abs", "static_sweep.txt", "anchor 2025-26", anchor[1],
        f"${'-' if anchor[1] < 0 else '+'}\\pounds{abs(anchor[1]):.0f}$m", "Sections/conclusion.tex")
    add("anchor_ret_2526_abs", "static_sweep.txt", "retention 2025-26", ret[1],
        f"${'-' if ret[1] < 0 else '+'}\\pounds{abs(ret[1]):.0f}$m", "Sections/conclusion.tex")
    m = re.search(r"Total VAT revenue at GBP 90k, 2025-26: ([\d.]+)bn", sw)
    add("base_2526", "static_sweep.txt", m.re.pattern, m.group(1), f"\\pounds{m.group(1)}bn", "Sections/static.tex")
    m = re.search(r"2023-24: in-scope base .*?= ([\d.]+)bn", sw)
    add("ubase_2324", "static_sweep.txt", m.re.pattern, m.group(1), f"\\pounds{m.group(1)}bn", "Sections/static.tex")
    base_m = _num(re.search(r"Total VAT revenue at GBP 90k, 2025-26: ([\d.]+)bn", sw).group(1)) * 1000.0
    rows = re.findall(r"^\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)$", sw.split("Threshold sweep")[1], re.M)
    for t, rev, firms in rows:
        if float(t) == 90.0:
            continue
        rev, firms = float(rev), float(firms)
        row = (f"\\pounds{_tex_int(float(t) * 1000)} & {_signed(rev, ',.1f')} & "
               f"{_signed(rev / base_m * 100.0, '.2f')} & {_signed(firms, '.1f')} \\\\")
        add(f"sweep_{int(float(t))}k", "static_sweep.txt", "sweep table row (revenue, %, firms)",
            [rev, firms], row, "Sections/static.tex")

    # --- reform menu ------------------------------------------------------
    menu = _read("reform_menu_common_base.txt")
    for key, label in (("raise100k", "Raise threshold to GBP100,000"), ("taper", "Graduated taper [85k,141.7k]"),
                       ("taper_flat50", "Flat 50% marginal taper [85k,141.7k]"),
                       ("rate10", "Reduced rate 10% [85k,105k]"), ("rate15", "Reduced rate 15% [85k,105k]")):
        m = re.search(re.escape(label) + r"\s+\S.*?(-?\d+)\s*$", menu, re.M)
        val = int(m.group(1))
        add(f"menu_{key}", "reform_menu_common_base.txt", label, val,
            _signed(val), "Sections/static.tex")

    # --- dynamic -----------------------------------------------------------
    dyn = _read("dynamic_reform_results.txt")
    for key, label in (("rate10", "Reduced rate 10%"), ("rate15", "Reduced rate 15%")):
        ln = [line for line in dyn.splitlines() if line.strip().startswith(label)][0]
        vals = [int(x) for x in re.findall(r"£\s*([+-]?\d+)m", ln)]
        add(f"dyn_{key}_e017", "dynamic_reform_results.txt", label, vals[2],
            f"${'-' if vals[2] < 0 else '+'}\\pounds{abs(vals[2])}$m", "Sections/behavioural.tex")
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
    m = re.search(r"^20% \(baseline notch\)\s+[\d,]+\s+\[[\d., ]+\)\s+([\d,]+)", dom, re.M)
    base = _num(m.group(1))
    add("dominated_base_obs", "dominated_region_mass.txt", m.re.pattern, base,
        _tex_int(round(base, -2)), "Sections/model.tex")
    for key, lab in (("dominated_15_obs", "15% band"), ("dominated_10_obs", "10% band")):
        m = re.search(r"^" + re.escape(lab) + r"\s+[\d,]+\s+\[[\d., ]+\)\s+([\d,]+)", dom, re.M)
        v = _num(m.group(1))
        add(key, "dominated_region_mass.txt", m.re.pattern, v, _tex_int(round(v, -2)),
            "Sections/model.tex")
    for tag, lab in (("15", "15% band"), ("10", "10% band")):
        m = re.search(re.escape(lab) + r"\s+secondary \[[^)]+\) width GBP ([\d,]+)\s+OBS = ([\d,]+)", dom)
        if tag == "15":
            add("secondary_width_15", "dominated_region_mass.txt", m.re.pattern, _num(m.group(1)),
                f"\\pounds{_tex_int(_num(m.group(1)))}", "Sections/model.tex")
        add(f"secondary_{tag}_obs", "dominated_region_mass.txt", m.re.pattern, _num(m.group(2)),
            _tex_int(round(_num(m.group(2)), -2)), "Sections/model.tex")
        m = re.search(r"primary OBS [\d,]+ \+ secondary OBS [\d,]+ = TOTAL ([\d,]+)\s+\(baseline [\d,]+, ([+-][\d.]+)%\)",
                      dom.split(lab + "  secondary")[1])
        add(f"dominated_total_{tag}", "dominated_region_mass.txt", m.re.pattern, _num(m.group(1)),
            _tex_int(round(_num(m.group(1)), -2)), "Sections/model.tex")
        add(f"dominated_total_{tag}_pct", "dominated_region_mass.txt", m.re.pattern,
            abs(float(m.group(2))), f"{abs(float(m.group(2))):.1f}\\%", "Sections/model.tex",
            allow_short=True)
    menu_txt = _read("reform_menu_common_base.txt")
    m = re.search(r"Raise-to-100k band \[85k,100k\): ([\d,]+) firms released", menu_txt)
    rel = _num(m.group(1))
    for f in ("Sections/static.tex", "Sections/model.tex"):
        add(f"released_100k_{f.split('/')[1][:-4]}", "reform_menu_common_base.txt", m.re.pattern,
            rel, _tex_int(round(rel, -2)), f)

    # --- seeds ---------------------------------------------------------------
    seeds = _read("seed_sensitivity.txt")
    m = re.search(r"half-range across seeds: E ±(\d+) \| b_llat ±([\d.]+) \| raise ±£([\d.]+)m \| taper ±£([\d.]+)m \| base ±£([\d.]+)bn", seeds)
    add("seed_E", "seed_sensitivity.txt", m.re.pattern, m.group(1), f"\\pm{m.group(1)}", "Appendix/a_inference.tex")
    add("seed_base", "seed_sensitivity.txt", m.re.pattern, m.group(5), f"\\pm\\pounds{m.group(5)}", "Appendix/a_inference.tex")
    return C


def _token_pattern(tex: str) -> re.Pattern:
    """Whitespace-insensitive pattern for ``tex`` that may not sit inside a
    longer number: no digit (or ``{,}``-grouped digit) immediately before a
    leading digit, and no digit/decimal continuation after a trailing one."""
    parts = [re.escape(p) for p in tex.split()]
    pat = r"\s+".join(parts)
    if tex[:1].isdigit():
        pat = r"(?<![\d.])(?<!\{,\})" + pat
    if tex[-1:].isdigit():
        pat = pat + r"(?!\d|\.\d|\{,\}\d)"
    return re.compile(pat)


def check(claims: list[dict]) -> list[str]:
    misses = []
    for c in claims:
        text = (ROOT / c["tex_file"]).read_text()
        if not _token_pattern(c["tex"]).search(text):
            misses.append(f"{c['key']}: '{c['tex']}' not found in {c['tex_file']} (artifact {c['artifact']})")
    return misses


def main(argv=None) -> int:
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", type=Path, default=REPO,
                    help="repository root holding results/, paper/ and README.md")
    args = ap.parse_args(argv)
    ROOT = args.root.resolve()
    claims = build_claims()
    rendered = json.dumps(claims, indent=2) + "\n"
    out = ROOT / CLAIMS_JSON
    if not args.check:
        out.write_text(rendered)
        print(f"wrote {CLAIMS_JSON} ({len(claims)} claims)")
        return 0
    status = 0
    if not out.exists() or out.read_text() != rendered:
        print(f"STALE {CLAIMS_JSON} differs from the artifacts; run scripts/claims.py")
        status = 1
    misses = check(claims)
    for m in misses:
        print("MISS", m)
    print(f"{len(claims) - len(misses)}/{len(claims)} claims found in the manuscript")
    return 1 if misses else status


if __name__ == "__main__":
    sys.exit(main())
