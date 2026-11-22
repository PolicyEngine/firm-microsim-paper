"""Every manuscript headline number traces to a checked artifact (issue #40).

The mutation tests guard the check itself: a changed artifact value must
produce a miss, and ``--check`` must not rewrite ``results/claims.json``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "claims.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )


def test_claims_manifest_matches_manuscript() -> None:
    before = (REPO / "results" / "claims.json").read_bytes()
    r = _run("--check")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (REPO / "results" / "claims.json").read_bytes() == before


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    for sub in ("results", "paper"):
        shutil.copytree(REPO / sub, tmp_path / sub,
                        ignore=shutil.ignore_patterns("*.png", "*.pdf", "site", "out"))
    shutil.copy(REPO / "README.md", tmp_path / "README.md")
    return tmp_path


def _mutate(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"{old!r} not in {path.name}"
    path.write_text(text.replace(old, new, 1))


@pytest.mark.parametrize(
    ("artifact", "old", "new", "key"),
    [
        # flip the sign of the raise-to-£100k costing
        ("reform_menu_common_base.txt", "Location                            -742",
         "Location                            +742", "menu_raise100k"),
        # the dominated-band count (previously checked against the string "0")
        ("dominated_region_mass.txt", " 142,834 ", " 152,834 ",
         "dominated_base_obs"),
        # direction of the 15% reduced-rate total relative to the baseline band
        ("dominated_region_mass.txt", "baseline 142,834, +0.9%", "baseline 142,834, -0.9%",
         "dominated_total_15_pct"),
        # one firm-count cell of the threshold sweep table
        ("static_sweep.txt", "-27.381047", "-28.381047", "sweep_95k"),
    ],
)
def test_changed_artifact_value_is_a_miss(tree, artifact, old, new, key) -> None:
    _mutate(tree / "results" / artifact, old, new)
    assert _run("--root", str(tree)).returncode == 0  # rebuild claims.json
    r = _run("--check", "--root", str(tree))
    assert r.returncode == 1
    assert f"MISS {key}:" in r.stdout, r.stdout


def test_stale_claims_json_fails_check(tree) -> None:
    _mutate(tree / "results" / "static_sweep.txt", "-27.381047", "-28.381047")
    r = _run("--check", "--root", str(tree))
    assert r.returncode == 1
    assert "STALE" in r.stdout
