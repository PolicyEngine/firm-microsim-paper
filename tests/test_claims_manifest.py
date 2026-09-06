"""Every manuscript headline number traces to a checked artifact (issue #40)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_claims_manifest_matches_manuscript() -> None:
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "claims.py"), "--check"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
