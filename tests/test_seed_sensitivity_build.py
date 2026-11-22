"""seed_sensitivity always regenerates each seed (review C3)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "seed_sensitivity_script", REPO / "scripts" / "seed_sensitivity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_seed_is_generated_not_read_from_cached_csv(tmp_path, monkeypatch) -> None:
    mod = _load_script()
    cols = mod.COLS
    # A planted one-row canonical CSV must be ignored.
    pd.DataFrame([[1.0] * len(cols)], columns=cols).to_csv(
        tmp_path / f"synthetic_firms_{mod.VINTAGE}.csv", index=False
    )
    monkeypatch.setattr(mod, "SYNTHETIC_DATA_DIR", tmp_path, raising=False)

    calls = []
    fresh = pd.DataFrame({c: [2.0, 3.0] for c in cols + ["extra"]})

    def fake_generate(*, vintage, seed, write):
        calls.append((vintage, seed, write))
        return fresh

    monkeypatch.setattr(mod, "generate", fake_generate)
    out = mod.build_for_seed(mod.DEFAULT_CONFIG.seed)
    assert calls == [(mod.VINTAGE, mod.DEFAULT_CONFIG.seed, False)]
    assert list(out.columns) == cols
    assert len(out) == 2
