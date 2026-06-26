from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from firm_microsim.config import Config


def test_import_public_packages() -> None:
    import firm_microsim
    import firm_microsim.bunching
    import firm_microsim.dynamic
    import firm_microsim.notch
    import firm_microsim.static

    assert firm_microsim.__version__ == "1.0.0"


def test_cli_help_entry_points() -> None:
    modules = [
        "firm_microsim",
        "firm_microsim.static",
        "firm_microsim.bunching",
        "firm_microsim.notch",
        "firm_microsim.dynamic",
    ]
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout


def test_console_script_help_entry_points() -> None:
    scripts = [
        "firm-microsim",
        "firm-microsim-report",
        "firm-microsim-figures",
        "firm-microsim-static",
        "firm-microsim-bunching",
        "firm-microsim-notch",
        "firm-microsim-dynamic",
        "firm-microsim-placebo",
        "firm-microsim-dominated-region",
        "firm-microsim-reform-menu",
    ]
    bin_dir = Path(sys.executable).parent
    for script in scripts:
        result = subprocess.run(
            [str(bin_dir / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in result.stdout


def test_dynamic_rejects_unsupported_vintage() -> None:
    bin_dir = Path(sys.executable).parent
    result = subprocess.run(
        [str(bin_dir / "firm-microsim-dynamic"), "--vintage", "2024-25"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_config_data_vintage_sets_matching_threshold() -> None:
    cfg_2023 = Config(data_vintage="2023-24")
    cfg_2024 = Config(data_vintage="2024-25")

    assert cfg_2023.vat_threshold == 85.0
    assert cfg_2023.processed_dir.name == "2023-24"
    assert cfg_2024.vat_threshold == 90.0
    assert cfg_2024.processed_dir.name == "2024-25"


def test_config_preserves_explicit_threshold_override() -> None:
    cfg = Config(data_vintage="2024-25", vat_threshold=88.0)

    assert cfg.vat_threshold == 88.0
    assert cfg.processed_dir.name == "2024-25"


def test_config_uses_vat_threshold_env_override(monkeypatch) -> None:
    monkeypatch.setenv("VAT_THRESHOLD", "92.5")

    cfg = Config(data_vintage="2024-25")

    assert cfg.vat_threshold == 92.5
    assert cfg.processed_dir.name == "2024-25"
