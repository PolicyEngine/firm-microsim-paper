from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
