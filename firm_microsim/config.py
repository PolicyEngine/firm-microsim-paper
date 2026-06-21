"""Configuration for the synthetic UK firm-population generator.

This is the SINGLE place to change the VAT threshold and all paths,
seeds, and optimizer hyperparameters. The package is intentionally
*single-version*: there is one ``VAT_THRESHOLD`` parameter rather than
separate 85k / 90k scripts. Change it here (or override on the CLI /
in :func:`firm_microsim.generate.generate`) and the entire pipeline,
including band edges and output, follows.

All monetary values are expressed in thousands of pounds (£k) unless
otherwise noted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
# config.py lives in   <repo>/firm_microsim/config.py
# so the repo root is one level up from this file's parent package.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR: Path = DATA_DIR / "synthetic"

# Figure outputs live in a top-level results/ directory (snake_case PNGs),
# and the paper sources in paper/ — mirroring the project house style.
RESULTS_DIR: Path = REPO_ROOT / "results"
PAPER_DIR: Path = REPO_ROOT / "paper"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Data vintages
# ---------------------------------------------------------------------------
# The project keeps TWO official-data vintages side by side under
# ``data/processed/<vintage>/``. Switching vintage is a one-liner: set the
# ``DATA_VINTAGE`` env var, pass ``data_vintage=`` to :class:`Config`, or call
# :meth:`Config.for_vintage`. Each vintage pins its own VAT threshold.
#
#   * "2023-24" — baseline matching the paper; £85k registration threshold.
#   * "2024-25" — latest gov release (ONS 2025 + HMRC 2024-25); £90k threshold.
VINTAGES: Dict[str, Dict[str, float]] = {
    "2023-24": {"threshold": 85.0},
    "2024-25": {"threshold": 90.0},
}
DEFAULT_VINTAGE: str = os.environ.get("DATA_VINTAGE", "2023-24")

# Processed input file names (placed in PROCESSED_DATA_DIR by an upstream
# ETL process). These map onto the original ONS + HMRC official tables.
INPUT_FILES: Dict[str, str] = {
    "ons_turnover": "ons_firm_turnover.csv",
    "ons_employment": "ons_firm_employment.csv",
    "hmrc_population_band": "hmrc_vat_population_by_turnover_band.csv",
    "hmrc_population_sector": "hmrc_vat_population_by_sector.csv",
    "hmrc_liability_band": "hmrc_vat_liability_by_turnover_band.csv",
    "hmrc_liability_sector": "hmrc_vat_liability_by_sector.csv",
}

# Generic, threshold-agnostic output file name. NOT hardcoded to 85/90.
OUTPUT_FILE: str = "synthetic_firms.csv"


@dataclass(frozen=True)
class Config:
    """Immutable run configuration for the generator.

    Attributes:
        vat_threshold: VAT registration threshold in £thousands. This is the
            single configurable threshold that drives band edges and the
            mandatory-vs-voluntary VAT split. Default 85 (£85k).
        seed: Random seed for full reproducibility (torch + numpy).
        device: Torch device string ('cpu', 'cuda', 'mps').
        n_iterations: Adam optimizer iterations for weight calibration.
        learning_rate: Adam learning rate.
        early_stopping_patience: Iterations without improvement before stopping.
        dropout_keep_rate: Fraction of firm weights kept each training step.
        l1_reg_coef: L1 regularization coefficient on log-weights.
        grad_clip_norm: Max gradient norm for clipping.
        turnover_importance: Importance weight on the 7 turnover-band targets.
        sector_importance: Importance weight on HMRC sector targets.
        employment_importance: Importance weight on ONS employment-band targets.
        vat_liability_sector_importance: Weight on VAT-liability-by-sector targets.
        vat_liability_band_importance: Weight on VAT-liability-by-band targets.
        data_dir: Root data directory (overridable for tests / alt layouts).
        processed_dir: Directory holding processed input CSVs.
        synthetic_dir: Directory to which the synthetic CSV is written.
        output_file: Output CSV file name (generic, not threshold-specific).
    """

    # --- Data vintage (selects processed-data subdir + default threshold) --
    data_vintage: str = DEFAULT_VINTAGE

    # --- The single configurable threshold -------------------------------
    # Defaults to the vintage's threshold; VAT_THRESHOLD env overrides if set.
    vat_threshold: float = float(
        os.environ.get("VAT_THRESHOLD", VINTAGES[DEFAULT_VINTAGE]["threshold"])
    )

    # --- Reproducibility / compute ---------------------------------------
    seed: int = 42
    device: str = "cpu"

    # --- Optimizer hyperparameters ---------------------------------------
    n_iterations: int = 1_000
    learning_rate: float = 0.01
    early_stopping_patience: int = 100
    dropout_keep_rate: float = 0.95  # keep 95%, drop 5% each step
    l1_reg_coef: float = 0.01
    grad_clip_norm: float = 1.0

    # --- Multi-objective importance weights ------------------------------
    turnover_importance: float = 5.0  # turnover bands ~5x (most critical)
    sector_importance: float = 1.0
    employment_importance: float = 1.0
    vat_liability_sector_importance: float = 1.0
    vat_liability_band_importance: float = 2.0

    # --- Paths (overridable) ---------------------------------------------
    data_dir: Path = DATA_DIR
    processed_dir: Path = PROCESSED_DATA_DIR / DEFAULT_VINTAGE
    synthetic_dir: Path = SYNTHETIC_DATA_DIR
    output_file: str = OUTPUT_FILE

    input_files: Dict[str, str] = field(default_factory=lambda: dict(INPUT_FILES))

    def __post_init__(self) -> None:
        # Keep processed_dir in sync with data_vintage unless the caller
        # passed a bespoke path (e.g. a tmp dir in tests). A path is treated
        # as "canonical" (and thus re-derived from the vintage) if it is the
        # bare processed dir or any known vintage subdir.
        canonical = {PROCESSED_DATA_DIR / v for v in VINTAGES} | {PROCESSED_DATA_DIR}
        if self.processed_dir in canonical:
            object.__setattr__(
                self, "processed_dir", PROCESSED_DATA_DIR / self.data_vintage
            )

    @classmethod
    def for_vintage(cls, vintage: str, **overrides) -> "Config":
        """Build a config for a named data vintage (one-call switch).

        Sets ``data_vintage``, the matching ``vat_threshold``, and the
        ``data/processed/<vintage>/`` input directory consistently. Any
        keyword in ``overrides`` takes precedence (e.g. a custom seed).

        Example:
            >>> Config.for_vintage("2024-25")  # £90k, latest gov data
        """
        if vintage not in VINTAGES:
            raise ValueError(
                f"Unknown vintage {vintage!r}; choose from {sorted(VINTAGES)}"
            )
        params = {
            "data_vintage": vintage,
            "vat_threshold": VINTAGES[vintage]["threshold"],
        }
        params.update(overrides)
        return cls(**params)

    @property
    def output_path(self) -> Path:
        """Full path to the synthetic output CSV."""
        return self.synthetic_dir / self.output_file

    def input_path(self, key: str) -> Path:
        """Resolve a processed input CSV path by logical key."""
        return self.processed_dir / self.input_files[key]


# A module-level default instance for convenience / scripting.
DEFAULT_CONFIG = Config()

# Convenience module-level alias: the single threshold value (£k).
VAT_THRESHOLD: float = DEFAULT_CONFIG.vat_threshold
