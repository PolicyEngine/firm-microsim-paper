"""Regression test for the calibration dropout bias (referee finding 3).

Dropout was applied without the inverted-dropout rescale: training predictions
used ~keep_rate of the weight mass while the returned weights use all of it, so
final unmasked totals were driven to target / keep_rate (~+5.3% at keep_rate
0.95). With the rescale in place the final totals should match the targets.
"""

from __future__ import annotations

import torch

from firm_microsim.calibration import TargetSpec, optimize_weights
from firm_microsim.config import Config


def _solve_total(keep_rate: float) -> float:
    """Fit weights so a row of ones sums to a known target, return the final
    (unmasked) predicted total relative to that target."""
    torch.manual_seed(0)
    spec = TargetSpec(n_sectors=1, n_vat_sectors=0, n_near=1)
    n_firms = 2_000
    target_matrix = torch.ones((spec.n_targets, n_firms), dtype=torch.float32)
    target_values = torch.full((spec.n_targets,), float(n_firms), dtype=torch.float32)

    config = Config(
        device="cpu",
        n_iterations=1_500,
        learning_rate=0.02,
        early_stopping_patience=1_500,  # disable early stop for a clean read
        dropout_keep_rate=keep_rate,
        l1_reg_coef=0.0,  # isolate the target-matching behaviour
    )
    weights = optimize_weights(config, target_matrix, target_values, spec)
    predicted_total = float(weights.sum())
    return predicted_total / n_firms


def test_dropout_does_not_bias_totals_upward():
    ratio = _solve_total(keep_rate=0.95)
    # With the inverted-dropout rescale the final total tracks the target; the
    # old bug produced ~1/0.95 = 1.053. Require well under half that bias.
    assert abs(ratio - 1.0) < 0.02, f"final total biased by {ratio - 1.0:+.3%}"


def test_dropout_unbiased_at_lower_keep_rate():
    # A harsher keep_rate exaggerates the old bug (1/0.8 = 1.25); the fix should
    # still land near 1.0.
    ratio = _solve_total(keep_rate=0.8)
    assert abs(ratio - 1.0) < 0.03, f"final total biased by {ratio - 1.0:+.3%}"
