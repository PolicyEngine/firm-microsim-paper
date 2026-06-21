"""Static (mechanical, no-behaviour) VAT-threshold costing and figures.

    from static import StaticVATModel
    model = StaticVATModel()           # uses the £90k (2024-25) synthetic data
    model.threshold_sweep("2025-26")   # revenue / firm changes vs £90k
    model.anchor_reform()              # £85k->£90k, model vs HMRC, by year

Run ``python -m static`` to regenerate all three figures into ``results/``.
"""

from .model import POLICY_THRESHOLD, SWEEP_THRESHOLDS, FISCAL_YEARS, StaticVATModel

__all__ = [
    "StaticVATModel",
    "FISCAL_YEARS",
    "POLICY_THRESHOLD",
    "SWEEP_THRESHOLDS",
]
