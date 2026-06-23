"""Structural VAT-notch model (Kleven--Waseem) for the UK registration threshold.

The §6 structural counterpart to the reduced-form bunching package: once turnover
crosses the threshold, VAT falls on the firm's *entire* turnover, creating a
discrete notch, a dominated region, and a marginal buncher. Calibrated to the
reduced-form turnover elasticity, it re-solves firms under counterfactual
schedules.

    from notch import NotchModel
    m = NotchModel("2023-24")        # £85k vintage
    m.summary()                       # dominated region, marginal buncher, ...

Run ``python -m notch`` to regenerate the notch figure and print the summary.
"""

from .model import ALPHA, E_MEAN, E_MEDIAN, TAU, NotchModel

__all__ = ["NotchModel", "TAU", "ALPHA", "E_MEDIAN", "E_MEAN"]
