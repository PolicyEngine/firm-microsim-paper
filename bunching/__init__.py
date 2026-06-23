"""VAT-threshold bunching: reduced-form excess-mass estimator.

Reproduces the paper's reduced-form bunching results on the synthetic firm
population. The structural notch model lives in the ``notch`` package.

Run ``python -m bunching`` to regenerate the bunching figure and print the
estimates.
"""

from .model import (
    BunchingEstimator,
    DEFAULT_DEGREE,
    DEFAULT_WINDOW,
    TAU_E,
    TAU_MAX,
)

__all__ = [
    "BunchingEstimator",
    "TAU_MAX",
    "DEFAULT_DEGREE",
    "DEFAULT_WINDOW",
    "TAU_E",
]
