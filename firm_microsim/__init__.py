"""firm_microsim — synthetic UK firm-population generator.

A single-version, ONS+HMRC-calibrated microsimulation generator for the UK
business population. The VAT threshold is one configurable parameter
(:data:`firm_microsim.config.VAT_THRESHOLD`, default £85k).

Example:
    >>> import firm_microsim
    >>> df = firm_microsim.generate(threshold=85)  # doctest: +SKIP
"""

from .config import DEFAULT_CONFIG, VAT_THRESHOLD, Config
from .generate import generate
from .validate import ValidationReport

__version__ = "1.0.0"

__all__ = [
    "generate",
    "Config",
    "DEFAULT_CONFIG",
    "VAT_THRESHOLD",
    "ValidationReport",
    "__version__",
]
