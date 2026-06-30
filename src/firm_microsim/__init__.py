"""firm_microsim — synthetic UK firm-population generator.

A single-version, ONS+HMRC-calibrated microsimulation generator for the UK
business population. The VAT threshold is one configurable parameter
(:data:`firm_microsim.config.VAT_THRESHOLD`, default £85k).

Example:
    >>> import firm_microsim
    >>> df = firm_microsim.generate(threshold=85)  # doctest: +SKIP
"""

from __future__ import annotations

import sys
import types

from .config import DEFAULT_CONFIG, VAT_THRESHOLD, Config

__version__ = "1.0.0"

__all__ = [
    "generate",
    "Config",
    "DEFAULT_CONFIG",
    "VAT_THRESHOLD",
    "ValidationReport",
    "__version__",
]


def generate(*args, **kwargs):
    """Generate a synthetic firm population without importing torch at package import."""

    from .generate import generate as _generate

    return _generate(*args, **kwargs)


def __getattr__(name: str):
    """Lazily expose heavyweight public helpers."""

    if name == "ValidationReport":
        from .validate import ValidationReport

        return ValidationReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _FirmMicrosimModule(types.ModuleType):
    """Keep the historic package-level generate function stable.

    Importing the ``firm_microsim.generate`` submodule makes Python assign that
    module to ``firm_microsim.generate``. Before lazy imports this package
    eagerly re-exported the callable and kept that public attribute stable; this
    hook preserves that behavior without importing torch during package import.
    """

    def __setattr__(self, name: str, value):
        if name == "generate" and isinstance(value, types.ModuleType):
            super().__setattr__("_generate_submodule", value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _FirmMicrosimModule
