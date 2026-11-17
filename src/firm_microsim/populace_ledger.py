"""Deprecated alias for :mod:`firm_microsim.microcosm_ledger`."""  # compat: populace alias, remove after 1.1.0

import warnings  # compat: populace alias, remove after 1.1.0

from .microcosm_ledger import *  # noqa: F403  # compat: populace alias, remove after 1.1.0
from .microcosm_ledger import (  # noqa: F401  # compat: populace alias, remove after 1.1.0
    REFERENCE_MICROCOSM_LEDGER_2024_25 as REFERENCE_POPULACE_LEDGER_2024_25,  # compat: populace alias, remove after 1.1.0
    cli,  # compat: populace alias, remove after 1.1.0
    load_microcosm_source_data as load_populace_source_data,  # compat: populace alias, remove after 1.1.0
    run_microcosm_from_ledger_facts as run_populace_from_ledger_facts,  # compat: populace alias, remove after 1.1.0
)

warnings.warn(  # compat: populace alias, remove after 1.1.0
    "firm_microsim.populace_ledger is now firm_microsim.microcosm_ledger; "  # compat: populace alias, remove after 1.1.0
    "the old name goes in the next release",  # compat: populace alias, remove after 1.1.0
    DeprecationWarning,  # compat: populace alias, remove after 1.1.0
    stacklevel=2,  # compat: populace alias, remove after 1.1.0
)  # compat: populace alias, remove after 1.1.0

if __name__ == "__main__":  # compat: populace alias, remove after 1.1.0
    cli()  # compat: populace alias, remove after 1.1.0
