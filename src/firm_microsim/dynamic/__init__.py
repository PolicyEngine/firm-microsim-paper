"""Iso-elastic (Kleven-Waseem) dynamic VAT-notch structural simulator.

This package reformulates the behavioural forward-solve onto the correct
iso-elastic quasi-linear model, in which a SINGLE turnover elasticity ``e``
governs the response of turnover to the net-of-tax rate. It replaces the
previous Cobb-Douglas formulation, which tied the response to the production
returns ``alpha`` (implying an absurd elasticity ~90) and whose forward-solve
did not depend on ``e`` at all.

Every headline number is cross-checked against the analytic dominated region,
the verified iso-elastic marginal buncher (reused from :mod:`notch.model` and
re-derived independently here), the elasticity identity
``d ln y*/d ln(1-tau) = e``, and the trusted static reform costs (recovered as
the ``e -> 0`` limit of the behavioural costs). See :func:`dynamic.model.crosscheck`.
"""

from .model import (
    E_HEADLINE,
    ELASTICITIES,
    TAU_MAX,
    T_STAR,
    TAPER_TOP,
    build_reforms,
    crosscheck,
    dominated_region_width,
    forward_solve_iso,
    iso_cost,
    iso_mc,
    iso_profit,
    load_reform_data,
    make_schedule_raise,
    make_schedule_reduced_rate,
    marginal_buncher,
    marginal_buncher_iso,
    recover_ability,
    reform_revenue,
    schedule_taper,
)

__all__ = [
    "TAU_MAX",
    "T_STAR",
    "TAPER_TOP",
    "ELASTICITIES",
    "E_HEADLINE",
    "iso_cost",
    "iso_mc",
    "iso_profit",
    "recover_ability",
    "forward_solve_iso",
    "marginal_buncher",
    "marginal_buncher_iso",
    "dominated_region_width",
    "reform_revenue",
    "schedule_taper",
    "make_schedule_raise",
    "make_schedule_reduced_rate",
    "build_reforms",
    "load_reform_data",
    "crosscheck",
]
