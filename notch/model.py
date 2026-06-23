"""Structural VAT-*notch* model (Kleven & Waseem, 2013) on the synthetic firms.

A VAT *notch* (not a kink): once turnover ``y`` crosses the registration
threshold ``T*`` the firm becomes liable for VAT on its ENTIRE turnover, not
just the increment above ``T*``. After-tax revenue therefore drops discretely
at ``T*`` by ``tau * T*``. Following Kleven & Waseem (2013) and Saez (2010) the
firm's profit under a Cobb-Douglas cost ``c(y) = (y/A)^(1/alpha)`` is::

    pi(y) = y           - c(y)        for  y <  T*   (unregistered)
    pi(y) = (1 - tau)*y - c(y)        for  y >= T*   (registered)

From this we obtain:

* the **dominated region** ``(T*, T* + a)`` where no firm ever locates, with
  the exact Kleven-Waseem width ``a = T* * tau / (1 - tau)``;
* the **marginal buncher** ``y_H`` — the firm whose frictionless optimum sits
  at the top of the bunching segment and is exactly indifferent between bunching
  at ``T*`` and registering at its post-notch interior optimum.

Because the estimated returns-to-scale ``alpha ~= 0.989`` is near constant
returns, the Cobb-Douglas interior optimum on the taxed branch is numerically
degenerate (the exponent ``alpha/(1-alpha) ~= 90`` explodes), so the headline
marginal-buncher solve uses the standard iso-elastic Kleven-Waseem form,
parameterised by the structural turnover elasticity ``e`` calibrated to the
paper's reduced-form bunching (median ``e ~= 0.17``, mean ``e ~= 0.32``). The
Cobb-Douglas FOC productivity inversion is retained only as a documented
diagnostic.

All turnover quantities are in thousands of pounds (£k) internally, matching
the synthetic-firm data columns (``annual_turnover_k`` etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from firm_microsim.config import SYNTHETIC_DATA_DIR, VINTAGES

# ---------------------------------------------------------------------------
# Calibrated structural parameters
# ---------------------------------------------------------------------------
TAU = 0.20          # statutory UK VAT rate (full effective rate once registered)
ALPHA = 0.989       # returns-to-scale (near-CRS; flags taxed-branch degeneracy)

# Structural turnover elasticities for the marginal-buncher solve, calibrated to
# the paper's reduced-form bunching estimates (median ~0.17, mean ~0.32).
E_MEDIAN = 0.17
E_MEAN = 0.32


class NotchModel:
    """Kleven-Waseem VAT-notch model over the synthetic firm population.

    Parameters
    ----------
    vintage:
        Data vintage label (e.g. ``"2023-24"``). Selects both the synthetic
        firm CSV and the VAT threshold ``T*`` (£k) via
        :data:`firm_microsim.config.VINTAGES` — never hardcoded.
    tau:
        Statutory VAT rate applied to the whole turnover once registered.
    alpha:
        Cobb-Douglas returns-to-scale parameter.
    """

    def __init__(self, vintage: str = "2023-24", tau: float = TAU, alpha: float = ALPHA) -> None:
        if vintage not in VINTAGES:
            raise ValueError(
                f"Unknown vintage {vintage!r}; choose from {sorted(VINTAGES)}"
            )
        self.vintage = vintage
        self.tau = float(tau)
        self.alpha = float(alpha)
        # Threshold (£k) from the vintage — NOT a hardcoded 85.
        self.t_star = float(VINTAGES[vintage]["threshold"])
        # Cost-curvature exponent 1/alpha.
        self.exp = 1.0 / self.alpha

        path = SYNTHETIC_DATA_DIR / f"synthetic_firms_{vintage}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — generate it first with:\n"
                f"  python -m firm_microsim --vintage {vintage} "
                f"--output synthetic_firms_{vintage}.csv"
            )
        cols = ["annual_turnover_k", "weight"]
        header = pd.read_csv(path, nrows=0).columns
        if "annual_input_k" in header:
            cols.append("annual_input_k")
        self.firms = pd.read_csv(path, usecols=cols)

    # -- Cost, profit, interior optima -------------------------------------
    def cost(self, y, A):
        """Cobb-Douglas cost ``c(y) = (y/A)^(1/alpha)``."""
        return (y / A) ** self.exp

    def profit(self, y, A, net):
        """Profit ``pi(y) = net*y - c(y)``; ``net`` = 1 below ``T*``, ``1-tau`` above."""
        return net * y - self.cost(y, A)

    def interior_optimum(self, A, net):
        """Closed-form unconstrained interior optimum of ``pi(y)=net*y-(y/A)^(1/alpha)``.

        FOC ``net = (1/alpha) A^(-1/alpha) y^((1-alpha)/alpha)`` gives
        ``y* = A * (alpha * net)^(alpha/(1-alpha))``.

        For ``alpha`` near 1 the exponent ``alpha/(1-alpha)`` is very large, so
        this is well-defined on the unregistered branch (``net=1``) but
        explodes/vanishes for ``net<1`` — the economic signature of near-CRS,
        and the reason the marginal buncher uses the iso-elastic K-W form
        instead of the taxed-branch CD optimum.
        """
        return A * (self.alpha * net) ** (self.alpha / (1.0 - self.alpha))

    # -- (i) Dominated region width ----------------------------------------
    def dominated_region_width_kw(self) -> float:
        """Exact Kleven-Waseem dominated-region width ``a = T* * tau/(1-tau)``.

        ``T*+a`` is the post-notch turnover whose after-tax revenue just equals
        locating exactly at ``T*`` (unregistered):
        ``(1 - tau)*(T* + a) = T*``  =>  ``a = T* * tau / (1 - tau)``. No firm
        ever chooses turnover in ``(T*, T*+a)``.
        """
        return self.t_star * self.tau / (1.0 - self.tau)

    def dominated_region_width_numeric(self, A_at_tstar) -> float:
        """Cost-adjusted dominated-region upper edge (numeric cross-check).

        For a firm whose unregistered optimum sits exactly at ``T*``, find the
        smallest ``a>0`` with ``pi_registered(T*+a) = pi(T*)``.
        """
        pi_bunch = self.profit(self.t_star, A_at_tstar, net=1.0)
        grid = np.linspace(1e-6, 2.0 * self.dominated_region_width_kw(), 400000)
        pi_reg = self.profit(self.t_star + grid, A_at_tstar, net=1.0 - self.tau)
        below = pi_reg < pi_bunch
        if below[0] and not below[-1]:
            return float(grid[np.argmax(~below)])
        return self.dominated_region_width_kw()

    # -- (ii) Marginal buncher (iso-elastic Kleven-Waseem) -----------------
    def marginal_buncher(self, e: float) -> tuple[float, float]:
        """Return ``(n_H, dy_star)`` for structural turnover elasticity ``e``.

        Quasi-linear firm payoff with iso-elastic cost of scaling output,
        indexed by ability ``n`` (the firm's frictionless, no-tax optimum is
        ``y = n``)::

            u(y;n) = (1 - tau*1[y>=T*]) y - (n/(1+1/e)) (y/n)^(1+1/e).

        The marginal buncher ``n_H`` solves the indifference between bunching at
        ``T*`` (unregistered) and registering at its post-notch interior optimum
        ``y1 = n (1-tau)^e``. Its frictionless optimum ``y_H = n_H`` sits at the
        top of the bunching segment, so ``dy_star = y_H - T*``.
        """
        p = 1.0 + 1.0 / e
        t = self.t_star
        tau = self.tau

        def gap(n):
            u_bunch = t - (n / p) * (t / n) ** p
            y1 = n * (1.0 - tau) ** e
            u_tax = (1.0 - tau) * y1 - (n / p) * (y1 / n) ** p
            return u_bunch - u_tax

        n_lo = t * (1.0 + 1e-6)
        n_hi = t * 5.0
        # Expand the upper bracket if the root is not yet enclosed.
        while gap(n_lo) * gap(n_hi) > 0 and n_hi < t * 50:
            n_hi *= 1.5
        n_H = brentq(gap, n_lo, n_hi)
        return float(n_H), float(n_H - t)

    # -- Productivity recovery (documented diagnostic) ---------------------
    def recover_productivity(self, y_obs):
        """Cobb-Douglas FOC inversion of observed turnover to productivity ``A``.

        Unregistered (``y<T*``): ``A = y / (alpha)^(alpha/(1-alpha))``;
        registered (``y>=T*``):  ``A = y / (alpha*(1-tau))^(alpha/(1-alpha))``.

        DIAGNOSTIC ONLY. With ``alpha ~= 0.989`` the inversion exponent
        ``alpha/(1-alpha) ~= 90`` amplifies tiny output differences enormously,
        so the recovered ``A`` is numerically degenerate — the same near-CRS
        fragility that makes the taxed-branch interior optimum ill-defined. The
        headline behavioural object is the iso-elastic marginal buncher above.
        """
        y_obs = np.asarray(y_obs, dtype=float)
        net = np.where(y_obs >= self.t_star, 1.0 - self.tau, 1.0)
        return y_obs / (self.alpha * net) ** (self.alpha / (1.0 - self.alpha))

    def notch_predicted_optimum(self, ability, n_H, e: float = E_MEDIAN):
        """Forward-map a frictionless ability index through the notch decision.

        Each firm's ability ``n`` is its frictionless (no-tax) optimum:

        * ``n < T*``          : stay unregistered at ``n``;
        * ``T* <= n <= n_H``  : bunch at ``T*`` (dominated region stays empty);
        * ``n > n_H``         : register at the post-notch optimum ``n*(1-tau)^e``.
        """
        ability = np.asarray(ability, dtype=float)
        return np.where(
            ability < self.t_star,
            ability,
            np.where(ability <= n_H, self.t_star, ability * (1.0 - self.tau) ** e),
        )

    # -- Public results ----------------------------------------------------
    def summary(self) -> dict:
        """Return the headline notch quantities as a dictionary.

        Keys: ``vintage``, ``t_star``, ``a_kw`` (dominated-region width £k),
        ``region_lo``/``region_hi`` (dominated-region bounds £k), the marginal
        bunchers ``y_H_median``/``y_H_mean`` and their excess ``dy_median``/
        ``dy_mean`` at ``e=E_MEDIAN`` and ``e=E_MEAN`` respectively.
        """
        a_kw = self.dominated_region_width_kw()
        y_H_med, dy_med = self.marginal_buncher(E_MEDIAN)
        y_H_mean, dy_mean = self.marginal_buncher(E_MEAN)
        return {
            "vintage": self.vintage,
            "t_star": self.t_star,
            "tau": self.tau,
            "alpha": self.alpha,
            "a_kw": a_kw,
            "region_lo": self.t_star,
            "region_hi": self.t_star + a_kw,
            "e_median": E_MEDIAN,
            "e_mean": E_MEAN,
            "y_H_median": y_H_med,
            "y_H_mean": y_H_mean,
            "dy_median": dy_med,
            "dy_mean": dy_mean,
        }

    def implied_distribution(self, lo: float = 50.0, hi: float = 130.0):
        """Build the observed vs. notch-model-implied turnover distribution.

        Returns ``(centres, obs_hist, model_hist, a_kw, n_H)`` over £1k bins on
        ``[lo, hi]`` (£k), weighted by the synthetic firm weights. The observed
        turnover is treated as each firm's ability index and re-mapped through
        the notch decision rule (using ``e=E_MEDIAN``) to form the model-implied
        distribution.
        """
        df = self.firms
        tk = df["annual_turnover_k"].to_numpy(dtype=float)
        w = (
            df["weight"].to_numpy(dtype=float)
            if "weight" in df
            else np.ones(len(df))
        )
        window = (tk >= lo) & (tk <= hi) & (tk > 0)
        tk, w = tk[window], w[window]

        a_kw = self.dominated_region_width_kw()
        n_H, _ = self.marginal_buncher(E_MEDIAN)
        y_model = self.notch_predicted_optimum(tk, n_H)

        bins = np.arange(lo, hi + 0.5, 1.0)
        centres = 0.5 * (bins[:-1] + bins[1:])
        obs_hist, _ = np.histogram(tk, bins=bins, weights=w)
        model_hist, _ = np.histogram(y_model, bins=bins, weights=w)
        return centres, obs_hist, model_hist, a_kw, n_H


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    for vintage in ("2023-24", "2024-25"):
        m = NotchModel(vintage)
        s = m.summary()
        print(f"[{vintage}] T*=£{s['t_star']*1000:,.0f}  "
              f"a=£{s['a_kw']*1000:,.0f}  "
              f"region=(£{s['region_lo']*1000:,.0f}, £{s['region_hi']*1000:,.0f})")
        print(f"          y_H(e={E_MEDIAN})=£{s['y_H_median']*1000:,.0f}  "
              f"y_H(e={E_MEAN})=£{s['y_H_mean']*1000:,.0f}")
