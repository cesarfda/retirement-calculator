from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.accounts import apply_employer_match
from core.portfolio import Allocation, portfolio_returns
from utils.helpers import Percentiles


@dataclass
class SimulationResult:
    percentiles: Percentiles
    success_rate: float
    ending_balances: np.ndarray
    account_paths: np.ndarray


def _compute_percentiles(paths: np.ndarray) -> Percentiles:
    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    return Percentiles(
        p5=p5.tolist(),
        p25=p25.tolist(),
        p50=p50.tolist(),
        p75=p75.tolist(),
        p95=p95.tolist(),
    )


def run_simulation(
    initial_balances: dict[str, float],
    monthly_contributions: dict[str, float],
    allocation: Allocation,
    years: int,
    n_simulations: int,
    scenario: str,
    asset_returns: np.ndarray,
    retirement_months: int | None = None,
    withdrawal_rate: float = 0.04,
) -> SimulationResult:
    months = years * 12
    allocation = allocation.normalized()
    returns = portfolio_returns(asset_returns, allocation)

    balances = np.zeros((n_simulations, months + 1, 3))
    balances[:, 0, 0] = initial_balances.get("401k", 0.0)
    balances[:, 0, 1] = initial_balances.get("roth", 0.0)
    balances[:, 0, 2] = initial_balances.get("taxable", 0.0)

    monthly_401k = monthly_contributions.get("401k", 0.0)
    monthly_roth = monthly_contributions.get("roth", 0.0)
    monthly_taxable = monthly_contributions.get("taxable", 0.0)
    match_rate = monthly_contributions.get("employer_match_rate", 0.0)
    match_cap = monthly_contributions.get("employer_match_cap", 0.0)

    for month in range(months):
        balances[:, month, :] = balances[:, month, :] * (1 + returns[:, month][:, None])

        if retirement_months is None or month < retirement_months:
            employer_match = apply_employer_match(monthly_401k, match_rate, match_cap)
            balances[:, month, 0] += monthly_401k + employer_match
            balances[:, month, 1] += monthly_roth
            balances[:, month, 2] += monthly_taxable
        else:
            total_balance = balances[:, month, :].sum(axis=1)
            withdrawal = (withdrawal_rate / 12) * total_balance
            withdrawal = np.minimum(withdrawal, total_balance)
            proportions = np.divide(
                balances[:, month, :],
                total_balance[:, None],
                out=np.zeros_like(balances[:, month, :]),
                where=total_balance[:, None] > 0,
            )
            balances[:, month, :] -= withdrawal[:, None] * proportions

        balances[:, month + 1, :] = balances[:, month, :]

    total_paths = balances.sum(axis=2)
    percentiles = _compute_percentiles(total_paths)
    ending_balances = total_paths[:, -1]
    success_rate = float(np.mean(ending_balances > 0))

    return SimulationResult(
        percentiles=percentiles,
        success_rate=success_rate,
        ending_balances=ending_balances,
        account_paths=balances,
    )
