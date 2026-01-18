"""Enhanced risk metrics for retirement simulation analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskMetrics:
    """
    Comprehensive risk metrics for retirement simulation results.

    Attributes:
        success_rate: Percentage of simulations ending with balance > 0
        perfect_withdrawal_rate: Highest sustainable withdrawal rate (estimated)
        average_shortfall: Average ending deficit when simulation fails
        median_years_of_income: Median ending balance / annual withdrawal amount
        worst_case_ending: 5th percentile ending balance
        best_case_ending: 95th percentile ending balance
        max_drawdown_median: Median of maximum drawdown across simulations
        max_drawdown_worst: 95th percentile of maximum drawdown (worst cases)
        probability_of_ruin_by_year: Array of ruin probability at each year
    """

    success_rate: float
    perfect_withdrawal_rate: float
    average_shortfall: float
    median_years_of_income: float
    worst_case_ending: float
    best_case_ending: float
    max_drawdown_median: float
    max_drawdown_worst: float
    probability_of_ruin_by_year: list[float]


def calculate_max_drawdown(path: np.ndarray) -> float:
    """
    Calculate maximum drawdown for a single simulation path.

    Args:
        path: 1D array of portfolio values over time

    Returns:
        Maximum drawdown as a positive percentage (0.20 = 20% drawdown)
    """
    if len(path) == 0:
        return 0.0

    # Running maximum
    running_max = np.maximum.accumulate(path)

    # Avoid division by zero
    running_max = np.where(running_max == 0, 1, running_max)

    # Drawdown at each point
    drawdowns = (running_max - path) / running_max

    return float(np.max(drawdowns))


def calculate_max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """
    Calculate maximum drawdown for each simulation path.

    Args:
        paths: 2D array of shape (n_simulations, n_months+1)

    Returns:
        1D array of maximum drawdown per simulation
    """
    n_simulations = paths.shape[0]
    drawdowns = np.zeros(n_simulations)

    for i in range(n_simulations):
        drawdowns[i] = calculate_max_drawdown(paths[i])

    return drawdowns


def estimate_perfect_withdrawal_rate(
    ending_balances: np.ndarray,
    initial_balance: float,
    years: int,
    target_success_rate: float = 0.95,
) -> float:
    """
    Estimate the highest sustainable withdrawal rate for a target success rate.

    This is a simplified estimate based on the distribution of ending balances.
    A more accurate method would require re-running simulations at different rates.

    Args:
        ending_balances: Array of ending balances from simulation
        initial_balance: Starting portfolio value
        years: Number of years in simulation
        target_success_rate: Desired success rate (default 95%)

    Returns:
        Estimated sustainable annual withdrawal rate
    """
    if initial_balance <= 0 or years <= 0:
        return 0.0

    # Get the percentile that corresponds to target success rate
    # e.g., 95% success means looking at 5th percentile outcome
    percentile = (1 - target_success_rate) * 100
    safe_ending = float(np.percentile(ending_balances, percentile))

    # If even the safe ending is negative, no withdrawal rate is safe
    if safe_ending < 0:
        return 0.0

    # Estimate: what annual withdrawal could you take from initial_balance
    # and still end with safe_ending after 'years' years?
    # Using a simplified constant-dollar withdrawal model:
    # If portfolio grows and you withdraw W per year for Y years,
    # ending = initial + gains - W*Y
    # So W = (initial + gains - ending) / Y = (initial - ending + gains) / Y
    # Since we don't know gains explicitly, we estimate based on
    # the ratio of ending to initial for the safe case.
    # If ending > initial, the portfolio grew enough to support withdrawals.
    # A simple heuristic: (safe_ending / years) gives sustainable annual draw
    # But we want rate relative to initial, so:
    # Sustainable rate ≈ (safe_ending / initial) / years as a baseline
    # Plus the "4% rule" style estimate based on portfolio size
    
    # More practical approach: what withdrawal rate would deplete
    # the portfolio to zero given the safe_ending trajectory?
    # If safe_ending > 0, total available = initial + growth - safe_ending
    # Since we don't track growth, use: available ≈ safe_ending (what's left)
    # means we could have withdrawn more. Estimate as safe_ending/years + baseline
    
    # Simplified heuristic based on ending balance:
    # If you ended with safe_ending, you could have withdrawn approximately
    # safe_ending / years more per year than you did (assuming no withdrawals in sim)
    sustainable_annual = safe_ending / years
    
    return sustainable_annual / initial_balance


def calculate_risk_metrics(
    total_paths: np.ndarray,
    ending_balances: np.ndarray,
    annual_withdrawal: float,
    initial_balance: float,
    years: int,
) -> RiskMetrics:
    """
    Calculate comprehensive risk metrics from simulation results.

    Args:
        total_paths: 2D array of shape (n_simulations, n_months+1) with total portfolio values
        ending_balances: 1D array of ending balances
        annual_withdrawal: Annual withdrawal amount in dollars
        initial_balance: Initial total portfolio balance
        years: Number of years simulated

    Returns:
        RiskMetrics with all calculated metrics
    """
    n_simulations = total_paths.shape[0]
    n_months = total_paths.shape[1] - 1

    # Basic success rate
    success_rate = float(np.mean(ending_balances > 0))

    # Perfect withdrawal rate estimate
    pwr = estimate_perfect_withdrawal_rate(
        ending_balances, initial_balance, years, target_success_rate=0.95
    )

    # Average shortfall (only for failed simulations)
    failed_mask = ending_balances <= 0
    if np.any(failed_mask):
        average_shortfall = float(np.mean(np.abs(ending_balances[failed_mask])))
    else:
        average_shortfall = 0.0

    # Years of income remaining
    if annual_withdrawal > 0:
        years_of_income = ending_balances / annual_withdrawal
        median_years_of_income = float(np.median(years_of_income))
    else:
        median_years_of_income = float("inf") if success_rate > 0 else 0.0

    # Ending balance percentiles
    worst_case_ending = float(np.percentile(ending_balances, 5))
    best_case_ending = float(np.percentile(ending_balances, 95))

    # Maximum drawdown statistics
    max_drawdowns = calculate_max_drawdowns(total_paths)
    max_drawdown_median = float(np.median(max_drawdowns))
    max_drawdown_worst = float(np.percentile(max_drawdowns, 95))

    # Probability of ruin by year (at each 12-month interval)
    n_years = years
    prob_ruin_by_year = []
    for year in range(1, n_years + 1):
        month_idx = min(year * 12, n_months)
        balances_at_year = total_paths[:, month_idx]
        ruin_rate = float(np.mean(balances_at_year <= 0))
        prob_ruin_by_year.append(ruin_rate)

    return RiskMetrics(
        success_rate=success_rate,
        perfect_withdrawal_rate=pwr,
        average_shortfall=average_shortfall,
        median_years_of_income=median_years_of_income,
        worst_case_ending=worst_case_ending,
        best_case_ending=best_case_ending,
        max_drawdown_median=max_drawdown_median,
        max_drawdown_worst=max_drawdown_worst,
        probability_of_ruin_by_year=prob_ruin_by_year,
    )
