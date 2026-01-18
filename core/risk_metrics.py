"""Enhanced risk metrics for retirement simulation analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LegacyMetrics:
    """
    Metrics for legacy/bequest planning.

    For retirees who want to leave money to heirs, these metrics
    provide probability estimates for various legacy amounts.

    Attributes:
        prob_leave_100k: Probability of ending with > $100k
        prob_leave_500k: Probability of ending with > $500k
        prob_leave_1m: Probability of ending with > $1M
        expected_legacy: Expected (mean) ending balance
        median_legacy: Median ending balance
        legacy_at_risk: 10th percentile (conservative legacy estimate)
    """

    prob_leave_100k: float
    prob_leave_500k: float
    prob_leave_1m: float
    expected_legacy: float
    median_legacy: float
    legacy_at_risk: float  # 10th percentile


@dataclass(frozen=True)
class SpendingFlexibilityResult:
    """
    Results from spending flexibility analysis.

    Measures how much success rate improves if retiree can
    reduce spending during market downturns.

    Attributes:
        base_success_rate: Success rate with fixed spending
        flexible_success_rate: Success rate with flexible spending
        improvement: Percentage point improvement
        avg_spending_ratio: Average actual spending vs target (< 1 if cuts occurred)
        years_with_cuts: Average number of years with spending cuts
    """

    base_success_rate: float
    flexible_success_rate: float
    improvement: float
    avg_spending_ratio: float
    years_with_cuts: float


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
    # New enhanced metrics
    cvar_95: float = 0.0  # Conditional VaR at 95% confidence
    cvar_99: float = 0.0  # Conditional VaR at 99% confidence
    legacy_metrics: LegacyMetrics | None = None
    spending_flexibility: SpendingFlexibilityResult | None = None


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


def calculate_cvar(
    ending_balances: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR measures the expected loss in the worst (1-confidence)% of scenarios.
    It's more informative than VaR because it captures the severity of tail losses,
    not just whether they exceed a threshold.

    Example: CVaR at 95% confidence tells you the average ending balance
    in the worst 5% of simulations.

    Args:
        ending_balances: Array of ending balances from simulation
        confidence: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Average ending balance in worst (1-confidence)% of scenarios
    """
    if len(ending_balances) == 0:
        return 0.0

    # VaR is the threshold at the (1-confidence) percentile
    var_percentile = (1 - confidence) * 100
    var_threshold = float(np.percentile(ending_balances, var_percentile))

    # CVaR is the average of all values at or below VaR
    tail_values = ending_balances[ending_balances <= var_threshold]

    if len(tail_values) == 0:
        return var_threshold

    return float(np.mean(tail_values))


def calculate_legacy_metrics(
    ending_balances: np.ndarray,
    thresholds: list[float] | None = None,
) -> LegacyMetrics:
    """
    Calculate legacy/bequest probability metrics.

    For retirees who want to leave money to heirs, these metrics
    show the probability of leaving various amounts.

    Args:
        ending_balances: Array of ending balances from simulation
        thresholds: Custom thresholds (default: 100k, 500k, 1M)

    Returns:
        LegacyMetrics with probabilities and expected values
    """
    if thresholds is None:
        thresholds = [100_000, 500_000, 1_000_000]

    n_sims = len(ending_balances)

    # Probability of leaving various amounts
    prob_100k = float(np.mean(ending_balances > thresholds[0])) if len(thresholds) > 0 else 0.0
    prob_500k = float(np.mean(ending_balances > thresholds[1])) if len(thresholds) > 1 else 0.0
    prob_1m = float(np.mean(ending_balances > thresholds[2])) if len(thresholds) > 2 else 0.0

    # Expected and median legacy (only count positive endings)
    positive_endings = ending_balances[ending_balances > 0]
    if len(positive_endings) > 0:
        expected_legacy = float(np.mean(positive_endings))
        median_legacy = float(np.median(positive_endings))
    else:
        expected_legacy = 0.0
        median_legacy = 0.0

    # Legacy at risk (10th percentile - conservative estimate)
    legacy_at_risk = float(np.percentile(ending_balances, 10))

    return LegacyMetrics(
        prob_leave_100k=prob_100k,
        prob_leave_500k=prob_500k,
        prob_leave_1m=prob_1m,
        expected_legacy=expected_legacy,
        median_legacy=median_legacy,
        legacy_at_risk=legacy_at_risk,
    )


def estimate_spending_flexibility_impact(
    total_paths: np.ndarray,
    base_success_rate: float,
    withdrawal_rate: float,
    reduction_amount: float = 0.10,
    reduction_trigger: float = 0.15,
) -> SpendingFlexibilityResult:
    """
    Estimate how spending flexibility improves retirement outcomes.

    Many retirees can reduce spending during market downturns.
    This estimates the improvement in success rate if spending
    can be reduced when portfolio drops significantly.

    This is a simplified estimate that doesn't re-run the simulation,
    but uses heuristics based on the path characteristics.

    Args:
        total_paths: Portfolio value paths (n_sims, n_months+1)
        base_success_rate: Success rate with fixed spending
        withdrawal_rate: Annual withdrawal rate
        reduction_amount: How much spending can be reduced (e.g., 0.10 = 10%)
        reduction_trigger: Portfolio drop that triggers reduction (e.g., 0.15 = 15%)

    Returns:
        SpendingFlexibilityResult with estimated improvement
    """
    n_sims, n_months_plus_1 = total_paths.shape
    n_months = n_months_plus_1 - 1

    # Identify simulations that failed
    failed_sims = total_paths[:, -1] <= 0
    n_failed = np.sum(failed_sims)

    if n_failed == 0:
        # No failures - flexibility doesn't help
        return SpendingFlexibilityResult(
            base_success_rate=base_success_rate,
            flexible_success_rate=base_success_rate,
            improvement=0.0,
            avg_spending_ratio=1.0,
            years_with_cuts=0.0,
        )

    # For failed simulations, estimate how many could be saved
    # by spending cuts during drawdowns

    # Calculate max drawdown for failed simulations
    failed_paths = total_paths[failed_sims]
    saved_count = 0
    total_years_with_cuts = 0
    total_spending_ratio = 0.0

    for path in failed_paths:
        # Find months where portfolio dropped significantly
        peak = np.maximum.accumulate(path)
        drawdown = (peak - path) / np.maximum(peak, 1)

        # Count months where cuts would trigger
        cut_months = np.sum(drawdown > reduction_trigger)
        years_with_cuts = cut_months / 12

        # Estimate savings from cuts
        # Each year of 10% cuts saves roughly 10% of annual withdrawal
        annual_withdrawal_value = path[0] * withdrawal_rate
        total_saved = years_with_cuts * reduction_amount * annual_withdrawal_value

        # If saved amount > shortfall at end, this simulation could succeed
        shortfall = abs(path[-1])
        if total_saved > shortfall * 0.5:  # Conservative: need to save 50% more than shortfall
            saved_count += 1

        total_years_with_cuts += years_with_cuts
        total_spending_ratio += 1.0 - (years_with_cuts * reduction_amount / (n_months / 12))

    # Calculate improvement
    new_successes = saved_count
    flexible_success_rate = base_success_rate + (new_successes / n_sims)
    flexible_success_rate = min(flexible_success_rate, 1.0)  # Cap at 100%

    improvement = flexible_success_rate - base_success_rate

    # Average spending ratio and years with cuts (across all sims)
    avg_spending_ratio = total_spending_ratio / max(n_failed, 1)
    avg_years_with_cuts = total_years_with_cuts / max(n_failed, 1)

    return SpendingFlexibilityResult(
        base_success_rate=base_success_rate,
        flexible_success_rate=flexible_success_rate,
        improvement=improvement,
        avg_spending_ratio=avg_spending_ratio,
        years_with_cuts=avg_years_with_cuts,
    )


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

    # Calculate CVaR (Expected Shortfall)
    cvar_95 = calculate_cvar(ending_balances, confidence=0.95)
    cvar_99 = calculate_cvar(ending_balances, confidence=0.99)

    # Calculate legacy metrics
    legacy = calculate_legacy_metrics(ending_balances)

    # Calculate spending flexibility impact (simplified estimate)
    flexibility = estimate_spending_flexibility_impact(
        total_paths=total_paths,
        base_success_rate=success_rate,
        withdrawal_rate=annual_withdrawal / initial_balance if initial_balance > 0 else 0.04,
    )

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
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        legacy_metrics=legacy,
        spending_flexibility=flexibility,
    )
