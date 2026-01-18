"""Monte Carlo simulation engine for retirement planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.accounts import apply_employer_match
from core.portfolio import Allocation, portfolio_returns
from utils.helpers import Percentiles, annual_to_monthly_rate

if TYPE_CHECKING:
    from core.glide_path import GlidePath


@dataclass(frozen=True)
class AccountBalances:
    """Initial account balances."""

    balance_401k: float = 0.0
    balance_roth: float = 0.0
    balance_taxable: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "AccountBalances":
        """Create from dictionary (backward compatibility)."""
        return cls(
            balance_401k=data.get("401k", 0.0),
            balance_roth=data.get("roth", 0.0),
            balance_taxable=data.get("taxable", 0.0),
        )

    def total(self) -> float:
        """Return total balance across all accounts."""
        return self.balance_401k + self.balance_roth + self.balance_taxable

    def as_array(self) -> np.ndarray:
        """Return balances as numpy array [401k, roth, taxable]."""
        return np.array([self.balance_401k, self.balance_roth, self.balance_taxable])


@dataclass(frozen=True)
class MonthlyContributions:
    """Monthly contribution amounts and employer match settings."""

    contrib_401k: float = 0.0
    contrib_roth: float = 0.0
    contrib_taxable: float = 0.0
    employer_match_rate: float = 0.0
    employer_match_cap: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "MonthlyContributions":
        """Create from dictionary (backward compatibility)."""
        return cls(
            contrib_401k=data.get("401k", 0.0),
            contrib_roth=data.get("roth", 0.0),
            contrib_taxable=data.get("taxable", 0.0),
            employer_match_rate=data.get("employer_match_rate", 0.0),
            employer_match_cap=data.get("employer_match_cap", 0.0),
        )

    def scaled(self, factor: float) -> "MonthlyContributions":
        """Return contributions scaled by a factor (e.g., for soft retirement)."""
        return MonthlyContributions(
            contrib_401k=self.contrib_401k * factor,
            contrib_roth=self.contrib_roth * factor,
            contrib_taxable=self.contrib_taxable * factor,
            employer_match_rate=self.employer_match_rate,
            employer_match_cap=self.employer_match_cap,
        )


@dataclass(frozen=True)
class Guardrails:
    """
    Guardrails withdrawal strategy that adjusts spending based on portfolio performance.

    When portfolio value rises significantly above target, increase withdrawals (ceiling).
    When portfolio value drops significantly below target, decrease withdrawals (floor).

    Attributes:
        enabled: Whether guardrails are active
        ceiling: Maximum increase factor (e.g., 1.10 = 10% raise max)
        floor: Maximum decrease factor (e.g., 0.95 = 5% cut max)
        upper_threshold: Portfolio ratio above which ceiling kicks in (e.g., 1.20)
        lower_threshold: Portfolio ratio below which floor kicks in (e.g., 0.80)
    """

    enabled: bool = False
    ceiling: float = 1.10
    floor: float = 0.95
    upper_threshold: float = 1.20
    lower_threshold: float = 0.80

    def apply(
        self,
        base_withdrawal: float,
        current_balance: float,
        target_balance: float,
    ) -> float:
        """
        Adjust withdrawal amount based on portfolio performance.

        Args:
            base_withdrawal: Baseline withdrawal amount
            current_balance: Current portfolio value
            target_balance: Expected/target portfolio value

        Returns:
            Adjusted withdrawal amount
        """
        if not self.enabled or target_balance <= 0:
            return base_withdrawal

        ratio = current_balance / target_balance

        if ratio >= self.upper_threshold:
            # Doing well - can increase spending
            return base_withdrawal * self.ceiling
        elif ratio <= self.lower_threshold:
            # Struggling - reduce spending
            return base_withdrawal * self.floor
        else:
            # Within normal range
            return base_withdrawal


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a retirement simulation."""

    years: int
    n_simulations: int
    retirement_months: int | None = None
    soft_retirement_months: int | None = None
    soft_contribution_factor: float = 1.0
    withdrawal_rate: float = 0.04
    expense_ratio: float = 0.001  # Default 0.1% annual expense ratio
    guardrails: Guardrails | None = None

    @property
    def total_months(self) -> int:
        """Total simulation length in months."""
        return self.years * 12

    @property
    def monthly_expense_drag(self) -> float:
        """Monthly expense ratio drag on returns."""
        return annual_to_monthly_rate(self.expense_ratio)


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation."""

    percentiles: Percentiles
    success_rate: float
    ending_balances: np.ndarray
    account_paths: np.ndarray
    total_paths: np.ndarray  # Added for risk metrics

    @property
    def initial_balance(self) -> float:
        """Initial total balance."""
        return float(self.account_paths[:, 0, :].sum(axis=1).mean())


def _compute_percentiles(paths: np.ndarray) -> Percentiles:
    """Compute percentile statistics across simulation paths."""
    p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    return Percentiles(
        p5=p5.tolist(),
        p25=p25.tolist(),
        p50=p50.tolist(),
        p75=p75.tolist(),
        p95=p95.tolist(),
    )


def _apply_contributions(
    balances: np.ndarray,
    contributions: MonthlyContributions,
) -> None:
    """Apply monthly contributions to account balances (in-place)."""
    employer_match = apply_employer_match(
        contributions.contrib_401k,
        contributions.employer_match_rate,
        contributions.employer_match_cap,
    )
    balances[:, 0] += contributions.contrib_401k + employer_match
    balances[:, 1] += contributions.contrib_roth
    balances[:, 2] += contributions.contrib_taxable


def _apply_withdrawals(
    balances: np.ndarray,
    withdrawal_rate: float,
    guardrails: Guardrails | None,
    target_balances: np.ndarray | None,
) -> None:
    """
    Apply withdrawals from accounts proportionally (in-place).

    Implements SORR-aware withdrawal: withdrawals happen at start of period
    before returns are applied.
    """
    total_balance = balances.sum(axis=1)
    base_withdrawal = (withdrawal_rate / 12) * total_balance

    # Apply guardrails if enabled
    if guardrails is not None and guardrails.enabled and target_balances is not None:
        # Vectorized guardrails application
        target_total = target_balances.sum(axis=1)
        ratio = np.divide(
            total_balance,
            target_total,
            out=np.ones_like(total_balance),
            where=target_total > 0,
        )

        # Apply ceiling/floor based on thresholds
        adjustment = np.ones_like(ratio)
        adjustment = np.where(ratio >= guardrails.upper_threshold, guardrails.ceiling, adjustment)
        adjustment = np.where(ratio <= guardrails.lower_threshold, guardrails.floor, adjustment)
        withdrawal = base_withdrawal * adjustment
    else:
        withdrawal = base_withdrawal

    # Cap withdrawal at available balance
    withdrawal = np.minimum(withdrawal, total_balance)

    # Calculate proportional withdrawal from each account
    proportions = np.divide(
        balances,
        total_balance[:, None],
        out=np.zeros_like(balances),
        where=total_balance[:, None] > 0,
    )
    balances -= withdrawal[:, None] * proportions


def run_simulation(
    initial_balances: dict[str, float] | AccountBalances,
    monthly_contributions: dict[str, float] | MonthlyContributions,
    allocation: Allocation,
    years: int,
    n_simulations: int,
    scenario: str,
    asset_returns: np.ndarray,
    retirement_months: int | None = None,
    soft_retirement_months: int | None = None,
    soft_contribution_factor: float = 1.0,
    withdrawal_rate: float = 0.04,
    expense_ratio: float = 0.001,
    guardrails: Guardrails | None = None,
    glide_path: "GlidePath | None" = None,
) -> SimulationResult:
    """
    Run Monte Carlo retirement simulation.

    Args:
        initial_balances: Starting account balances (dict or AccountBalances)
        monthly_contributions: Monthly contribution amounts (dict or MonthlyContributions)
        allocation: Asset allocation (used if glide_path is None)
        years: Number of years to simulate
        n_simulations: Number of simulation paths
        scenario: Market scenario name (for logging/info)
        asset_returns: Pre-sampled asset returns array (n_sims, n_months, n_assets)
        retirement_months: Month when retirement begins (None = never retire)
        soft_retirement_months: Month when soft retirement begins (reduced contributions)
        soft_contribution_factor: Contribution multiplier after soft retirement
        withdrawal_rate: Annual withdrawal rate in retirement (e.g., 0.04 = 4%)
        expense_ratio: Annual fund expense ratio (e.g., 0.001 = 0.1%)
        guardrails: Optional guardrails withdrawal strategy
        glide_path: Optional glide path for dynamic allocation

    Returns:
        SimulationResult with paths, percentiles, and success metrics
    """
    # Convert dicts to dataclasses if needed (backward compatibility)
    if isinstance(initial_balances, dict):
        initial_balances = AccountBalances.from_dict(initial_balances)
    if isinstance(monthly_contributions, dict):
        monthly_contributions = MonthlyContributions.from_dict(monthly_contributions)

    months = years * 12
    config = SimulationConfig(
        years=years,
        n_simulations=n_simulations,
        retirement_months=retirement_months,
        soft_retirement_months=soft_retirement_months,
        soft_contribution_factor=soft_contribution_factor,
        withdrawal_rate=withdrawal_rate,
        expense_ratio=expense_ratio,
        guardrails=guardrails,
    )

    # Initialize balance arrays: (n_simulations, months + 1, 3 accounts)
    balances = np.zeros((n_simulations, months + 1, 3))
    balances[:, 0, :] = initial_balances.as_array()

    # Pre-compute monthly expense drag
    expense_drag = config.monthly_expense_drag

    # Track "target" balances for guardrails (based on expected path without volatility)
    # This is simplified - using median of previous month as target
    target_balances: np.ndarray | None = None

    for month in range(months):
        # Get allocation for this month (static or from glide path)
        if glide_path is not None:
            month_allocation = glide_path.get_allocation_at_month(
                month,
                retirement_months or months,
                months,
            )
        else:
            month_allocation = allocation.normalized()

        # Calculate portfolio returns for this month with this allocation
        month_returns = portfolio_returns(
            asset_returns[:, month : month + 1, :],
            month_allocation,
        )[:, 0]

        # Apply expense ratio drag
        effective_returns = month_returns - expense_drag

        is_retired = retirement_months is not None and month >= retirement_months
        is_soft_retired = (
            soft_retirement_months is not None and month >= soft_retirement_months
        )

        if is_retired:
            # RETIREMENT PHASE: Apply withdrawals FIRST (SORR-aware)
            # Then apply returns to remaining balance

            # Use previous month's ending balance as target for guardrails
            if month > 0:
                target_balances = balances[:, month, :].copy()

            _apply_withdrawals(
                balances[:, month, :],
                withdrawal_rate,
                guardrails,
                target_balances,
            )

            # Apply returns after withdrawal
            balances[:, month, :] *= 1 + effective_returns[:, None]

        else:
            # ACCUMULATION PHASE: Apply returns first, then add contributions
            balances[:, month, :] *= 1 + effective_returns[:, None]

            # Determine contribution amount
            if is_soft_retired:
                effective_contributions = monthly_contributions.scaled(
                    soft_contribution_factor
                )
            else:
                effective_contributions = monthly_contributions

            _apply_contributions(balances[:, month, :], effective_contributions)

        # Copy to next month's starting balance
        balances[:, month + 1, :] = balances[:, month, :]

    # Compute results
    total_paths = balances.sum(axis=2)
    percentiles = _compute_percentiles(total_paths)
    ending_balances = total_paths[:, -1]
    success_rate = float(np.mean(ending_balances > 0))

    return SimulationResult(
        percentiles=percentiles,
        success_rate=success_rate,
        ending_balances=ending_balances,
        account_paths=balances,
        total_paths=total_paths,
    )
