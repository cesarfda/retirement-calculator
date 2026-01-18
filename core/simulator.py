"""Monte Carlo simulation engine for retirement planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from core.accounts import apply_employer_match
from core.portfolio import Allocation, portfolio_returns
from core.tax_config import CURRENT_IRS_LIMITS, IRSLimits, calculate_rmd
from utils.helpers import Percentiles, annual_to_monthly_rate

if TYPE_CHECKING:
    from core.glide_path import GlidePath
    from core.taxes import FilingStatus


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

    def scaled_by_factors(self, factors: dict[str, float]) -> "MonthlyContributions":
        """Return contributions scaled by per-account factors."""
        return MonthlyContributions(
            contrib_401k=self.contrib_401k * factors.get("401k", 1.0),
            contrib_roth=self.contrib_roth * factors.get("roth", 1.0),
            contrib_taxable=self.contrib_taxable * factors.get("taxable", 1.0),
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
class TaxConfig:
    """
    Tax configuration for simulation.

    Attributes:
        enabled: Whether to model taxes on withdrawals
        filing_status: Tax filing status (single or married)
        cost_basis_ratio: Fraction of taxable account that is cost basis
        enforce_rmd: Whether to enforce Required Minimum Distributions
        enforce_contribution_limits: Whether to enforce IRS contribution limits
        tax_efficient_withdrawal: Whether to use tax-efficient withdrawal ordering
        irs_limits: IRS limits to use (defaults to current year)
    """

    enabled: bool = False
    filing_status: str = "single"  # "single" or "mfj"
    cost_basis_ratio: float = 0.5
    enforce_rmd: bool = True
    enforce_contribution_limits: bool = True
    tax_efficient_withdrawal: bool = False
    irs_limits: IRSLimits = CURRENT_IRS_LIMITS


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a retirement simulation."""

    years: int
    n_simulations: int
    current_age: int = 35
    retirement_months: int | None = None
    soft_retirement_months: int | None = None
    soft_contribution_factor: float = 1.0
    soft_contribution_factors: dict[str, float] | None = None
    withdrawal_rate: float = 0.04
    expense_ratio: float = 0.001  # Default 0.1% annual expense ratio
    guardrails: Guardrails | None = None
    tax_config: TaxConfig | None = None

    @property
    def total_months(self) -> int:
        """Total simulation length in months."""
        return self.years * 12

    @property
    def monthly_expense_drag(self) -> float:
        """Monthly expense ratio drag on returns."""
        return annual_to_monthly_rate(self.expense_ratio)

    def age_at_month(self, month: int) -> float:
        """Calculate age at a given month."""
        return self.current_age + month / 12


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
    ytd_contributions: dict[str, float] | None = None,
    age: float = 35,
    irs_limits: IRSLimits | None = None,
    enforce_limits: bool = False,
) -> dict[str, float]:
    """
    Apply monthly contributions to account balances (in-place).

    Args:
        balances: Account balances array to modify
        contributions: Monthly contribution amounts
        ytd_contributions: Year-to-date contributions for limit tracking
        age: Current age for catch-up eligibility
        irs_limits: IRS limits to enforce
        enforce_limits: Whether to enforce contribution limits

    Returns:
        Updated ytd_contributions dict
    """
    if ytd_contributions is None:
        ytd_contributions = {"401k": 0.0, "roth": 0.0}

    effective_401k = contributions.contrib_401k
    effective_roth = contributions.contrib_roth

    if enforce_limits and irs_limits is not None:
        # Enforce 401k limit (including catch-up if age >= 50)
        annual_401k_limit = irs_limits.get_401k_limit(int(age))
        remaining_401k = max(0, annual_401k_limit - ytd_contributions["401k"])
        effective_401k = min(contributions.contrib_401k, remaining_401k)

        # Enforce Roth IRA limit (including catch-up if age >= 50)
        annual_roth_limit = irs_limits.get_roth_ira_limit(int(age))
        remaining_roth = max(0, annual_roth_limit - ytd_contributions["roth"])
        effective_roth = min(contributions.contrib_roth, remaining_roth)

    employer_match = apply_employer_match(
        effective_401k,
        contributions.employer_match_rate,
        contributions.employer_match_cap,
    )

    balances[:, 0] += effective_401k + employer_match
    balances[:, 1] += effective_roth
    balances[:, 2] += contributions.contrib_taxable  # No limit on taxable

    # Update YTD tracking
    ytd_contributions["401k"] += effective_401k
    ytd_contributions["roth"] += effective_roth

    return ytd_contributions


def _apply_withdrawals_tax_aware(
    balances: np.ndarray,
    withdrawal_rate: float,
    age: float,
    irs_limits: IRSLimits | None = None,
    cost_basis_ratio: float = 0.5,
    tax_efficient: bool = False,
) -> None:
    """
    Apply withdrawals with RMD enforcement and optional tax-efficient ordering.

    Args:
        balances: Account balances array (n_sims, 3) - modified in-place
        withdrawal_rate: Annual withdrawal rate
        age: Current age for RMD calculation
        irs_limits: IRS limits for RMD start age
        cost_basis_ratio: Fraction of taxable that is cost basis
        tax_efficient: Whether to use tax-efficient withdrawal order
    """
    n_sims = balances.shape[0]
    total_balance = balances.sum(axis=1)
    monthly_withdrawal = (withdrawal_rate / 12) * total_balance

    # Calculate RMD if applicable
    rmd_start_age = irs_limits.rmd_start_age if irs_limits else 73
    monthly_rmd = np.zeros(n_sims)

    if int(age) >= rmd_start_age:
        # RMD is calculated on prior year-end balance, but we approximate
        # with current 401k balance divided by 12 for monthly
        annual_rmd = np.array([
            calculate_rmd(balances[i, 0], int(age), rmd_start_age)
            for i in range(n_sims)
        ])
        monthly_rmd = annual_rmd / 12

    if tax_efficient:
        # Tax-efficient withdrawal order:
        # 1. Take RMD from 401k first (mandatory)
        # 2. Withdraw from taxable (capital gains rates)
        # 3. Withdraw from 401k (ordinary income)
        # 4. Withdraw from Roth last (tax-free)

        withdrawal_401k = np.zeros(n_sims)
        withdrawal_roth = np.zeros(n_sims)
        withdrawal_taxable = np.zeros(n_sims)
        remaining_need = monthly_withdrawal.copy()

        # Step 1: Take RMD from 401k
        rmd_withdrawal = np.minimum(monthly_rmd, balances[:, 0])
        withdrawal_401k += rmd_withdrawal
        remaining_need -= rmd_withdrawal

        # Step 2: Withdraw from taxable
        taxable_available = balances[:, 2]
        taxable_withdrawal = np.minimum(remaining_need, taxable_available)
        withdrawal_taxable = taxable_withdrawal
        remaining_need -= taxable_withdrawal

        # Step 3: Withdraw more from 401k if needed
        additional_401k_available = balances[:, 0] - withdrawal_401k
        additional_401k = np.minimum(remaining_need, additional_401k_available)
        withdrawal_401k += additional_401k
        remaining_need -= additional_401k

        # Step 4: Withdraw from Roth as last resort
        roth_available = balances[:, 1]
        roth_withdrawal = np.minimum(remaining_need, roth_available)
        withdrawal_roth = roth_withdrawal

        # Apply withdrawals
        balances[:, 0] -= withdrawal_401k
        balances[:, 1] -= withdrawal_roth
        balances[:, 2] -= withdrawal_taxable

    else:
        # Proportional withdrawal (original behavior) but ensure RMD is met
        # Ensure minimum RMD is withdrawn from 401k
        rmd_met = np.minimum(monthly_rmd, balances[:, 0])

        # Calculate remaining withdrawal needed
        remaining_withdrawal = np.maximum(0, monthly_withdrawal - rmd_met)

        # Proportional withdrawal for the rest
        remaining_balance = balances.copy()
        remaining_balance[:, 0] -= rmd_met  # Already taking RMD from 401k

        remaining_total = remaining_balance.sum(axis=1)
        proportions = np.divide(
            remaining_balance,
            remaining_total[:, None],
            out=np.zeros_like(remaining_balance),
            where=remaining_total[:, None] > 0,
        )

        additional_withdrawal = remaining_withdrawal[:, None] * proportions
        additional_withdrawal = np.minimum(additional_withdrawal, remaining_balance)

        # Apply RMD from 401k
        balances[:, 0] -= rmd_met
        # Apply proportional withdrawal from remaining
        balances -= additional_withdrawal


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
    soft_contribution_factors: dict[str, float] | None = None,
    withdrawal_rate: float = 0.04,
    expense_ratio: float = 0.001,
    guardrails: Guardrails | None = None,
    glide_path: "GlidePath | None" = None,
    current_age: int = 35,
    tax_config: TaxConfig | None = None,
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
        soft_contribution_factors: Optional per-account contribution multipliers
        withdrawal_rate: Annual withdrawal rate in retirement (e.g., 0.04 = 4%)
        expense_ratio: Annual fund expense ratio (e.g., 0.001 = 0.1%)
        guardrails: Optional guardrails withdrawal strategy
        glide_path: Optional glide path for dynamic allocation
        current_age: Current age at simulation start (for RMD and catch-up)
        tax_config: Optional tax configuration for RMD and contribution limits

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
        current_age=current_age,
        retirement_months=retirement_months,
        soft_retirement_months=soft_retirement_months,
        soft_contribution_factor=soft_contribution_factor,
        soft_contribution_factors=soft_contribution_factors,
        withdrawal_rate=withdrawal_rate,
        expense_ratio=expense_ratio,
        guardrails=guardrails,
        tax_config=tax_config,
    )

    # Initialize balance arrays: (n_simulations, months + 1, 3 accounts)
    balances = np.zeros((n_simulations, months + 1, 3))
    balances[:, 0, :] = initial_balances.as_array()

    # Pre-compute monthly expense drag
    expense_drag = config.monthly_expense_drag

    # Track "target" balances for guardrails (based on expected path without volatility)
    # This is simplified - using median of previous month as target
    target_balances: np.ndarray | None = None

    # Year-to-date contribution tracking for IRS limits
    ytd_contributions: dict[str, float] = {"401k": 0.0, "roth": 0.0}

    # Tax configuration
    enforce_limits = (
        tax_config is not None
        and tax_config.enabled
        and tax_config.enforce_contribution_limits
    )
    enforce_rmd = (
        tax_config is not None and tax_config.enabled and tax_config.enforce_rmd
    )
    tax_efficient = (
        tax_config is not None
        and tax_config.enabled
        and tax_config.tax_efficient_withdrawal
    )
    irs_limits = tax_config.irs_limits if tax_config else CURRENT_IRS_LIMITS
    cost_basis_ratio = tax_config.cost_basis_ratio if tax_config else 0.5

    for month in range(months):
        # Calculate current age
        age = config.age_at_month(month)

        # Reset YTD contributions at start of each year (every 12 months)
        if month > 0 and month % 12 == 0:
            ytd_contributions = {"401k": 0.0, "roth": 0.0}

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

            if enforce_rmd or tax_efficient:
                # Use tax-aware withdrawal with RMD enforcement
                _apply_withdrawals_tax_aware(
                    balances[:, month, :],
                    withdrawal_rate,
                    age,
                    irs_limits if enforce_rmd else None,
                    cost_basis_ratio,
                    tax_efficient,
                )
            else:
                # Original proportional withdrawal
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
                if soft_contribution_factors is not None:
                    effective_contributions = monthly_contributions.scaled_by_factors(
                        soft_contribution_factors
                    )
                else:
                    effective_contributions = monthly_contributions.scaled(
                        soft_contribution_factor
                    )
            else:
                effective_contributions = monthly_contributions

            ytd_contributions = _apply_contributions(
                balances[:, month, :],
                effective_contributions,
                ytd_contributions,
                age,
                irs_limits,
                enforce_limits,
            )

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
