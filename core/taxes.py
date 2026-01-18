"""Tax calculation logic for retirement withdrawals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from core.tax_config import (
    FEDERAL_BRACKETS_2024_MFJ,
    FEDERAL_BRACKETS_2024_SINGLE,
    LTCG_BRACKETS_2024_MFJ,
    LTCG_BRACKETS_2024_SINGLE,
)


class FilingStatus(Enum):
    """Tax filing status."""

    SINGLE = "single"
    MARRIED_FILING_JOINTLY = "mfj"


@dataclass(frozen=True)
class TaxResult:
    """Result of a tax calculation."""

    taxable_income: float
    federal_tax: float
    effective_rate: float

    @property
    def after_tax_income(self) -> float:
        """Income after federal taxes."""
        return self.taxable_income - self.federal_tax


@dataclass(frozen=True)
class WithdrawalTaxResult:
    """Result of withdrawal tax calculation."""

    gross_withdrawal: float
    federal_tax: float
    capital_gains_tax: float
    total_tax: float
    net_withdrawal: float
    niit: float = 0.0  # Net Investment Income Tax

    @property
    def effective_rate(self) -> float:
        """Effective tax rate on the withdrawal."""
        if self.gross_withdrawal <= 0:
            return 0.0
        return self.total_tax / self.gross_withdrawal


def get_brackets(
    filing_status: FilingStatus,
    bracket_type: Literal["income", "capital_gains"] = "income",
) -> list[tuple[float, float]]:
    """
    Get tax brackets for a filing status.

    Args:
        filing_status: Single or married filing jointly
        bracket_type: 'income' for ordinary income, 'capital_gains' for LTCG

    Returns:
        List of (upper_bound, rate) tuples
    """
    if bracket_type == "capital_gains":
        if filing_status == FilingStatus.MARRIED_FILING_JOINTLY:
            return LTCG_BRACKETS_2024_MFJ
        return LTCG_BRACKETS_2024_SINGLE
    else:
        if filing_status == FilingStatus.MARRIED_FILING_JOINTLY:
            return FEDERAL_BRACKETS_2024_MFJ
        return FEDERAL_BRACKETS_2024_SINGLE


def calculate_federal_tax(
    taxable_income: float,
    filing_status: FilingStatus = FilingStatus.SINGLE,
) -> TaxResult:
    """
    Calculate federal income tax using marginal tax brackets.

    Args:
        taxable_income: Income after deductions
        filing_status: Tax filing status

    Returns:
        TaxResult with calculated tax and effective rate
    """
    if taxable_income <= 0:
        return TaxResult(taxable_income=0.0, federal_tax=0.0, effective_rate=0.0)

    brackets = get_brackets(filing_status, "income")
    tax = 0.0
    prev_bound = 0.0

    for upper_bound, rate in brackets:
        if taxable_income <= prev_bound:
            break

        bracket_income = min(taxable_income, upper_bound) - prev_bound
        if bracket_income > 0:
            tax += bracket_income * rate

        prev_bound = upper_bound

    effective_rate = tax / taxable_income if taxable_income > 0 else 0.0

    return TaxResult(
        taxable_income=taxable_income,
        federal_tax=tax,
        effective_rate=effective_rate,
    )


def calculate_capital_gains_tax(
    long_term_gains: float,
    ordinary_income: float,
    filing_status: FilingStatus = FilingStatus.SINGLE,
) -> float:
    """
    Calculate long-term capital gains tax.

    Capital gains are stacked on top of ordinary income to determine
    which brackets they fall into.

    Args:
        long_term_gains: Long-term capital gains amount
        ordinary_income: Ordinary taxable income (determines starting bracket)
        filing_status: Tax filing status

    Returns:
        Capital gains tax amount
    """
    if long_term_gains <= 0:
        return 0.0

    brackets = get_brackets(filing_status, "capital_gains")
    tax = 0.0
    gains_remaining = long_term_gains
    income_so_far = ordinary_income

    for upper_bound, rate in brackets:
        if gains_remaining <= 0:
            break

        # How much room in this bracket?
        room_in_bracket = max(0, upper_bound - income_so_far)
        if room_in_bracket <= 0:
            income_so_far = upper_bound
            continue

        # Apply this bracket's rate to gains that fit
        gains_in_bracket = min(gains_remaining, room_in_bracket)
        tax += gains_in_bracket * rate

        gains_remaining -= gains_in_bracket
        income_so_far += gains_in_bracket

    return tax


def calculate_withdrawal_taxes(
    withdrawal_401k: float,
    withdrawal_roth: float,
    withdrawal_taxable: float,
    cost_basis_ratio: float = 0.5,
    other_income: float = 0.0,
    standard_deduction: float = 14_600,
    filing_status: FilingStatus = FilingStatus.SINGLE,
) -> WithdrawalTaxResult:
    """
    Calculate taxes on retirement withdrawals.

    Tax treatment by account:
    - 401k/Traditional: Fully taxable as ordinary income
    - Roth IRA: Tax-free (qualified withdrawals)
    - Taxable: Only gains taxed (at capital gains rates)

    Args:
        withdrawal_401k: Amount withdrawn from 401k/Traditional IRA
        withdrawal_roth: Amount withdrawn from Roth IRA
        withdrawal_taxable: Amount withdrawn from taxable brokerage
        cost_basis_ratio: Fraction of taxable withdrawal that is cost basis (not taxed)
        other_income: Other taxable income (Social Security, pension, etc.)
        standard_deduction: Standard deduction amount
        filing_status: Tax filing status

    Returns:
        WithdrawalTaxResult with gross, taxes, and net amounts
    """
    gross_withdrawal = withdrawal_401k + withdrawal_roth + withdrawal_taxable

    # 401k withdrawals are fully taxable as ordinary income
    ordinary_income = withdrawal_401k + other_income

    # Apply standard deduction
    taxable_ordinary = max(0, ordinary_income - standard_deduction)

    # Calculate federal tax on ordinary income
    federal_result = calculate_federal_tax(taxable_ordinary, filing_status)
    federal_tax = federal_result.federal_tax

    # Taxable account: only gains are taxed (at capital gains rates)
    # cost_basis_ratio = what fraction is original investment (not taxed)
    taxable_gains = withdrawal_taxable * (1 - cost_basis_ratio)

    # Capital gains are stacked on ordinary income for bracket determination
    capital_gains_tax = calculate_capital_gains_tax(
        taxable_gains,
        taxable_ordinary,
        filing_status,
    )

    # Roth withdrawals are tax-free (assuming qualified)
    # No tax on withdrawal_roth

    # Calculate NIIT on investment income (capital gains from taxable)
    # MAGI for NIIT purposes includes the 401k withdrawal and gains
    magi = ordinary_income + taxable_gains
    niit = calculate_niit(
        investment_income=taxable_gains,
        magi=magi,
        filing_status=filing_status,
    )

    total_tax = federal_tax + capital_gains_tax + niit
    net_withdrawal = gross_withdrawal - total_tax

    return WithdrawalTaxResult(
        gross_withdrawal=gross_withdrawal,
        federal_tax=federal_tax,
        capital_gains_tax=capital_gains_tax,
        total_tax=total_tax,
        net_withdrawal=net_withdrawal,
        niit=niit,
    )


def estimate_marginal_rate(
    current_income: float,
    filing_status: FilingStatus = FilingStatus.SINGLE,
) -> float:
    """
    Estimate marginal tax rate at a given income level.

    Useful for determining optimal Roth conversion or withdrawal amounts.

    Args:
        current_income: Current taxable income
        filing_status: Tax filing status

    Returns:
        Marginal tax rate (e.g., 0.22 for 22% bracket)
    """
    brackets = get_brackets(filing_status, "income")

    for upper_bound, rate in brackets:
        if current_income <= upper_bound:
            return rate

    # If income exceeds all brackets, return top rate
    return brackets[-1][1]


def calculate_niit(
    investment_income: float,
    magi: float,
    filing_status: FilingStatus = FilingStatus.SINGLE,
) -> float:
    """
    Calculate Net Investment Income Tax (NIIT).

    The NIIT is a 3.8% surtax on investment income for high earners,
    enacted as part of the Affordable Care Act.

    Investment income includes:
    - Capital gains (including from taxable account withdrawals)
    - Dividends
    - Interest (except municipal bonds)
    - Rental income
    - Passive business income

    Args:
        investment_income: Total net investment income
        magi: Modified Adjusted Gross Income
        filing_status: Tax filing status

    Returns:
        NIIT amount (3.8% of lesser of investment income or excess MAGI)
    """
    if investment_income <= 0:
        return 0.0

    # MAGI thresholds for NIIT
    if filing_status == FilingStatus.MARRIED_FILING_JOINTLY:
        threshold = 250_000
    else:
        threshold = 200_000  # Single, head of household, etc.

    excess_magi = max(0, magi - threshold)

    if excess_magi <= 0:
        return 0.0

    # NIIT applies to lesser of: investment income or excess MAGI
    taxable_amount = min(investment_income, excess_magi)

    return taxable_amount * 0.038


def calculate_bracket_room(
    current_income: float,
    target_rate: float,
    filing_status: FilingStatus = FilingStatus.SINGLE,
) -> float:
    """
    Calculate room remaining in brackets up to a target rate.

    Useful for Roth conversions - how much can you convert before
    hitting a higher bracket?

    Args:
        current_income: Current taxable income
        target_rate: Maximum marginal rate to stay at or below
        filing_status: Tax filing status

    Returns:
        Dollar amount of room available before exceeding target rate
    """
    brackets = get_brackets(filing_status, "income")
    room = 0.0

    for upper_bound, rate in brackets:
        if rate > target_rate:
            break

        if current_income < upper_bound:
            room += upper_bound - max(current_income, 0)
            current_income = upper_bound

    return room


@dataclass
class WithdrawalStrategy:
    """
    Configuration for tax-efficient withdrawal strategy.

    Attributes:
        prioritize_taxable: Withdraw from taxable first (preserves tax-advantaged growth)
        fill_bracket: Target tax bracket to fill with 401k withdrawals
        minimize_roth: Use Roth last to maximize tax-free growth
    """

    prioritize_taxable: bool = True
    fill_bracket: float = 0.22  # Fill up to 22% bracket
    minimize_roth: bool = True


def calculate_optimal_withdrawal(
    needed_after_tax: float,
    balance_401k: float,
    balance_roth: float,
    balance_taxable: float,
    cost_basis_ratio: float = 0.5,
    other_income: float = 0.0,
    rmd_amount: float = 0.0,
    standard_deduction: float = 14_600,
    filing_status: FilingStatus = FilingStatus.SINGLE,
    strategy: WithdrawalStrategy | None = None,
) -> tuple[float, float, float]:
    """
    Calculate optimal withdrawal amounts from each account.

    Implements tax-efficient withdrawal ordering:
    1. Take required RMD from 401k first
    2. Withdraw from taxable (lowest effective rate due to basis)
    3. Fill lower tax brackets with 401k if beneficial
    4. Use Roth last (tax-free, preserve for later)

    Args:
        needed_after_tax: After-tax spending needs
        balance_401k: Available 401k balance
        balance_roth: Available Roth balance
        balance_taxable: Available taxable balance
        cost_basis_ratio: Fraction of taxable that is cost basis
        other_income: Other taxable income
        rmd_amount: Required minimum distribution (must be taken)
        standard_deduction: Standard deduction amount
        filing_status: Tax filing status
        strategy: Optional withdrawal strategy configuration

    Returns:
        Tuple of (withdrawal_401k, withdrawal_roth, withdrawal_taxable)
    """
    if strategy is None:
        strategy = WithdrawalStrategy()

    withdrawal_401k = 0.0
    withdrawal_roth = 0.0
    withdrawal_taxable = 0.0
    remaining_need = needed_after_tax

    # Step 1: Take RMD first (mandatory)
    if rmd_amount > 0:
        rmd_withdrawal = min(rmd_amount, balance_401k)
        withdrawal_401k += rmd_withdrawal

        # Calculate after-tax value of RMD
        rmd_tax_result = calculate_withdrawal_taxes(
            withdrawal_401k=rmd_withdrawal,
            withdrawal_roth=0,
            withdrawal_taxable=0,
            other_income=other_income,
            standard_deduction=standard_deduction,
            filing_status=filing_status,
        )
        remaining_need -= rmd_tax_result.net_withdrawal

    if remaining_need <= 0:
        return (withdrawal_401k, withdrawal_roth, withdrawal_taxable)

    # Step 2: Withdraw from taxable account (favorable capital gains rates)
    if strategy.prioritize_taxable and balance_taxable > 0:
        # Estimate how much we need gross to get remaining_need after taxes
        # Capital gains tax is relatively low, so gross ≈ net * 1.1 as estimate
        estimated_gross = remaining_need * 1.1
        taxable_withdrawal = min(estimated_gross, balance_taxable)

        # Calculate actual after-tax value
        taxable_tax_result = calculate_withdrawal_taxes(
            withdrawal_401k=withdrawal_401k,
            withdrawal_roth=0,
            withdrawal_taxable=taxable_withdrawal,
            cost_basis_ratio=cost_basis_ratio,
            other_income=other_income,
            standard_deduction=standard_deduction,
            filing_status=filing_status,
        )

        withdrawal_taxable = taxable_withdrawal
        remaining_need -= (
            taxable_withdrawal
            - taxable_tax_result.capital_gains_tax
        )

    if remaining_need <= 0:
        return (withdrawal_401k, withdrawal_roth, withdrawal_taxable)

    # Step 3: Fill lower tax brackets with 401k
    current_taxable_income = withdrawal_401k + other_income - standard_deduction
    bracket_room = calculate_bracket_room(
        max(0, current_taxable_income),
        strategy.fill_bracket,
        filing_status,
    )

    if bracket_room > 0 and (balance_401k - withdrawal_401k) > 0:
        additional_401k = min(
            bracket_room,
            balance_401k - withdrawal_401k,
            remaining_need * 1.3,  # Gross up for taxes
        )
        withdrawal_401k += additional_401k

        # Recalculate remaining need
        tax_result = calculate_withdrawal_taxes(
            withdrawal_401k=withdrawal_401k,
            withdrawal_roth=0,
            withdrawal_taxable=withdrawal_taxable,
            cost_basis_ratio=cost_basis_ratio,
            other_income=other_income,
            standard_deduction=standard_deduction,
            filing_status=filing_status,
        )
        remaining_need = needed_after_tax - tax_result.net_withdrawal

    if remaining_need <= 0:
        return (withdrawal_401k, withdrawal_roth, withdrawal_taxable)

    # Step 4: Use Roth for remaining need (tax-free, so gross = net)
    if strategy.minimize_roth:
        # Only use Roth for what's truly needed
        roth_withdrawal = min(remaining_need, balance_roth)
        withdrawal_roth = roth_withdrawal
        remaining_need -= roth_withdrawal

    if remaining_need <= 0:
        return (withdrawal_401k, withdrawal_roth, withdrawal_taxable)

    # Step 5: If still need more, take additional from 401k (even at higher rates)
    if remaining_need > 0 and (balance_401k - withdrawal_401k) > 0:
        # Gross up remaining need for higher tax bracket
        marginal_rate = estimate_marginal_rate(
            withdrawal_401k + other_income - standard_deduction,
            filing_status,
        )
        additional_gross = remaining_need / (1 - marginal_rate)
        additional_401k = min(additional_gross, balance_401k - withdrawal_401k)
        withdrawal_401k += additional_401k

    return (withdrawal_401k, withdrawal_roth, withdrawal_taxable)
