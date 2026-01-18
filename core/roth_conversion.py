"""Roth conversion optimization strategies for tax-efficient retirement planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.taxes import FilingStatus


@dataclass(frozen=True)
class RothConversionStrategy:
    """
    Configuration for Roth conversion ladder strategy.

    Roth conversions move money from traditional 401k/IRA to Roth accounts,
    paying taxes now at current rates to avoid higher taxes later (especially
    when RMDs force large taxable withdrawals).

    Attributes:
        enabled: Whether to perform Roth conversions during simulation
        target_bracket: Maximum marginal tax rate to fill (e.g., 0.22 for 22%)
        max_annual_conversion: Cap on annual conversion amount
        start_age: Age to begin conversions (typically after retirement)
        stop_age: Age to stop conversions (before RMDs begin at 73)
        min_traditional_balance: Minimum 401k balance to trigger conversion
    """

    enabled: bool = False
    target_bracket: float = 0.22  # Fill up to 22% bracket
    max_annual_conversion: float = 50_000.0
    start_age: int = 60  # Start after early retirement
    stop_age: int = 72  # Stop before RMDs begin
    min_traditional_balance: float = 10_000.0

    def is_active_at_age(self, age: float) -> bool:
        """Check if conversions should occur at this age."""
        return self.enabled and self.start_age <= age < self.stop_age


# Federal tax brackets for 2024 (single filer)
# Format: (upper_bound, marginal_rate)
FEDERAL_BRACKETS_SINGLE: list[tuple[float, float]] = [
    (11_600, 0.10),
    (47_150, 0.12),
    (100_525, 0.22),
    (191_950, 0.24),
    (243_725, 0.32),
    (609_350, 0.35),
    (float("inf"), 0.37),
]

# Federal tax brackets for 2024 (married filing jointly)
FEDERAL_BRACKETS_MFJ: list[tuple[float, float]] = [
    (23_200, 0.10),
    (94_300, 0.12),
    (201_050, 0.22),
    (383_900, 0.24),
    (487_450, 0.32),
    (731_200, 0.35),
    (float("inf"), 0.37),
]

# Standard deductions for 2024
STANDARD_DEDUCTION_SINGLE = 14_600
STANDARD_DEDUCTION_MFJ = 29_200


def get_brackets_for_status(filing_status: str) -> list[tuple[float, float]]:
    """Get tax brackets for filing status."""
    if filing_status == "mfj" or filing_status == "married":
        return FEDERAL_BRACKETS_MFJ
    return FEDERAL_BRACKETS_SINGLE


def get_standard_deduction(filing_status: str) -> float:
    """Get standard deduction for filing status."""
    if filing_status == "mfj" or filing_status == "married":
        return STANDARD_DEDUCTION_MFJ
    return STANDARD_DEDUCTION_SINGLE


def calculate_bracket_room(
    current_taxable_income: float,
    target_bracket_rate: float,
    filing_status: str = "single",
) -> float:
    """
    Calculate room remaining in tax brackets up to target rate.

    This determines how much additional income (from Roth conversion)
    can be added before exceeding the target marginal rate.

    Args:
        current_taxable_income: Current taxable income (after deductions)
        target_bracket_rate: Maximum marginal rate to stay at or below
        filing_status: Tax filing status ("single" or "mfj")

    Returns:
        Dollar amount of room available before exceeding target rate
    """
    brackets = get_brackets_for_status(filing_status)
    room = 0.0
    income_so_far = max(0, current_taxable_income)

    for upper_bound, rate in brackets:
        if rate > target_bracket_rate:
            # Stop when we hit brackets above target
            break

        if income_so_far < upper_bound:
            # Room in this bracket
            room += upper_bound - income_so_far
            income_so_far = upper_bound

    return room


def calculate_optimal_roth_conversion(
    traditional_balance: float,
    current_taxable_income: float,
    filing_status: str = "single",
    target_bracket: float = 0.22,
    max_conversion: float = 50_000.0,
    standard_deduction: float | None = None,
) -> float:
    """
    Calculate optimal Roth conversion amount to fill target bracket.

    The goal is to convert as much as possible while staying within
    favorable tax brackets, reducing future RMDs and tax burden.

    Args:
        traditional_balance: Balance in 401k/Traditional IRA
        current_taxable_income: Other taxable income this year (before deduction)
        filing_status: Tax filing status
        target_bracket: Maximum marginal tax rate to fill
        max_conversion: Maximum conversion amount (from strategy)
        standard_deduction: Override standard deduction (uses default if None)

    Returns:
        Recommended conversion amount for this year
    """
    if traditional_balance <= 0:
        return 0.0

    # Apply standard deduction to get taxable income
    if standard_deduction is None:
        standard_deduction = get_standard_deduction(filing_status)

    taxable_income = max(0, current_taxable_income - standard_deduction)

    # Calculate room in brackets up to target rate
    bracket_room = calculate_bracket_room(
        taxable_income,
        target_bracket,
        filing_status,
    )

    if bracket_room <= 0:
        # Already above target bracket - no conversion recommended
        return 0.0

    # Convert up to the smaller of: bracket room, max conversion, or balance
    conversion = min(bracket_room, max_conversion, traditional_balance)

    return conversion


def calculate_conversion_tax(
    conversion_amount: float,
    other_taxable_income: float,
    filing_status: str = "single",
    standard_deduction: float | None = None,
) -> float:
    """
    Calculate federal tax on a Roth conversion.

    Roth conversions are taxed as ordinary income. This calculates
    the marginal tax on the conversion amount.

    Args:
        conversion_amount: Amount being converted
        other_taxable_income: Other taxable income (before deduction)
        filing_status: Tax filing status
        standard_deduction: Override standard deduction

    Returns:
        Tax owed on the conversion
    """
    if conversion_amount <= 0:
        return 0.0

    if standard_deduction is None:
        standard_deduction = get_standard_deduction(filing_status)

    brackets = get_brackets_for_status(filing_status)

    # Calculate tax without conversion
    base_taxable = max(0, other_taxable_income - standard_deduction)
    base_tax = _calculate_tax_from_brackets(base_taxable, brackets)

    # Calculate tax with conversion
    total_taxable = base_taxable + conversion_amount
    total_tax = _calculate_tax_from_brackets(total_taxable, brackets)

    # Marginal tax on conversion
    return total_tax - base_tax


def _calculate_tax_from_brackets(
    taxable_income: float,
    brackets: list[tuple[float, float]],
) -> float:
    """Calculate tax using marginal brackets."""
    if taxable_income <= 0:
        return 0.0

    tax = 0.0
    prev_bound = 0.0

    for upper_bound, rate in brackets:
        if taxable_income <= prev_bound:
            break

        bracket_income = min(taxable_income, upper_bound) - prev_bound
        if bracket_income > 0:
            tax += bracket_income * rate

        prev_bound = upper_bound

    return tax


def estimate_rmd_tax_savings(
    traditional_balance: float,
    conversion_amount: float,
    years_to_rmd: int,
    expected_growth_rate: float = 0.06,
    current_tax_rate: float = 0.22,
    future_tax_rate: float = 0.24,
) -> float:
    """
    Estimate tax savings from Roth conversion vs future RMD.

    Compares:
    - Tax paid now on conversion at current rate
    - Tax that would be paid on RMD at future rate (with growth)

    Args:
        traditional_balance: Current traditional balance
        conversion_amount: Amount to convert
        years_to_rmd: Years until RMDs begin
        expected_growth_rate: Expected annual growth rate
        current_tax_rate: Current marginal tax rate
        future_tax_rate: Expected future marginal tax rate

    Returns:
        Estimated tax savings (positive = conversion saves money)
    """
    if conversion_amount <= 0 or years_to_rmd <= 0:
        return 0.0

    # Tax paid now on conversion
    current_tax = conversion_amount * current_tax_rate

    # Future value of converted amount (now in Roth, tax-free growth)
    future_roth_value = conversion_amount * ((1 + expected_growth_rate) ** years_to_rmd)

    # If left in traditional, it would grow and be taxed at RMD
    future_traditional_value = conversion_amount * ((1 + expected_growth_rate) ** years_to_rmd)
    future_tax = future_traditional_value * future_tax_rate

    # Present value of future tax (discount at growth rate)
    pv_future_tax = future_tax / ((1 + expected_growth_rate) ** years_to_rmd)

    # Savings = avoided future tax - current tax paid
    return pv_future_tax - current_tax


@dataclass
class RothConversionResult:
    """Result of a Roth conversion calculation."""

    conversion_amount: float
    tax_owed: float
    effective_rate: float
    bracket_room_remaining: float
    estimated_future_savings: float

    @property
    def net_conversion(self) -> float:
        """Amount added to Roth after paying tax from other sources."""
        return self.conversion_amount


def analyze_roth_conversion(
    traditional_balance: float,
    other_income: float,
    filing_status: str = "single",
    strategy: RothConversionStrategy | None = None,
    years_to_rmd: int = 10,
    expected_growth_rate: float = 0.06,
) -> RothConversionResult:
    """
    Analyze optimal Roth conversion for current year.

    Args:
        traditional_balance: Balance in 401k/Traditional IRA
        other_income: Other taxable income this year
        filing_status: Tax filing status
        strategy: Roth conversion strategy (uses defaults if None)
        years_to_rmd: Years until RMDs begin
        expected_growth_rate: Expected annual growth rate

    Returns:
        RothConversionResult with recommended conversion and analysis
    """
    if strategy is None:
        strategy = RothConversionStrategy(enabled=True)

    if not strategy.enabled or traditional_balance < strategy.min_traditional_balance:
        return RothConversionResult(
            conversion_amount=0.0,
            tax_owed=0.0,
            effective_rate=0.0,
            bracket_room_remaining=0.0,
            estimated_future_savings=0.0,
        )

    # Calculate optimal conversion
    conversion = calculate_optimal_roth_conversion(
        traditional_balance=traditional_balance,
        current_taxable_income=other_income,
        filing_status=filing_status,
        target_bracket=strategy.target_bracket,
        max_conversion=strategy.max_annual_conversion,
    )

    # Calculate tax on conversion
    tax = calculate_conversion_tax(
        conversion_amount=conversion,
        other_taxable_income=other_income,
        filing_status=filing_status,
    )

    # Calculate effective rate
    effective_rate = tax / conversion if conversion > 0 else 0.0

    # Calculate remaining bracket room
    std_ded = get_standard_deduction(filing_status)
    taxable_after = max(0, other_income - std_ded) + conversion
    remaining_room = calculate_bracket_room(
        taxable_after,
        strategy.target_bracket,
        filing_status,
    )

    # Estimate future savings
    savings = estimate_rmd_tax_savings(
        traditional_balance=traditional_balance,
        conversion_amount=conversion,
        years_to_rmd=years_to_rmd,
        expected_growth_rate=expected_growth_rate,
        current_tax_rate=effective_rate,
        future_tax_rate=strategy.target_bracket + 0.02,  # Assume slightly higher future rate
    )

    return RothConversionResult(
        conversion_amount=conversion,
        tax_owed=tax,
        effective_rate=effective_rate,
        bracket_room_remaining=remaining_room,
        estimated_future_savings=savings,
    )
