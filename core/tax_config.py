"""IRS tax limits and configuration for retirement planning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IRSLimits:
    """
    IRS contribution limits for a given tax year.

    These limits are updated annually by the IRS and should be
    verified against current IRS publications.

    Attributes:
        year: Tax year these limits apply to
        limit_401k: Annual 401(k) elective deferral limit
        limit_401k_catchup: Additional catch-up contribution for age 50+
        limit_roth_ira: Annual Roth IRA contribution limit
        limit_roth_catchup: Additional Roth IRA catch-up for age 50+
        rmd_start_age: Age at which RMDs begin (SECURE 2.0 Act)
        standard_deduction_single: Standard deduction for single filers
        standard_deduction_married: Standard deduction for married filing jointly
    """

    year: int
    limit_401k: float = 23_000
    limit_401k_catchup: float = 7_500
    limit_roth_ira: float = 7_000
    limit_roth_catchup: float = 1_000
    rmd_start_age: int = 73
    standard_deduction_single: float = 14_600
    standard_deduction_married: float = 29_200

    def get_401k_limit(self, age: int) -> float:
        """Get total 401k limit including catch-up if eligible."""
        base = self.limit_401k
        if age >= 50:
            base += self.limit_401k_catchup
        return base

    def get_roth_ira_limit(self, age: int) -> float:
        """Get total Roth IRA limit including catch-up if eligible."""
        base = self.limit_roth_ira
        if age >= 50:
            base += self.limit_roth_catchup
        return base


# IRS limits by year
IRS_LIMITS_2024 = IRSLimits(
    year=2024,
    limit_401k=23_000,
    limit_401k_catchup=7_500,
    limit_roth_ira=7_000,
    limit_roth_catchup=1_000,
    rmd_start_age=73,
    standard_deduction_single=14_600,
    standard_deduction_married=29_200,
)

IRS_LIMITS_2025 = IRSLimits(
    year=2025,
    limit_401k=23_500,
    limit_401k_catchup=7_500,
    limit_roth_ira=7_000,
    limit_roth_catchup=1_000,
    rmd_start_age=73,
    standard_deduction_single=15_000,  # Projected
    standard_deduction_married=30_000,  # Projected
)

# Default to current year
CURRENT_IRS_LIMITS = IRS_LIMITS_2024


# IRS Uniform Lifetime Table for RMD calculations
# Source: IRS Publication 590-B, Table III
# Maps age to distribution period (divisor)
RMD_UNIFORM_LIFETIME_TABLE: dict[int, float] = {
    72: 27.4,
    73: 26.5,
    74: 25.5,
    75: 24.6,
    76: 23.7,
    77: 22.9,
    78: 22.0,
    79: 21.1,
    80: 20.2,
    81: 19.4,
    82: 18.5,
    83: 17.7,
    84: 16.8,
    85: 16.0,
    86: 15.2,
    87: 14.4,
    88: 13.7,
    89: 12.9,
    90: 12.2,
    91: 11.5,
    92: 10.8,
    93: 10.1,
    94: 9.5,
    95: 8.9,
    96: 8.4,
    97: 7.8,
    98: 7.3,
    99: 6.8,
    100: 6.4,
    101: 6.0,
    102: 5.6,
    103: 5.2,
    104: 4.9,
    105: 4.6,
    106: 4.3,
    107: 4.1,
    108: 3.9,
    109: 3.7,
    110: 3.5,
    111: 3.4,
    112: 3.3,
    113: 3.1,
    114: 3.0,
    115: 2.9,
    116: 2.8,
    117: 2.7,
    118: 2.5,
    119: 2.3,
    120: 2.0,
}


def get_rmd_divisor(age: int) -> float:
    """
    Get the RMD distribution period for a given age.

    Args:
        age: Account holder's age at end of year

    Returns:
        Distribution period (divisor) from IRS Uniform Lifetime Table
    """
    if age < 72:
        return 0.0  # No RMD required
    if age > 120:
        return 2.0  # Use minimum divisor for very old ages
    return RMD_UNIFORM_LIFETIME_TABLE.get(age, 2.0)


def calculate_rmd(traditional_balance: float, age: int, rmd_start_age: int = 73) -> float:
    """
    Calculate Required Minimum Distribution for a traditional account.

    Args:
        traditional_balance: Balance in traditional 401k/IRA at end of prior year
        age: Account holder's age at end of current year
        rmd_start_age: Age at which RMDs begin (default 73 per SECURE 2.0)

    Returns:
        Required minimum distribution amount for the year
    """
    if age < rmd_start_age:
        return 0.0

    divisor = get_rmd_divisor(age)
    if divisor <= 0:
        return 0.0

    return traditional_balance / divisor


# Federal tax brackets for 2024 (single filer)
# Format: (upper_bound, marginal_rate)
FEDERAL_BRACKETS_2024_SINGLE: list[tuple[float, float]] = [
    (11_600, 0.10),
    (47_150, 0.12),
    (100_525, 0.22),
    (191_950, 0.24),
    (243_725, 0.32),
    (609_350, 0.35),
    (float("inf"), 0.37),
]

# Federal tax brackets for 2024 (married filing jointly)
FEDERAL_BRACKETS_2024_MFJ: list[tuple[float, float]] = [
    (23_200, 0.10),
    (94_300, 0.12),
    (201_050, 0.22),
    (383_900, 0.24),
    (487_450, 0.32),
    (731_200, 0.35),
    (float("inf"), 0.37),
]

# Long-term capital gains brackets for 2024 (single filer)
LTCG_BRACKETS_2024_SINGLE: list[tuple[float, float]] = [
    (47_025, 0.00),   # 0% rate
    (518_900, 0.15),  # 15% rate
    (float("inf"), 0.20),  # 20% rate
]

# Long-term capital gains brackets for 2024 (married filing jointly)
LTCG_BRACKETS_2024_MFJ: list[tuple[float, float]] = [
    (94_050, 0.00),   # 0% rate
    (583_750, 0.15),  # 15% rate
    (float("inf"), 0.20),  # 20% rate
]
