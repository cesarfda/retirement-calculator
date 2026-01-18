"""Glide path / bond tent allocation strategy for retirement planning."""

from __future__ import annotations

from dataclasses import dataclass

from core.portfolio import Allocation


@dataclass(frozen=True)
class GlidePath:
    """
    Defines a glide path for equity allocation over time.

    The "bond tent" strategy reduces equity exposure as retirement approaches
    (protecting against sequence of returns risk), then gradually increases
    equity exposure in early retirement as the risk window passes.

    Attributes:
        start_equity: Equity percentage at simulation start (e.g., 0.9 for 90%)
        retirement_equity: Equity percentage at retirement (most conservative point)
        end_equity: Equity percentage at end of simulation (e.g., 0.6 for 60%)
        international_ratio: Ratio of international to total equity (e.g., 0.33)
    """

    start_equity: float = 0.90
    retirement_equity: float = 0.50
    end_equity: float = 0.60
    international_ratio: float = 0.33  # Portion of equity allocated to international

    def __post_init__(self) -> None:
        """Validate glide path parameters."""
        if not 0 <= self.start_equity <= 1:
            raise ValueError("start_equity must be between 0 and 1")
        if not 0 <= self.retirement_equity <= 1:
            raise ValueError("retirement_equity must be between 0 and 1")
        if not 0 <= self.end_equity <= 1:
            raise ValueError("end_equity must be between 0 and 1")
        if not 0 <= self.international_ratio <= 1:
            raise ValueError("international_ratio must be between 0 and 1")

    def get_equity_at_month(
        self,
        month: int,
        retirement_month: int,
        total_months: int,
    ) -> float:
        """
        Calculate equity allocation for a given month.

        Uses linear interpolation:
        - From start to retirement: interpolate from start_equity to retirement_equity
        - From retirement to end: interpolate from retirement_equity to end_equity

        Args:
            month: Current month (0-indexed)
            retirement_month: Month when retirement begins
            total_months: Total simulation length in months

        Returns:
            Equity allocation as a float between 0 and 1
        """
        if month <= 0:
            return self.start_equity

        if month >= total_months:
            return self.end_equity

        if month < retirement_month:
            # Pre-retirement: decrease equity toward retirement
            if retirement_month == 0:
                return self.retirement_equity
            progress = month / retirement_month
            return self.start_equity + progress * (
                self.retirement_equity - self.start_equity
            )
        else:
            # Post-retirement: increase equity from conservative point
            post_retirement_months = total_months - retirement_month
            if post_retirement_months == 0:
                return self.retirement_equity
            progress = (month - retirement_month) / post_retirement_months
            return self.retirement_equity + progress * (
                self.end_equity - self.retirement_equity
            )

    def get_allocation_at_month(
        self,
        month: int,
        retirement_month: int,
        total_months: int,
    ) -> Allocation:
        """
        Get full asset allocation for a given month.

        Splits equity between US and international based on international_ratio.
        Remainder goes to bonds (SGOV).

        Args:
            month: Current month (0-indexed)
            retirement_month: Month when retirement begins
            total_months: Total simulation length in months

        Returns:
            Allocation with us, vxus, sgov weights
        """
        equity = self.get_equity_at_month(month, retirement_month, total_months)
        bonds = 1.0 - equity

        international = equity * self.international_ratio
        us = equity - international

        return Allocation(us=us, vxus=international, sgov=bonds)


def create_default_glide_path() -> GlidePath:
    """Create a sensible default glide path."""
    return GlidePath(
        start_equity=0.90,
        retirement_equity=0.50,
        end_equity=0.60,
        international_ratio=0.33,
    )


def create_aggressive_glide_path() -> GlidePath:
    """Create an aggressive glide path with higher equity throughout."""
    return GlidePath(
        start_equity=0.95,
        retirement_equity=0.65,
        end_equity=0.75,
        international_ratio=0.30,
    )


def create_conservative_glide_path() -> GlidePath:
    """Create a conservative glide path with lower equity throughout."""
    return GlidePath(
        start_equity=0.80,
        retirement_equity=0.40,
        end_equity=0.50,
        international_ratio=0.35,
    )
