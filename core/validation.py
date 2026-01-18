"""Input validation for retirement calculator parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.portfolio import Allocation


@dataclass
class ValidationResult:
    """Result of validation containing any errors found."""

    errors: list[tuple[str, str]] = field(default_factory=list)

    def add_error(self, field_name: str, message: str) -> None:
        """Add a validation error."""
        self.errors.append((field_name, message))

    def is_valid(self) -> bool:
        """Return True if no validation errors."""
        return len(self.errors) == 0

    def error_messages(self) -> list[str]:
        """Return formatted error messages."""
        return [f"{field_name}: {message}" for field_name, message in self.errors]


def validate_age(
    current_age: int,
    retirement_age: int,
    soft_retirement_age: int | None = None,
) -> ValidationResult:
    """Validate age-related inputs."""
    result = ValidationResult()

    if current_age < 18:
        result.add_error("current_age", "Must be at least 18")
    if current_age > 100:
        result.add_error("current_age", "Must be at most 100")

    if retirement_age <= current_age:
        result.add_error("retirement_age", "Must be greater than current age")
    if retirement_age > 100:
        result.add_error("retirement_age", "Must be at most 100")

    if soft_retirement_age is not None:
        if soft_retirement_age <= current_age:
            result.add_error("soft_retirement_age", "Must be greater than current age")
        if soft_retirement_age > retirement_age:
            result.add_error(
                "soft_retirement_age", "Must be at most retirement age"
            )

    return result


def validate_balances(balances: dict[str, float]) -> ValidationResult:
    """Validate account balances."""
    result = ValidationResult()

    for account, balance in balances.items():
        if balance < 0:
            result.add_error(f"balance_{account}", "Cannot be negative")
        if balance > 1_000_000_000:  # 1 billion sanity check
            result.add_error(f"balance_{account}", "Exceeds maximum allowed value")

    return result


def validate_contributions(contributions: dict[str, float]) -> ValidationResult:
    """Validate monthly contributions."""
    result = ValidationResult()

    # Expected contribution fields
    contribution_fields = ["401k", "roth", "taxable"]

    for field_name in contribution_fields:
        if field_name in contributions:
            value = contributions[field_name]
            if value < 0:
                result.add_error(f"contrib_{field_name}", "Cannot be negative")
            if value > 100_000:  # Monthly contribution sanity check
                result.add_error(
                    f"contrib_{field_name}", "Exceeds reasonable monthly contribution"
                )

    # Validate match parameters
    match_rate = contributions.get("employer_match_rate", 0.0)
    if match_rate < 0 or match_rate > 2.0:
        result.add_error("employer_match_rate", "Must be between 0 and 200%")

    match_cap = contributions.get("employer_match_cap", 0.0)
    if match_cap < 0:
        result.add_error("employer_match_cap", "Cannot be negative")

    return result


def validate_allocation(allocation: Allocation) -> ValidationResult:
    """Validate asset allocation."""
    result = ValidationResult()

    total = allocation.us + allocation.vxus + allocation.sgov

    if allocation.us < 0 or allocation.vxus < 0 or allocation.sgov < 0:
        result.add_error("allocation", "Individual allocations cannot be negative")

    # Allow small floating point tolerance
    if abs(total) < 0.001:
        result.add_error("allocation", "Total allocation cannot be zero")

    return result


def validate_simulation_params(
    years: int,
    n_simulations: int,
    withdrawal_rate: float,
    expense_ratio: float = 0.0,
) -> ValidationResult:
    """Validate simulation parameters."""
    result = ValidationResult()

    if years < 1:
        result.add_error("years", "Must simulate at least 1 year")
    if years > 60:
        result.add_error("years", "Cannot simulate more than 60 years")

    if n_simulations < 10:
        result.add_error("n_simulations", "Must run at least 10 simulations")
    if n_simulations > 100_000:
        result.add_error("n_simulations", "Cannot run more than 100,000 simulations")

    if withdrawal_rate < 0.01:
        result.add_error("withdrawal_rate", "Must be at least 1%")
    if withdrawal_rate > 0.15:
        result.add_error("withdrawal_rate", "Cannot exceed 15%")

    if expense_ratio < 0:
        result.add_error("expense_ratio", "Cannot be negative")
    if expense_ratio > 0.03:
        result.add_error("expense_ratio", "Expense ratio above 3% is unusually high")

    return result


def validate_all(
    current_age: int,
    retirement_age: int,
    soft_retirement_age: int | None,
    balances: dict[str, float],
    contributions: dict[str, float],
    allocation: Allocation,
    years: int,
    n_simulations: int,
    withdrawal_rate: float,
    expense_ratio: float = 0.0,
) -> ValidationResult:
    """Run all validations and combine results."""
    combined = ValidationResult()

    # Run each validation
    validations = [
        validate_age(current_age, retirement_age, soft_retirement_age),
        validate_balances(balances),
        validate_contributions(contributions),
        validate_allocation(allocation),
        validate_simulation_params(years, n_simulations, withdrawal_rate, expense_ratio),
    ]

    for result in validations:
        combined.errors.extend(result.errors)

    return combined
