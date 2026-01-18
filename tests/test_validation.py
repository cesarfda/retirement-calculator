"""Tests for input validation."""

import pytest

from core.portfolio import Allocation
from core.validation import (
    validate_age,
    validate_balances,
    validate_contributions,
    validate_allocation,
    validate_simulation_params,
    validate_all,
    ValidationResult,
)


class TestAgeValidation:
    """Tests for age validation."""

    def test_valid_ages(self):
        """Normal age range passes validation."""
        result = validate_age(35, 65, 55)
        assert result.is_valid()

    def test_current_age_too_young(self):
        """Age under 18 fails."""
        result = validate_age(17, 65)
        assert not result.is_valid()
        assert any("current_age" in e[0] for e in result.errors)

    def test_current_age_too_old(self):
        """Age over 100 fails."""
        result = validate_age(101, 110)
        assert not result.is_valid()

    def test_retirement_before_current(self):
        """Retirement age before current fails."""
        result = validate_age(40, 35)
        assert not result.is_valid()
        assert any("retirement_age" in e[0] for e in result.errors)

    def test_soft_retirement_after_retirement(self):
        """Soft retirement after retirement fails."""
        result = validate_age(35, 65, 70)
        assert not result.is_valid()
        assert any("soft_retirement_age" in e[0] for e in result.errors)

    def test_soft_retirement_before_current(self):
        """Soft retirement before current age fails."""
        result = validate_age(35, 65, 30)
        assert not result.is_valid()


class TestBalanceValidation:
    """Tests for balance validation."""

    def test_valid_balances(self):
        """Normal balances pass."""
        result = validate_balances({"401k": 100000, "roth": 50000, "taxable": 25000})
        assert result.is_valid()

    def test_zero_balances(self):
        """Zero balances pass."""
        result = validate_balances({"401k": 0, "roth": 0, "taxable": 0})
        assert result.is_valid()

    def test_negative_balance(self):
        """Negative balance fails."""
        result = validate_balances({"401k": -1000})
        assert not result.is_valid()

    def test_extreme_balance(self):
        """Balance over 1 billion fails sanity check."""
        result = validate_balances({"401k": 2_000_000_000})
        assert not result.is_valid()


class TestContributionValidation:
    """Tests for contribution validation."""

    def test_valid_contributions(self):
        """Normal contributions pass."""
        result = validate_contributions({
            "401k": 1000,
            "roth": 500,
            "taxable": 500,
            "employer_match_rate": 0.5,
            "employer_match_cap": 500,
        })
        assert result.is_valid()

    def test_negative_contribution(self):
        """Negative contribution fails."""
        result = validate_contributions({"401k": -100})
        assert not result.is_valid()

    def test_excessive_contribution(self):
        """Unreasonably high monthly contribution fails."""
        result = validate_contributions({"401k": 200000})
        assert not result.is_valid()

    def test_invalid_match_rate(self):
        """Match rate over 200% fails."""
        result = validate_contributions({"employer_match_rate": 2.5})
        assert not result.is_valid()


class TestAllocationValidation:
    """Tests for allocation validation."""

    def test_valid_allocation(self):
        """Normal allocation passes."""
        alloc = Allocation(us=0.6, vxus=0.3, sgov=0.1)
        result = validate_allocation(alloc)
        assert result.is_valid()

    def test_zero_allocation(self):
        """Zero total allocation fails."""
        alloc = Allocation(us=0, vxus=0, sgov=0)
        result = validate_allocation(alloc)
        assert not result.is_valid()

    def test_negative_allocation(self):
        """Negative allocation fails."""
        alloc = Allocation(us=-0.1, vxus=0.5, sgov=0.6)
        result = validate_allocation(alloc)
        assert not result.is_valid()


class TestSimulationParamsValidation:
    """Tests for simulation parameter validation."""

    def test_valid_params(self):
        """Normal parameters pass."""
        result = validate_simulation_params(
            years=30,
            n_simulations=1000,
            withdrawal_rate=0.04,
            expense_ratio=0.001,
        )
        assert result.is_valid()

    def test_zero_years(self):
        """Zero years fails."""
        result = validate_simulation_params(0, 1000, 0.04)
        assert not result.is_valid()

    def test_excessive_years(self):
        """Over 60 years fails."""
        result = validate_simulation_params(70, 1000, 0.04)
        assert not result.is_valid()

    def test_too_few_simulations(self):
        """Under 10 simulations fails."""
        result = validate_simulation_params(30, 5, 0.04)
        assert not result.is_valid()

    def test_low_withdrawal_rate(self):
        """Under 1% withdrawal rate fails."""
        result = validate_simulation_params(30, 1000, 0.005)
        assert not result.is_valid()

    def test_high_withdrawal_rate(self):
        """Over 15% withdrawal rate fails."""
        result = validate_simulation_params(30, 1000, 0.20)
        assert not result.is_valid()

    def test_negative_expense_ratio(self):
        """Negative expense ratio fails."""
        result = validate_simulation_params(30, 1000, 0.04, expense_ratio=-0.001)
        assert not result.is_valid()


class TestValidateAll:
    """Tests for combined validation."""

    def test_all_valid(self):
        """All valid inputs pass."""
        result = validate_all(
            current_age=35,
            retirement_age=65,
            soft_retirement_age=55,
            balances={"401k": 100000, "roth": 50000},
            contributions={"401k": 1000, "roth": 500},
            allocation=Allocation(us=0.6, vxus=0.3, sgov=0.1),
            years=30,
            n_simulations=1000,
            withdrawal_rate=0.04,
            expense_ratio=0.001,
        )
        assert result.is_valid()

    def test_multiple_errors(self):
        """Multiple invalid inputs collect all errors."""
        result = validate_all(
            current_age=15,  # Invalid
            retirement_age=10,  # Invalid
            soft_retirement_age=None,
            balances={"401k": -1000},  # Invalid
            contributions={"401k": 1000},
            allocation=Allocation(us=0, vxus=0, sgov=0),  # Invalid
            years=100,  # Invalid
            n_simulations=1000,
            withdrawal_rate=0.04,
        )
        assert not result.is_valid()
        assert len(result.errors) >= 4
