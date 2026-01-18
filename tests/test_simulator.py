"""Tests for the simulation engine."""

import numpy as np
import pytest

from core.portfolio import Allocation
from core.simulator import (
    AccountBalances,
    MonthlyContributions,
    Guardrails,
    SimulationConfig,
    run_simulation,
)
from core.glide_path import create_default_glide_path


class TestAccountBalances:
    """Tests for AccountBalances dataclass."""

    def test_from_dict(self):
        """Create from dictionary."""
        data = {"401k": 100_000, "roth": 50_000, "taxable": 25_000}
        balances = AccountBalances.from_dict(data)
        assert balances.balance_401k == 100_000
        assert balances.balance_roth == 50_000
        assert balances.balance_taxable == 25_000

    def test_total(self, default_balances):
        """Total is sum of all accounts."""
        assert default_balances.total() == 215_000

    def test_as_array(self, default_balances):
        """Convert to numpy array."""
        arr = default_balances.as_array()
        assert arr[0] == 150_000
        assert arr[1] == 40_000
        assert arr[2] == 25_000


class TestMonthlyContributions:
    """Tests for MonthlyContributions dataclass."""

    def test_from_dict(self):
        """Create from dictionary."""
        data = {
            "401k": 1000,
            "roth": 500,
            "taxable": 300,
            "employer_match_rate": 0.5,
            "employer_match_cap": 500,
        }
        contrib = MonthlyContributions.from_dict(data)
        assert contrib.contrib_401k == 1000
        assert contrib.employer_match_rate == 0.5

    def test_scaled(self, default_contributions):
        """Scaling multiplies contribution amounts."""
        scaled = default_contributions.scaled(0.5)
        assert scaled.contrib_401k == 450
        assert scaled.contrib_roth == 200
        assert scaled.contrib_taxable == 150
        # Match settings unchanged
        assert scaled.employer_match_rate == 0.5


class TestGuardrails:
    """Tests for Guardrails withdrawal strategy."""

    def test_disabled_guardrails(self):
        """Disabled guardrails return base withdrawal."""
        gr = Guardrails(enabled=False)
        result = gr.apply(1000, 500_000, 400_000)
        assert result == 1000

    def test_upper_threshold_ceiling(self, default_guardrails):
        """Above upper threshold applies ceiling."""
        # Current 600k, target 400k = 1.5 ratio > 1.2 threshold
        result = default_guardrails.apply(1000, 600_000, 400_000)
        assert result == 1100  # 1000 * 1.10

    def test_lower_threshold_floor(self, default_guardrails):
        """Below lower threshold applies floor."""
        # Current 300k, target 400k = 0.75 ratio < 0.8 threshold
        result = default_guardrails.apply(1000, 300_000, 400_000)
        assert result == 950  # 1000 * 0.95

    def test_within_thresholds(self, default_guardrails):
        """Within thresholds returns base withdrawal."""
        # Current 400k, target 400k = 1.0 ratio (within 0.8-1.2)
        result = default_guardrails.apply(1000, 400_000, 400_000)
        assert result == 1000


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_total_months(self):
        """Total months is years * 12."""
        config = SimulationConfig(years=30, n_simulations=1000)
        assert config.total_months == 360

    def test_monthly_expense_drag(self):
        """Monthly expense drag calculation."""
        config = SimulationConfig(years=30, n_simulations=1000, expense_ratio=0.001)
        # 0.1% annual -> small monthly equivalent
        assert config.monthly_expense_drag > 0
        assert config.monthly_expense_drag < 0.0001


class TestRunSimulation:
    """Tests for the main simulation function."""

    def test_basic_simulation(
        self, default_balances, default_contributions, default_allocation, flat_returns
    ):
        """Basic simulation runs successfully."""
        result = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=default_contributions,
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
            retirement_months=12,
        )
        
        assert result.success_rate >= 0
        assert result.success_rate <= 1
        assert len(result.ending_balances) == 10
        assert result.account_paths.shape == (10, 13, 3)  # 12 months + 1, 3 accounts

    def test_accumulation_grows_balance(
        self, default_balances, default_contributions, default_allocation, flat_returns
    ):
        """Balance grows during accumulation (no retirement)."""
        result = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=default_contributions,
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
            retirement_months=None,  # No retirement
        )
        
        # Ending balance should be higher than starting
        initial = default_balances.total()
        assert all(result.ending_balances > initial)

    def test_retirement_withdrawals(
        self, default_balances, zero_contributions, default_allocation, zero_returns
    ):
        """Retirement withdrawals reduce balance."""
        result = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=zero_contributions,
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=zero_returns,
            retirement_months=0,  # Immediately retire
            withdrawal_rate=0.04,
        )
        
        # With zero returns and 4% withdrawal, balance should decrease
        initial = default_balances.total()
        assert all(result.ending_balances < initial)

    def test_expense_ratio_reduces_returns(
        self, default_balances, zero_contributions, default_allocation, flat_returns
    ):
        """Higher expense ratio results in lower ending balance."""
        result_low = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=zero_contributions,
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
            expense_ratio=0.001,
        )
        
        result_high = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=zero_contributions,
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
            expense_ratio=0.02,
        )
        
        # Lower expense ratio should have higher ending balance
        assert result_low.ending_balances.mean() > result_high.ending_balances.mean()

    def test_glide_path_integration(
        self, default_balances, default_contributions, flat_returns
    ):
        """Simulation works with glide path."""
        glide_path = create_default_glide_path()
        
        result = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=default_contributions,
            allocation=Allocation(us=0.6, vxus=0.3, sgov=0.1),  # Will be overridden
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
            glide_path=glide_path,
            retirement_months=6,
        )
        
        assert result.success_rate >= 0
        assert len(result.ending_balances) == 10

    def test_guardrails_integration(
        self, default_balances, zero_contributions, default_allocation, flat_returns
    ):
        """Simulation works with guardrails enabled."""
        guardrails = Guardrails(
            enabled=True,
            ceiling=1.10,
            floor=0.95,
            upper_threshold=1.20,
            lower_threshold=0.80,
        )
        
        result = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=zero_contributions,
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
            retirement_months=0,
            guardrails=guardrails,
        )
        
        assert result.success_rate >= 0

    def test_soft_retirement(
        self, default_balances, default_contributions, default_allocation, flat_returns
    ):
        """Soft retirement reduces contributions."""
        result_full = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=default_contributions,
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
            retirement_months=12,
            soft_retirement_months=None,
        )
        
        result_soft = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=default_contributions,
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
            retirement_months=12,
            soft_retirement_months=0,  # Soft retire from start
            soft_contribution_factor=0.5,
        )
        
        # Full contributions should result in higher balance
        assert result_full.ending_balances.mean() > result_soft.ending_balances.mean()

    def test_dict_backwards_compatibility(self, default_allocation, flat_returns):
        """Simulation accepts dict inputs for backward compatibility."""
        result = run_simulation(
            initial_balances={"401k": 100_000, "roth": 50_000, "taxable": 25_000},
            monthly_contributions={"401k": 1000, "roth": 500, "taxable": 300},
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
        )
        
        assert len(result.ending_balances) == 10

    def test_total_paths_included(
        self, default_balances, default_contributions, default_allocation, flat_returns
    ):
        """Result includes total_paths for risk metrics."""
        result = run_simulation(
            initial_balances=default_balances,
            monthly_contributions=default_contributions,
            allocation=default_allocation,
            years=1,
            n_simulations=10,
            scenario="Historical",
            asset_returns=flat_returns,
        )
        
        assert result.total_paths is not None
        assert result.total_paths.shape == (10, 13)  # n_sims, months+1
