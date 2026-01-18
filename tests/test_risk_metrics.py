"""Tests for risk metrics calculations."""

import numpy as np
import pytest

from core.risk_metrics import (
    calculate_max_drawdown,
    calculate_max_drawdowns,
    calculate_risk_metrics,
    estimate_perfect_withdrawal_rate,
)


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""

    def test_no_drawdown(self):
        """Monotonically increasing path has no drawdown."""
        path = np.array([100, 110, 120, 130, 140])
        dd = calculate_max_drawdown(path)
        assert dd == 0.0

    def test_single_drawdown(self):
        """Single drawdown is calculated correctly."""
        path = np.array([100, 110, 90, 95, 100])
        dd = calculate_max_drawdown(path)
        # Peak was 110, trough was 90: (110-90)/110 = 0.1818
        expected = (110 - 90) / 110
        assert abs(dd - expected) < 0.001

    def test_multiple_drawdowns(self):
        """Largest drawdown is returned."""
        path = np.array([100, 120, 100, 150, 90])
        dd = calculate_max_drawdown(path)
        # Peak 150, trough 90: (150-90)/150 = 0.4
        expected = (150 - 90) / 150
        assert abs(dd - expected) < 0.001

    def test_empty_path(self):
        """Empty path returns zero drawdown."""
        path = np.array([])
        dd = calculate_max_drawdown(path)
        assert dd == 0.0

    def test_flat_path(self):
        """Flat path has no drawdown."""
        path = np.array([100, 100, 100, 100])
        dd = calculate_max_drawdown(path)
        assert dd == 0.0


class TestMaxDrawdowns:
    """Tests for batch max drawdown calculation."""

    def test_multiple_paths(self):
        """Calculate drawdowns for multiple paths."""
        paths = np.array([
            [100, 110, 90, 100],  # 18.18% drawdown
            [100, 100, 100, 100],  # 0% drawdown
            [100, 50, 75, 60],  # 50% drawdown (100->50)
        ])
        drawdowns = calculate_max_drawdowns(paths)
        
        assert len(drawdowns) == 3
        assert abs(drawdowns[0] - 0.1818) < 0.01
        assert drawdowns[1] == 0.0
        assert abs(drawdowns[2] - 0.50) < 0.01


class TestPerfectWithdrawalRate:
    """Tests for perfect withdrawal rate estimation."""

    def test_successful_simulations(self):
        """High ending balances give positive PWR."""
        ending_balances = np.array([1_000_000] * 100)  # All end with $1M
        pwr = estimate_perfect_withdrawal_rate(
            ending_balances,
            initial_balance=500_000,
            years=30,
            target_success_rate=0.95,
        )
        assert pwr > 0

    def test_failed_simulations(self):
        """All negative endings give zero PWR."""
        ending_balances = np.array([-10_000] * 100)
        pwr = estimate_perfect_withdrawal_rate(
            ending_balances,
            initial_balance=500_000,
            years=30,
        )
        assert pwr == 0.0

    def test_zero_initial_balance(self):
        """Zero initial balance returns zero PWR."""
        ending_balances = np.array([100_000] * 100)
        pwr = estimate_perfect_withdrawal_rate(
            ending_balances,
            initial_balance=0,
            years=30,
        )
        assert pwr == 0.0


class TestRiskMetrics:
    """Tests for comprehensive risk metrics."""

    def test_perfect_success(self):
        """All successful simulations have 100% success rate."""
        n_sims = 100
        n_months = 360
        # Steadily growing paths
        total_paths = np.zeros((n_sims, n_months + 1))
        for i in range(n_months + 1):
            total_paths[:, i] = 100_000 + i * 100
        
        ending_balances = total_paths[:, -1]
        
        metrics = calculate_risk_metrics(
            total_paths=total_paths,
            ending_balances=ending_balances,
            annual_withdrawal=40_000,
            initial_balance=100_000,
            years=30,
        )
        
        assert metrics.success_rate == 1.0
        assert metrics.average_shortfall == 0.0
        assert all(p == 0.0 for p in metrics.probability_of_ruin_by_year)

    def test_partial_failure(self):
        """Mixed results give partial success rate."""
        n_sims = 100
        n_months = 12
        
        # Half succeed, half fail
        total_paths = np.zeros((n_sims, n_months + 1))
        total_paths[:50, :] = 100_000  # Successful
        total_paths[50:, :] = -10_000  # Failed
        
        ending_balances = total_paths[:, -1]
        
        metrics = calculate_risk_metrics(
            total_paths=total_paths,
            ending_balances=ending_balances,
            annual_withdrawal=40_000,
            initial_balance=100_000,
            years=1,
        )
        
        assert metrics.success_rate == 0.5
        assert metrics.average_shortfall == 10_000

    def test_drawdown_calculation(self):
        """Max drawdown is calculated correctly."""
        n_sims = 10
        n_months = 12
        
        # Create paths with known drawdown
        total_paths = np.ones((n_sims, n_months + 1)) * 100_000
        total_paths[:, 6] = 60_000  # 40% drop at month 6
        
        ending_balances = total_paths[:, -1]
        
        metrics = calculate_risk_metrics(
            total_paths=total_paths,
            ending_balances=ending_balances,
            annual_withdrawal=40_000,
            initial_balance=100_000,
            years=1,
        )
        
        assert abs(metrics.max_drawdown_median - 0.4) < 0.01

    def test_years_of_income(self):
        """Years of income calculation is correct."""
        n_sims = 10
        n_months = 12
        
        total_paths = np.ones((n_sims, n_months + 1)) * 200_000
        ending_balances = total_paths[:, -1]
        
        metrics = calculate_risk_metrics(
            total_paths=total_paths,
            ending_balances=ending_balances,
            annual_withdrawal=40_000,  # $200k / $40k = 5 years
            initial_balance=100_000,
            years=1,
        )
        
        assert abs(metrics.median_years_of_income - 5.0) < 0.1
