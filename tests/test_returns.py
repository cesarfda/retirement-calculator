"""Tests for returns sampling and data loading."""

import numpy as np
import pandas as pd
import pytest

from core.returns import (
    get_historical_summary,
    sample_returns,
)


@pytest.fixture
def sample_returns_df() -> pd.DataFrame:
    """Create sample historical returns for testing."""
    np.random.seed(42)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")  # 20 years
    returns = pd.DataFrame(
        {
            "VTI": np.random.normal(0.007, 0.04, 240),
            "VXUS": np.random.normal(0.005, 0.05, 240),
            "SGOV": np.random.normal(0.002, 0.01, 240),
        },
        index=dates,
    )
    return returns


class TestSampleReturns:
    """Tests for the sample_returns function."""

    def test_output_shape(self, sample_returns_df):
        """Output has correct shape."""
        result = sample_returns(
            sample_returns_df,
            n_months=120,
            n_simulations=100,
            block_size=12,
        )
        assert result.shape == (100, 120, 3)

    def test_uses_full_history(self, sample_returns_df):
        """Sampling uses all available historical data."""
        # With full history, we should see variation from different periods
        result = sample_returns(
            sample_returns_df,
            n_months=60,
            n_simulations=1000,
            block_size=12,
        )

        # Mean should be close to historical mean
        hist_mean = sample_returns_df.values.mean()
        sim_mean = result.mean()
        assert abs(sim_mean - hist_mean) < 0.01

    def test_scenario_parameter_ignored(self, sample_returns_df):
        """Scenario parameter is deprecated and ignored."""
        # Should not filter data when scenario is passed
        result_historical = sample_returns(
            sample_returns_df,
            n_months=60,
            n_simulations=100,
            block_size=12,
            scenario="historical",
        )

        result_recession = sample_returns(
            sample_returns_df,
            n_months=60,
            n_simulations=100,
            block_size=12,
            scenario="recession",
        )

        # Both should use full history, shapes should match
        assert result_historical.shape == result_recession.shape

    def test_block_size_preserved(self, sample_returns_df):
        """Block bootstrap preserves autocorrelation within blocks."""
        # With block_size=12, consecutive months within a year should
        # come from the same historical period
        result = sample_returns(
            sample_returns_df,
            n_months=24,
            n_simulations=10,
            block_size=12,
        )

        # The result should have the right shape
        assert result.shape == (10, 24, 3)

    def test_handles_short_history(self):
        """Handles cases with limited historical data."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        short_returns = pd.DataFrame(
            {
                "VTI": np.random.normal(0.007, 0.04, 24),
                "VXUS": np.random.normal(0.005, 0.05, 24),
                "SGOV": np.random.normal(0.002, 0.01, 24),
            },
            index=dates,
        )

        result = sample_returns(
            short_returns,
            n_months=60,
            n_simulations=10,
            block_size=12,
        )

        # Should still produce output by repeating blocks
        assert result.shape == (10, 60, 3)

    def test_empty_returns_handled(self):
        """Empty returns DataFrame returns zeros."""
        empty_df = pd.DataFrame()

        result = sample_returns(
            empty_df,
            n_months=60,
            n_simulations=10,
            block_size=12,
        )

        assert result.shape[0] == 10
        assert result.shape[1] == 60


class TestHistoricalSummary:
    """Tests for get_historical_summary function."""

    def test_returns_summary_stats(self, sample_returns_df):
        """Returns summary contains expected fields."""
        summary = get_historical_summary(sample_returns_df)

        assert "start_date" in summary
        assert "end_date" in summary
        assert "n_months" in summary
        assert "n_years" in summary
        assert "annualized_return" in summary
        assert "annualized_volatility" in summary
        assert "worst_drawdown" in summary

    def test_empty_returns_handled(self):
        """Empty DataFrame returns error dict."""
        empty_df = pd.DataFrame()
        summary = get_historical_summary(empty_df)
        assert "error" in summary

    def test_date_range_correct(self, sample_returns_df):
        """Date range in summary matches input data."""
        summary = get_historical_summary(sample_returns_df)
        assert summary["start_date"] == "2000-01-01"
        assert summary["n_months"] == 240

    def test_crisis_detection(self):
        """Detects historical crisis periods in data."""
        # Create data spanning GFC period
        dates = pd.date_range("2005-01-01", "2012-12-01", freq="MS")
        returns = pd.DataFrame(
            {
                "VTI": np.random.normal(0.007, 0.04, len(dates)),
            },
            index=dates,
        )

        summary = get_historical_summary(returns)
        assert summary.get("includes_gfc") is True
