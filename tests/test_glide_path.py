"""Tests for glide path allocation strategy."""

import pytest

from core.glide_path import (
    GlidePath,
    create_default_glide_path,
    create_aggressive_glide_path,
    create_conservative_glide_path,
)
from core.portfolio import Allocation


class TestGlidePath:
    """Tests for GlidePath functionality."""

    def test_default_creation(self):
        """Default glide path creates successfully."""
        gp = create_default_glide_path()
        assert gp.start_equity == 0.90
        assert gp.retirement_equity == 0.50
        assert gp.end_equity == 0.60

    def test_equity_at_start(self, default_glide_path):
        """Equity at month 0 equals start_equity."""
        equity = default_glide_path.get_equity_at_month(0, 360, 600)
        assert abs(equity - default_glide_path.start_equity) < 0.01

    def test_equity_at_retirement(self, default_glide_path):
        """Equity at retirement equals retirement_equity."""
        retirement_month = 360
        equity = default_glide_path.get_equity_at_month(
            retirement_month, retirement_month, 600
        )
        assert abs(equity - default_glide_path.retirement_equity) < 0.01

    def test_equity_at_end(self, default_glide_path):
        """Equity at end equals end_equity."""
        equity = default_glide_path.get_equity_at_month(600, 360, 600)
        assert abs(equity - default_glide_path.end_equity) < 0.01

    def test_equity_decreases_pre_retirement(self, default_glide_path):
        """Equity decreases as retirement approaches."""
        retirement_month = 360
        total_months = 600
        
        equity_early = default_glide_path.get_equity_at_month(60, retirement_month, total_months)
        equity_mid = default_glide_path.get_equity_at_month(180, retirement_month, total_months)
        equity_late = default_glide_path.get_equity_at_month(300, retirement_month, total_months)
        
        assert equity_early > equity_mid > equity_late

    def test_equity_increases_post_retirement(self, default_glide_path):
        """Equity increases after retirement (bond tent)."""
        retirement_month = 360
        total_months = 600
        
        equity_retire = default_glide_path.get_equity_at_month(360, retirement_month, total_months)
        equity_later = default_glide_path.get_equity_at_month(480, retirement_month, total_months)
        
        assert equity_later > equity_retire

    def test_allocation_sums_to_one(self, default_glide_path):
        """Allocation always sums to 1.0."""
        for month in [0, 100, 200, 360, 400, 500, 600]:
            alloc = default_glide_path.get_allocation_at_month(month, 360, 600)
            total = alloc.us + alloc.vxus + alloc.sgov
            assert abs(total - 1.0) < 0.001

    def test_international_ratio(self, default_glide_path):
        """International allocation follows international_ratio."""
        alloc = default_glide_path.get_allocation_at_month(0, 360, 600)
        equity = alloc.us + alloc.vxus
        expected_international = equity * default_glide_path.international_ratio
        assert abs(alloc.vxus - expected_international) < 0.01

    def test_invalid_parameters_raise(self):
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            GlidePath(start_equity=1.5)  # Over 1.0
        
        with pytest.raises(ValueError):
            GlidePath(retirement_equity=-0.1)  # Negative

    def test_aggressive_vs_conservative(self):
        """Aggressive has higher equity than conservative."""
        aggressive = create_aggressive_glide_path()
        conservative = create_conservative_glide_path()
        
        assert aggressive.start_equity > conservative.start_equity
        assert aggressive.retirement_equity > conservative.retirement_equity
        assert aggressive.end_equity > conservative.end_equity
