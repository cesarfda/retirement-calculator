"""Tests for portfolio allocation and returns."""

import numpy as np
import pytest

from core.portfolio import Allocation, allocation_from_dict, portfolio_returns


class TestAllocation:
    """Tests for Allocation dataclass."""

    def test_from_dict_normalized(self):
        """Dict values are normalized to sum to 1."""
        alloc = allocation_from_dict({"US": 60, "VXUS": 30, "SGOV": 10})
        assert abs(alloc.us - 0.6) < 0.01
        assert abs(alloc.vxus - 0.3) < 0.01
        assert abs(alloc.sgov - 0.1) < 0.01
        assert abs(alloc.us + alloc.vxus + alloc.sgov - 1.0) < 0.001

    def test_from_dict_already_normalized(self):
        """Already normalized values stay the same."""
        alloc = allocation_from_dict({"US": 0.6, "VXUS": 0.3, "SGOV": 0.1})
        assert abs(alloc.us - 0.6) < 0.01
        assert abs(alloc.vxus - 0.3) < 0.01
        assert abs(alloc.sgov - 0.1) < 0.01

    def test_as_array(self):
        """Test conversion to numpy array."""
        alloc = Allocation(us=0.5, vxus=0.3, sgov=0.2)
        arr = alloc.as_array()
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3
        assert arr[0] == 0.5
        assert arr[1] == 0.3
        assert arr[2] == 0.2

    def test_normalized_zero_allocation(self):
        """Zero allocation normalizes to zeros."""
        alloc = Allocation(us=0, vxus=0, sgov=0).normalized()
        assert alloc.us == 0.0
        assert alloc.vxus == 0.0
        assert alloc.sgov == 0.0

    def test_normalized_unbalanced(self):
        """Unbalanced allocation gets normalized."""
        alloc = Allocation(us=100, vxus=50, sgov=50).normalized()
        assert abs(alloc.us - 0.5) < 0.01
        assert abs(alloc.vxus - 0.25) < 0.01
        assert abs(alloc.sgov - 0.25) < 0.01


class TestPortfolioReturns:
    """Tests for portfolio returns calculation."""

    def test_weighted_returns(self, default_allocation):
        """Portfolio returns are weighted sum of asset returns."""
        # Shape: (n_simulations, n_months, n_assets)
        asset_returns = np.array([[[0.10, 0.05, 0.02]]])  # Single month, single sim
        
        result = portfolio_returns(asset_returns, default_allocation)
        
        # Expected: 0.6 * 0.10 + 0.3 * 0.05 + 0.1 * 0.02 = 0.077
        expected = 0.6 * 0.10 + 0.3 * 0.05 + 0.1 * 0.02
        assert result.shape == (1, 1)
        assert abs(result[0, 0] - expected) < 0.001

    def test_returns_shape(self, default_allocation, flat_returns):
        """Output shape is (n_sims, n_months)."""
        result = portfolio_returns(flat_returns, default_allocation)
        assert result.shape == (10, 12)  # n_sims=10, n_months=12 from fixture

    def test_zero_returns(self, default_allocation, zero_returns):
        """Zero asset returns give zero portfolio returns."""
        result = portfolio_returns(zero_returns, default_allocation)
        assert np.allclose(result, 0.0)
