#!/usr/bin/env python3
"""Basic tests for retirement calculator core functionality."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.accounts import apply_employer_match
from core.portfolio import Allocation, allocation_from_dict, portfolio_returns
from core.simulator import run_simulation
from utils.helpers import annual_to_monthly_rate, format_currency


def test_allocation():
    """Test portfolio allocation functions."""
    print("Testing allocation...")
    alloc = allocation_from_dict({"US": 0.6, "VXUS": 0.3, "SGOV": 0.1})
    assert abs(alloc.us - 0.6) < 0.01
    assert abs(alloc.vxus - 0.3) < 0.01
    assert abs(alloc.sgov - 0.1) < 0.01
    assert abs(alloc.us + alloc.vxus + alloc.sgov - 1.0) < 0.01  # Should be normalized
    
    # Test normalization
    alloc2 = allocation_from_dict({"US": 60, "VXUS": 30, "SGOV": 10})
    assert abs(alloc2.us - 0.6) < 0.01
    assert abs(alloc2.vxus - 0.3) < 0.01
    assert abs(alloc2.sgov - 0.1) < 0.01
    assert abs(alloc2.us + alloc2.vxus + alloc2.sgov - 1.0) < 0.01  # Should be normalized
    print("  ✓ Allocation tests passed")


def test_helpers():
    """Test helper functions."""
    print("Testing helpers...")
    # Test annual to monthly rate
    annual = 0.12
    monthly = annual_to_monthly_rate(annual)
    expected = (1.12 ** (1/12)) - 1
    assert abs(monthly - expected) < 0.0001
    
    # Test currency formatting
    assert format_currency(1234.56) == "$1,235"
    assert format_currency(1000000) == "$1,000,000"
    print("  ✓ Helper tests passed")


def test_accounts():
    """Test account functions."""
    print("Testing accounts...")
    # Test employer match
    match = apply_employer_match(1000, 0.5, 500)
    assert match == 250.0  # min(1000, 500) * 0.5 = 250
    
    match = apply_employer_match(300, 0.5, 500)
    assert match == 150.0  # min(300, 500) * 0.5 = 150
    print("  ✓ Account tests passed")


def test_portfolio_returns():
    """Test portfolio returns calculation."""
    print("Testing portfolio returns...")
    alloc = Allocation(us=0.6, vxus=0.3, sgov=0.1)
    # Shape: (n_simulations, n_months, n_assets)
    asset_returns = np.random.randn(10, 12, 3) * 0.01
    portfolio_rets = portfolio_returns(asset_returns, alloc)
    assert portfolio_rets.shape == (10, 12)
    print("  ✓ Portfolio returns tests passed")


def test_simulation():
    """Test simulation with simple data."""
    print("Testing simulation...")
    alloc = Allocation(us=0.6, vxus=0.3, sgov=0.1)
    
    # Create simple mock returns (low volatility, positive returns)
    n_sims = 10
    n_months = 12
    asset_returns = np.ones((n_sims, n_months, 3)) * 0.005  # 0.5% per month
    
    result = run_simulation(
        initial_balances={"401k": 100000, "roth": 40000, "taxable": 25000},
        monthly_contributions={
            "401k": 900,
            "roth": 400,
            "taxable": 300,
            "employer_match_rate": 0.5,
            "employer_match_cap": 500,
        },
        allocation=alloc,
        years=1,
        n_simulations=n_sims,
        scenario="Historical",
        asset_returns=asset_returns,
        retirement_months=12,
    )
    
    assert result.success_rate >= 0
    assert result.success_rate <= 1
    assert len(result.ending_balances) == n_sims
    assert result.ending_balances.min() > 0  # Should have positive ending balance with contributions
    assert len(result.percentiles.p50) == n_months + 1
    print("  ✓ Simulation tests passed")


def main():
    """Run all tests."""
    print("Running basic tests for retirement calculator...\n")
    try:
        test_allocation()
        test_helpers()
        test_accounts()
        test_portfolio_returns()
        test_simulation()
        print("\n✅ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
