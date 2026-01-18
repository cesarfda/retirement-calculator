"""Shared pytest fixtures for retirement calculator tests."""

import numpy as np
import pytest

from core.portfolio import Allocation
from core.simulator import AccountBalances, MonthlyContributions, Guardrails
from core.glide_path import GlidePath, create_default_glide_path


@pytest.fixture
def default_allocation() -> Allocation:
    """Standard 60/30/10 allocation."""
    return Allocation(us=0.6, vxus=0.3, sgov=0.1)


@pytest.fixture
def conservative_allocation() -> Allocation:
    """Conservative 40/20/40 allocation."""
    return Allocation(us=0.4, vxus=0.2, sgov=0.4)


@pytest.fixture
def default_balances() -> AccountBalances:
    """Standard starting balances."""
    return AccountBalances(
        balance_401k=150_000.0,
        balance_roth=40_000.0,
        balance_taxable=25_000.0,
    )


@pytest.fixture
def zero_balances() -> AccountBalances:
    """Zero starting balances."""
    return AccountBalances(
        balance_401k=0.0,
        balance_roth=0.0,
        balance_taxable=0.0,
    )


@pytest.fixture
def default_contributions() -> MonthlyContributions:
    """Standard monthly contributions."""
    return MonthlyContributions(
        contrib_401k=900.0,
        contrib_roth=400.0,
        contrib_taxable=300.0,
        employer_match_rate=0.5,
        employer_match_cap=500.0,
    )


@pytest.fixture
def zero_contributions() -> MonthlyContributions:
    """Zero contributions."""
    return MonthlyContributions(
        contrib_401k=0.0,
        contrib_roth=0.0,
        contrib_taxable=0.0,
        employer_match_rate=0.0,
        employer_match_cap=0.0,
    )


@pytest.fixture
def default_guardrails() -> Guardrails:
    """Standard guardrails configuration."""
    return Guardrails(
        enabled=True,
        ceiling=1.10,
        floor=0.95,
        upper_threshold=1.20,
        lower_threshold=0.80,
    )


@pytest.fixture
def disabled_guardrails() -> Guardrails:
    """Disabled guardrails."""
    return Guardrails(enabled=False)


@pytest.fixture
def default_glide_path() -> GlidePath:
    """Standard glide path."""
    return create_default_glide_path()


@pytest.fixture
def flat_returns() -> np.ndarray:
    """Flat 0.5% monthly returns for predictable testing."""
    n_sims = 10
    n_months = 12
    n_assets = 3
    return np.ones((n_sims, n_months, n_assets)) * 0.005


@pytest.fixture
def zero_returns() -> np.ndarray:
    """Zero returns for testing contribution-only scenarios."""
    n_sims = 10
    n_months = 12
    n_assets = 3
    return np.zeros((n_sims, n_months, n_assets))


@pytest.fixture
def volatile_returns() -> np.ndarray:
    """Volatile returns with known seed for reproducibility."""
    np.random.seed(42)
    n_sims = 100
    n_months = 120  # 10 years
    n_assets = 3
    # Monthly returns with realistic volatility
    return np.random.normal(0.007, 0.04, (n_sims, n_months, n_assets))


@pytest.fixture
def crash_returns() -> np.ndarray:
    """Returns simulating a market crash in year 1."""
    n_sims = 10
    n_months = 60  # 5 years
    n_assets = 3
    returns = np.ones((n_sims, n_months, n_assets)) * 0.005
    # 40% crash spread over first 6 months
    returns[:, 0:6, :] = -0.08
    return returns
