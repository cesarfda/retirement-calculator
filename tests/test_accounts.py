"""Tests for account functions."""

import pytest

from core.accounts import apply_employer_match, AccountBalances, total_balance


class TestEmployerMatch:
    """Tests for employer match calculation."""

    def test_match_under_cap(self):
        """Contribution below cap gets full match."""
        match = apply_employer_match(300, 0.5, 500)
        assert match == 150.0  # min(300, 500) * 0.5 = 150

    def test_match_at_cap(self):
        """Contribution at cap gets capped match."""
        match = apply_employer_match(500, 0.5, 500)
        assert match == 250.0  # min(500, 500) * 0.5 = 250

    def test_match_over_cap(self):
        """Contribution over cap is capped."""
        match = apply_employer_match(1000, 0.5, 500)
        assert match == 250.0  # min(1000, 500) * 0.5 = 250

    def test_zero_match_rate(self):
        """Zero match rate gives zero match."""
        match = apply_employer_match(1000, 0.0, 500)
        assert match == 0.0

    def test_zero_cap(self):
        """Zero cap gives zero match."""
        match = apply_employer_match(1000, 0.5, 0)
        assert match == 0.0

    def test_full_match(self):
        """100% match rate."""
        match = apply_employer_match(500, 1.0, 500)
        assert match == 500.0


class TestAccountBalances:
    """Tests for AccountBalances dataclass."""

    def test_total_balance(self):
        """Test total balance calculation."""
        balances = AccountBalances(
            balance_401k=100_000,
            balance_roth=50_000,
            balance_taxable=25_000,
        )
        assert total_balance(balances) == 175_000

    def test_zero_balances(self):
        """Test with zero balances."""
        balances = AccountBalances(
            balance_401k=0.0,
            balance_roth=0.0,
            balance_taxable=0.0,
        )
        assert total_balance(balances) == 0.0
