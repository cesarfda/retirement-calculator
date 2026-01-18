"""Tests for helper utility functions."""

import pytest

from utils.helpers import (
    annual_to_monthly_rate,
    format_currency,
    weighted_sum,
    Percentiles,
)


class TestAnnualToMonthlyRate:
    """Tests for rate conversion."""

    def test_zero_rate(self):
        """Zero annual rate gives zero monthly rate."""
        assert annual_to_monthly_rate(0.0) == 0.0

    def test_positive_rate(self):
        """Positive rate conversion is correct."""
        annual = 0.12
        monthly = annual_to_monthly_rate(annual)
        expected = (1.12 ** (1/12)) - 1
        assert abs(monthly - expected) < 0.0001

    def test_compounding(self):
        """Monthly rate compounds to annual rate."""
        annual = 0.10
        monthly = annual_to_monthly_rate(annual)
        # Compounding 12 months should give back annual rate
        compounded = (1 + monthly) ** 12 - 1
        assert abs(compounded - annual) < 0.0001


class TestFormatCurrency:
    """Tests for currency formatting."""

    def test_whole_dollars(self):
        """Whole dollar formatting."""
        assert format_currency(1000) == "$1,000"

    def test_with_cents(self):
        """Cents are rounded."""
        assert format_currency(1234.56) == "$1,235"

    def test_millions(self):
        """Large numbers with commas."""
        assert format_currency(1_000_000) == "$1,000,000"

    def test_zero(self):
        """Zero formats correctly."""
        assert format_currency(0) == "$0"

    def test_negative(self):
        """Negative numbers format correctly."""
        result = format_currency(-1000)
        assert "-" in result
        assert "1,000" in result


class TestWeightedSum:
    """Tests for weighted sum calculation."""

    def test_equal_weights(self):
        """Equal weights gives simple average."""
        values = [10, 20, 30]
        weights = [1, 1, 1]
        result = weighted_sum(values, weights)
        assert result == 60  # Sum, not average

    def test_weighted_calculation(self):
        """Weighted sum is correct."""
        values = [100, 200]
        weights = [0.3, 0.7]
        result = weighted_sum(values, weights)
        expected = 100 * 0.3 + 200 * 0.7  # 30 + 140 = 170
        assert result == expected

    def test_zero_weights(self):
        """Zero weights give zero sum."""
        values = [100, 200, 300]
        weights = [0, 0, 0]
        result = weighted_sum(values, weights)
        assert result == 0


class TestPercentiles:
    """Tests for Percentiles dataclass."""

    def test_creation(self):
        """Percentiles dataclass creates correctly."""
        p = Percentiles(
            p5=[1.0, 2.0],
            p25=[2.0, 3.0],
            p50=[3.0, 4.0],
            p75=[4.0, 5.0],
            p95=[5.0, 6.0],
        )
        assert p.p5 == [1.0, 2.0]
        assert p.p50 == [3.0, 4.0]

    def test_frozen(self):
        """Percentiles is immutable (frozen)."""
        p = Percentiles(
            p5=[1.0],
            p25=[2.0],
            p50=[3.0],
            p75=[4.0],
            p95=[5.0],
        )
        with pytest.raises(AttributeError):
            p.p5 = [10.0]
