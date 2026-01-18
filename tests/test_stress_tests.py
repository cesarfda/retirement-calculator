"""Tests for stress test functionality."""

import numpy as np
import pytest

from core.stress_tests import (
    AVAILABLE_STRESS_TESTS,
    STRESS_TEST_GFC_YEAR1,
    STRESS_TEST_GFC_RETIREMENT,
    STRESS_TEST_LOST_DECADE,
    StressTest,
    apply_stress_test,
    apply_multiple_stress_tests,
    get_stress_test_by_id,
    get_stress_test_by_name,
)


@pytest.fixture
def base_returns() -> np.ndarray:
    """Create base returns for testing stress test application."""
    np.random.seed(42)
    n_sims = 100
    n_months = 360  # 30 years
    n_assets = 3
    # Normal returns ~0.7% monthly with 4% volatility
    returns = np.random.normal(0.007, 0.04, (n_sims, n_months, n_assets))
    return returns


class TestStressTestDataclass:
    """Tests for StressTest dataclass."""

    def test_gfc_year1_values(self):
        """GFC Year 1 test has correct values."""
        assert STRESS_TEST_GFC_YEAR1.id == "gfc_year1"
        assert STRESS_TEST_GFC_YEAR1.shock_magnitude == -0.40
        assert STRESS_TEST_GFC_YEAR1.duration_months == 12
        assert STRESS_TEST_GFC_YEAR1.apply_at == 0

    def test_gfc_retirement_timing(self):
        """GFC at Retirement uses 'retirement' timing."""
        assert STRESS_TEST_GFC_RETIREMENT.apply_at == "retirement"

    def test_lost_decade_duration(self):
        """Lost decade spans 120 months (10 years)."""
        assert STRESS_TEST_LOST_DECADE.duration_months == 120

    def test_all_tests_have_unique_ids(self):
        """All stress tests have unique IDs."""
        ids = [st.id for st in AVAILABLE_STRESS_TESTS]
        assert len(ids) == len(set(ids))


class TestStressTestLookup:
    """Tests for stress test lookup functions."""

    def test_get_by_name(self):
        """Can look up stress test by name."""
        test = get_stress_test_by_name("GFC in Year 1")
        assert test is not None
        assert test.id == "gfc_year1"

    def test_get_by_id(self):
        """Can look up stress test by ID."""
        test = get_stress_test_by_id("gfc_year1")
        assert test is not None
        assert test.name == "GFC in Year 1"

    def test_unknown_name_returns_none(self):
        """Unknown name returns None."""
        test = get_stress_test_by_name("Unknown Test")
        assert test is None

    def test_unknown_id_returns_none(self):
        """Unknown ID returns None."""
        test = get_stress_test_by_id("unknown_id")
        assert test is None


class TestApplyStressTest:
    """Tests for apply_stress_test function."""

    def test_returns_copy(self, base_returns):
        """Stress test returns a copy, not modifying original."""
        original = base_returns.copy()
        result = apply_stress_test(base_returns, STRESS_TEST_GFC_YEAR1)

        np.testing.assert_array_equal(base_returns, original)
        assert not np.array_equal(result, base_returns)

    def test_same_shape(self, base_returns):
        """Result has same shape as input."""
        result = apply_stress_test(base_returns, STRESS_TEST_GFC_YEAR1)
        assert result.shape == base_returns.shape

    def test_gfc_reduces_returns_early(self, base_returns):
        """GFC Year 1 reduces returns in first 12 months."""
        result = apply_stress_test(base_returns, STRESS_TEST_GFC_YEAR1)

        # First 12 months should be lower than original
        early_original = base_returns[:, :12, :].mean()
        early_stressed = result[:, :12, :].mean()
        assert early_stressed < early_original

    def test_gfc_at_retirement_timing(self, base_returns):
        """GFC at Retirement applies at retirement month."""
        retirement_month = 180  # 15 years

        result = apply_stress_test(
            base_returns,
            STRESS_TEST_GFC_RETIREMENT,
            retirement_month=retirement_month,
        )

        # Returns before retirement should be unchanged
        pre_retirement = base_returns[:, :retirement_month, :]
        pre_retirement_stressed = result[:, :retirement_month, :]
        np.testing.assert_array_equal(pre_retirement, pre_retirement_stressed)

        # Returns at retirement should be reduced
        at_retirement = base_returns[:, retirement_month:retirement_month + 12, :].mean()
        at_retirement_stressed = result[:, retirement_month:retirement_month + 12, :].mean()
        assert at_retirement_stressed < at_retirement

    def test_lost_decade_affects_many_months(self, base_returns):
        """Lost decade affects first 120 months."""
        result = apply_stress_test(base_returns, STRESS_TEST_LOST_DECADE)

        # First 120 months should have reduced returns
        early_original = base_returns[:, :120, :].mean()
        early_stressed = result[:, :120, :].mean()
        assert early_stressed < early_original

        # Months after 120 should be unchanged
        late_original = base_returns[:, 120:, :]
        late_stressed = result[:, 120:, :]
        np.testing.assert_array_equal(late_original, late_stressed)

    def test_short_simulation_handled(self):
        """Stress test handles simulations shorter than stress duration."""
        short_returns = np.random.normal(0.007, 0.04, (10, 24, 3))  # Only 2 years

        # Lost decade (120 months) applied to 24 month simulation
        result = apply_stress_test(short_returns, STRESS_TEST_LOST_DECADE)

        # Should apply to all available months without error
        assert result.shape == short_returns.shape
        # All returns should be affected
        assert result.mean() < short_returns.mean()


class TestApplyMultipleStressTests:
    """Tests for apply_multiple_stress_tests function."""

    def test_returns_dict_of_results(self, base_returns):
        """Returns dictionary with result for each test."""
        tests = [STRESS_TEST_GFC_YEAR1, STRESS_TEST_LOST_DECADE]
        results = apply_multiple_stress_tests(base_returns, tests)

        assert len(results) == 2
        assert "gfc_year1" in results
        assert "lost_decade" in results

    def test_each_result_independent(self, base_returns):
        """Each stress test result is independent."""
        tests = [STRESS_TEST_GFC_YEAR1, STRESS_TEST_LOST_DECADE]
        results = apply_multiple_stress_tests(base_returns, tests)

        # Results should be different from each other
        assert not np.array_equal(results["gfc_year1"], results["lost_decade"])

    def test_empty_list_returns_empty_dict(self, base_returns):
        """Empty test list returns empty dict."""
        results = apply_multiple_stress_tests(base_returns, [])
        assert results == {}
