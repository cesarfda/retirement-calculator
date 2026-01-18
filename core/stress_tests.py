"""Stress test scenarios for retirement simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class StressTest:
    """
    Definition of a stress test scenario to overlay on simulated returns.

    Stress tests inject adverse market conditions at specific points in the
    simulation to test portfolio resilience under extreme scenarios.

    Attributes:
        id: Unique identifier for the stress test
        name: Human-readable name
        description: Detailed description of the scenario
        apply_at: When to apply the stress (month number or "retirement")
        shock_magnitude: Total portfolio decline (e.g., -0.40 for 40% crash)
        duration_months: How many months the stress event spans
        recovery_drag: Monthly return drag during recovery (e.g., -0.005)
        color: Color for visualization
    """

    id: str
    name: str
    description: str
    apply_at: int | Literal["retirement"]
    shock_magnitude: float
    duration_months: int
    recovery_drag: float = 0.0
    color: str = "#dc3545"  # Bootstrap danger red


# Predefined stress test scenarios based on historical events
STRESS_TEST_GFC_YEAR1 = StressTest(
    id="gfc_year1",
    name="GFC in Year 1",
    description="2008-style 40% crash spread over 12 months starting immediately",
    apply_at=0,
    shock_magnitude=-0.40,
    duration_months=12,
    recovery_drag=-0.005,
    color="#dc3545",
)

STRESS_TEST_GFC_RETIREMENT = StressTest(
    id="gfc_retirement",
    name="GFC at Retirement",
    description="2008-style 40% crash occurring right at retirement start",
    apply_at="retirement",
    shock_magnitude=-0.40,
    duration_months=12,
    recovery_drag=-0.005,
    color="#fd7e14",
)

STRESS_TEST_LOST_DECADE = StressTest(
    id="lost_decade",
    name="Lost Decade",
    description="10 years of near-zero real returns (similar to 2000-2010)",
    apply_at=0,
    shock_magnitude=0.0,
    duration_months=120,
    recovery_drag=-0.005,  # ~6% annual drag
    color="#6f42c1",
)

STRESS_TEST_STAGFLATION = StressTest(
    id="stagflation",
    name="Stagflation",
    description="5 years of high inflation with low growth (1970s-style)",
    apply_at=0,
    shock_magnitude=-0.15,
    duration_months=60,
    recovery_drag=-0.003,
    color="#20c997",
)

STRESS_TEST_COVID_CRASH = StressTest(
    id="covid_crash",
    name="COVID-style Crash",
    description="Sharp 35% crash over 2 months with quick recovery",
    apply_at=0,
    shock_magnitude=-0.35,
    duration_months=2,
    recovery_drag=0.0,
    color="#0dcaf0",
)

STRESS_TEST_SEQUENCE_RISK = StressTest(
    id="sequence_risk",
    name="Sequence of Returns Risk",
    description="Poor returns in first 5 years of retirement",
    apply_at="retirement",
    shock_magnitude=-0.20,
    duration_months=60,
    recovery_drag=-0.003,
    color="#ffc107",
)

# All available stress tests
AVAILABLE_STRESS_TESTS: list[StressTest] = [
    STRESS_TEST_GFC_YEAR1,
    STRESS_TEST_GFC_RETIREMENT,
    STRESS_TEST_LOST_DECADE,
    STRESS_TEST_STAGFLATION,
    STRESS_TEST_COVID_CRASH,
    STRESS_TEST_SEQUENCE_RISK,
]

# Lookup by ID
STRESS_TESTS_BY_ID: dict[str, StressTest] = {st.id: st for st in AVAILABLE_STRESS_TESTS}

# Lookup by name
STRESS_TESTS_BY_NAME: dict[str, StressTest] = {st.name: st for st in AVAILABLE_STRESS_TESTS}


def get_stress_test_by_name(name: str) -> StressTest | None:
    """Get a stress test by its human-readable name."""
    return STRESS_TESTS_BY_NAME.get(name)


def get_stress_test_by_id(id: str) -> StressTest | None:
    """Get a stress test by its ID."""
    return STRESS_TESTS_BY_ID.get(id)


def apply_stress_test(
    returns: np.ndarray,
    stress_test: StressTest,
    retirement_month: int | None = None,
) -> np.ndarray:
    """
    Apply a stress test to sampled returns.

    Args:
        returns: Sampled returns array of shape (n_simulations, n_months, n_assets)
        stress_test: Stress test to apply
        retirement_month: Month when retirement begins (for "retirement" timing)

    Returns:
        Modified returns array with stress test applied
    """
    result = returns.copy()
    n_sims, n_months, n_assets = result.shape

    # Determine start month
    if stress_test.apply_at == "retirement":
        if retirement_month is None:
            # Default to middle of simulation if retirement not specified
            start_month = n_months // 2
        else:
            start_month = retirement_month
    else:
        start_month = stress_test.apply_at

    # Ensure we don't go past the end of the simulation
    end_month = min(start_month + stress_test.duration_months, n_months)
    actual_duration = end_month - start_month

    if actual_duration <= 0:
        return result

    # Apply the shock spread over the duration
    if stress_test.shock_magnitude != 0:
        # Spread the total shock over the duration
        # Using log returns, we need to adjust so total effect = shock_magnitude
        monthly_shock = stress_test.shock_magnitude / actual_duration
        result[:, start_month:end_month, :] += monthly_shock

    # Apply recovery drag (additional monthly penalty)
    if stress_test.recovery_drag != 0:
        result[:, start_month:end_month, :] += stress_test.recovery_drag

    return result


def apply_multiple_stress_tests(
    returns: np.ndarray,
    stress_tests: list[StressTest],
    retirement_month: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Apply multiple stress tests to returns, returning results for each.

    Args:
        returns: Base sampled returns array
        stress_tests: List of stress tests to apply
        retirement_month: Month when retirement begins

    Returns:
        Dict mapping stress test ID to modified returns array
    """
    results = {}
    for stress_test in stress_tests:
        results[stress_test.id] = apply_stress_test(
            returns, stress_test, retirement_month
        )
    return results
