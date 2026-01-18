"""Retirement Calculator with Monte Carlo Simulation - Streamlit App."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.data_validation import (
    get_cache_status,
    validate_returns_data,
)
from core.exceptions import DataFetchError
from core.glide_path import (
    GlidePath,
    create_aggressive_glide_path,
    create_conservative_glide_path,
    create_default_glide_path,
)
from core.portfolio import Allocation, allocation_from_dict
from core.returns import get_historical_summary, get_monthly_returns, load_embedded_returns, sample_returns
from core.risk_metrics import calculate_risk_metrics
from core.simulator import Guardrails, TaxConfig, run_simulation
from core.stress_tests import (
    AVAILABLE_STRESS_TESTS,
    apply_stress_test,
    get_stress_test_by_name,
)
from core.tax_config import CURRENT_IRS_LIMITS
from core.validation import validate_all
from utils.helpers import annual_to_monthly_rate, format_currency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CACHE_DIR = Path("data/cache")
EMBEDDED_PATH = Path("data/historical_returns.json")

# Use longer-history proxies for better simulation accuracy:
# - SPY: S&P 500 (1993) - proxy for US total market (VTI inception 2001)
# - EFA: Developed Markets ex-US (2001) - proxy for international (VXUS inception 2011)
# - SHY: 1-3 Year Treasury (2002) - proxy for short-term bonds (SGOV inception 2020)
TICKERS = ["SPY", "EFA", "SHY"]
TICKER_DISPLAY_NAMES = {
    "SPY": "US Stocks (SPY→VTI)",
    "EFA": "International (EFA→VXUS)",
    "SHY": "Treasuries (SHY→SGOV)",
}

st.set_page_config(page_title="Retirement Calculator", layout="wide")

st.title("Retirement Calculator with Monte Carlo Simulation")


def parse_currency(value: str, fallback: float) -> float:
    """Parse a currency string to float."""
    cleaned = value.replace("$", "").replace(",", "").strip()
    if not cleaned:
        return fallback
    try:
        return float(cleaned)
    except ValueError:
        return fallback


def currency_input(label: str, value: float, key: str, help_text: str | None = None) -> float:
    """Create a currency input field."""
    raw_value = st.text_input(
        label,
        key=key,
        help=help_text,
        value=st.session_state.get(key, format_currency(value)),
    )
    parsed_value = parse_currency(raw_value, value)
    return parsed_value


# =============================================================================
# SIDEBAR INPUTS
# =============================================================================

with st.sidebar:
    st.header("Profile")
    current_age = st.number_input("Current age", min_value=18, max_value=80, value=29)
    retirement_age = st.number_input(
        "Retirement age", min_value=current_age + 1, max_value=90, value=65
    )
    soft_retirement_age = st.number_input(
        "Soft retirement age",
        min_value=current_age + 1,
        max_value=retirement_age,
        value=max(min(current_age + 10, retirement_age), current_age + 1),
    )

    st.header("Current balances")
    balance_401k = currency_input(
        "401k balance",
        value=190_000.0,
        key="balance_401k",
        help_text="Enter a dollar amount (commas and $ optional).",
    )
    balance_roth = currency_input(
        "Roth IRA balance",
        value=40_000.0,
        key="balance_roth",
        help_text="Enter a dollar amount (commas and $ optional).",
    )
    balance_taxable = currency_input(
        "After-tax balance",
        value=250_000.0,
        key="balance_taxable",
        help_text="Enter a dollar amount (commas and $ optional).",
    )

    st.header("Monthly contributions")
    contrib_401k = currency_input(
        "401k contribution",
        value=2000.0,
        key="contrib_401k",
        help_text="Monthly contribution in dollars.",
    )
    contrib_roth = currency_input(
        "Roth IRA contribution",
        value=400.0,
        key="contrib_roth",
        help_text="Monthly contribution in dollars.",
    )
    contrib_taxable = currency_input(
        "After-tax contribution",
        value=5_000.0,
        key="contrib_taxable",
        help_text="Monthly contribution in dollars.",
    )
    st.subheader("Soft retirement contribution choices")
    continue_401k = st.checkbox("Continue 401k contributions", value=True)
    continue_roth = st.checkbox("Continue Roth IRA contributions", value=True)
    continue_taxable = st.checkbox("Continue after-tax contributions", value=False)
    soft_contribution_factor = st.slider(
        "Contributions after soft retirement",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="Scale selected contributions after the soft retirement age "
        "(1.0 keeps contributions unchanged).",
    )

    st.header("Employer match")
    match_rate = st.slider("Match rate", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    match_cap = currency_input(
        "Match cap (monthly)",
        value=500.0,
        key="match_cap",
        help_text="Monthly employer match cap in dollars.",
    )

    st.header("Asset allocation")
    use_glide_path = st.toggle(
        "Use glide path",
        value=False,
        help="Automatically adjust allocation over time (bond tent strategy).",
    )

    if use_glide_path:
        glide_path_type = st.selectbox(
            "Glide path style",
            ["Default", "Aggressive", "Conservative", "Custom"],
            help="Default: 90% equity -> 50% at retirement -> 60% late. "
            "Aggressive: Higher equity throughout. "
            "Conservative: Lower equity throughout.",
        )

        if glide_path_type == "Custom":
            gp_start = st.slider(
                "Starting equity %", min_value=20, max_value=100, value=90
            )
            gp_retire = st.slider(
                "Retirement equity %", min_value=20, max_value=100, value=50
            )
            gp_end = st.slider("Ending equity %", min_value=20, max_value=100, value=60)
            gp_intl = st.slider(
                "International ratio",
                min_value=0.0,
                max_value=0.5,
                value=0.33,
                step=0.05,
            )
            glide_path: GlidePath | None = GlidePath(
                start_equity=gp_start / 100,
                retirement_equity=gp_retire / 100,
                end_equity=gp_end / 100,
                international_ratio=gp_intl,
            )
        elif glide_path_type == "Aggressive":
            glide_path = create_aggressive_glide_path()
        elif glide_path_type == "Conservative":
            glide_path = create_conservative_glide_path()
        else:
            glide_path = create_default_glide_path()

        # Show glide path info
        st.caption(
            f"Equity: {glide_path.start_equity:.0%} -> "
            f"{glide_path.retirement_equity:.0%} -> "
            f"{glide_path.end_equity:.0%}"
        )
        # Use glide path's starting allocation for static display
        us_alloc = int((1 - glide_path.international_ratio) * glide_path.start_equity * 100)
        vxus_alloc = int(glide_path.international_ratio * glide_path.start_equity * 100)
        sgov_alloc = 100 - us_alloc - vxus_alloc
    else:
        glide_path = None
        us_alloc = st.slider("US Stocks (SPY)", min_value=0, max_value=100, value=30)
        vxus_alloc = st.slider("International (EFA)", min_value=0, max_value=100, value=65)
        sgov_alloc = st.slider("Treasuries (SHY)", min_value=0, max_value=100, value=5)

    st.header("Simulation")
    years = st.slider("Years to project", min_value=5, max_value=71, value=70)
    n_simulations = st.select_slider(
        "Number of simulations", options=[250, 500, 1000, 2500, 5000], value=1000
    )
    withdrawal_rate = st.slider(
        "Withdrawal rate", min_value=0.02, max_value=0.08, value=0.04, step=0.005
    )

    st.header("Stress Tests")
    st.caption(
        "Main simulation uses full historical data (including crashes). "
        "Stress tests show additional worst-case scenarios for comparison."
    )
    run_stress_tests = st.toggle(
        "Run stress test comparisons",
        value=False,
        help="Run additional simulations with specific adverse scenarios overlaid.",
    )

    selected_stress_tests: list[str] = []
    if run_stress_tests:
        stress_test_options = [st.name for st in AVAILABLE_STRESS_TESTS]
        selected_stress_tests = st.multiselect(
            "Select stress scenarios",
            options=stress_test_options,
            default=["GFC in Year 1", "GFC at Retirement"],
            help="These scenarios will be shown as additional lines on the chart.",
        )

        # Show descriptions for selected tests
        for test_name in selected_stress_tests:
            test = get_stress_test_by_name(test_name)
            if test:
                st.caption(f"**{test.name}**: {test.description}")

    st.header("Costs")
    expense_ratio = st.slider(
        "Expense ratio",
        min_value=0.0,
        max_value=0.02,
        value=0.001,
        step=0.001,
        format="%.3f",
        help="Annual fund expense ratio. VTI: 0.03%, VXUS: 0.08%, SGOV: 0.09%. "
        "Use weighted average or higher for active funds.",
    )
    st.caption(f"Annual drag: {expense_ratio:.2%}")

    st.header("Guardrails")
    use_guardrails = st.toggle(
        "Enable guardrails",
        value=False,
        help="Adjust withdrawals based on portfolio performance. "
        "Increase spending when doing well, reduce when struggling.",
    )

    if use_guardrails:
        gr_ceiling = st.slider(
            "Max spending increase", min_value=1.0, max_value=1.25, value=1.10, step=0.05
        )
        gr_floor = st.slider(
            "Max spending decrease", min_value=0.80, max_value=1.0, value=0.95, step=0.05
        )
        gr_upper = st.slider(
            "Upper threshold", min_value=1.05, max_value=1.50, value=1.20, step=0.05
        )
        gr_lower = st.slider(
            "Lower threshold", min_value=0.50, max_value=0.95, value=0.80, step=0.05
        )
        guardrails: Guardrails | None = Guardrails(
            enabled=True,
            ceiling=gr_ceiling,
            floor=gr_floor,
            upper_threshold=gr_upper,
            lower_threshold=gr_lower,
        )
    else:
        guardrails = None

    st.header("Tax Modeling")
    enable_tax_modeling = st.toggle(
        "Enable tax modeling",
        value=False,
        help="Model RMDs, contribution limits, and tax-efficient withdrawals.",
    )

    if enable_tax_modeling:
        filing_status = st.selectbox(
            "Filing status",
            ["Single", "Married Filing Jointly"],
            help="Tax filing status affects tax brackets.",
        )
        filing_status_code = "single" if filing_status == "Single" else "mfj"

        cost_basis_ratio = st.slider(
            "Taxable account cost basis",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Fraction of taxable account that is original investment (not taxed on withdrawal).",
        )

        enforce_contribution_limits = st.toggle(
            "Enforce IRS contribution limits",
            value=True,
            help=f"Cap contributions at IRS limits (401k: ${CURRENT_IRS_LIMITS.limit_401k:,.0f}, "
            f"catch-up: ${CURRENT_IRS_LIMITS.limit_401k_catchup:,.0f} at age 50+).",
        )

        enforce_rmd = st.toggle(
            "Enforce Required Minimum Distributions",
            value=True,
            help=f"Require minimum 401k withdrawals starting at age {CURRENT_IRS_LIMITS.rmd_start_age}.",
        )

        tax_efficient_withdrawal = st.toggle(
            "Use tax-efficient withdrawal order",
            value=False,
            help="Withdraw from taxable first, then 401k, then Roth (preserves tax-advantaged growth).",
        )

        # Show contribution limit info
        age_50_plus = current_age >= 50
        annual_401k_limit = CURRENT_IRS_LIMITS.get_401k_limit(current_age)
        annual_roth_limit = CURRENT_IRS_LIMITS.get_roth_ira_limit(current_age)
        monthly_401k_max = annual_401k_limit / 12
        monthly_roth_max = annual_roth_limit / 12

        st.caption(
            f"Your limits: 401k ${annual_401k_limit:,.0f}/yr (${monthly_401k_max:,.0f}/mo) | "
            f"Roth ${annual_roth_limit:,.0f}/yr (${monthly_roth_max:,.0f}/mo)"
            + (" [catch-up eligible]" if age_50_plus else "")
        )

        # Warn if contributions exceed limits
        annual_401k_contrib = contrib_401k * 12
        annual_roth_contrib = contrib_roth * 12
        if annual_401k_contrib > annual_401k_limit:
            st.warning(
                f"401k contribution (${annual_401k_contrib:,.0f}/yr) exceeds limit (${annual_401k_limit:,.0f}/yr)"
            )
        if annual_roth_contrib > annual_roth_limit:
            st.warning(
                f"Roth contribution (${annual_roth_contrib:,.0f}/yr) exceeds limit (${annual_roth_limit:,.0f}/yr)"
            )

        # Show RMD info if applicable
        if retirement_age >= CURRENT_IRS_LIMITS.rmd_start_age:
            st.info(
                f"RMDs will begin at age {CURRENT_IRS_LIMITS.rmd_start_age}. "
                "You must withdraw a minimum amount from 401k each year."
            )

        # Roth Conversion Strategy
        st.subheader("Roth Conversion Strategy")
        enable_roth_conversion = st.toggle(
            "Enable Roth conversions",
            value=False,
            help="Convert 401k to Roth during low-income years to reduce future RMDs and taxes.",
        )

        roth_conversion_strategy = None
        if enable_roth_conversion:
            roth_target_bracket = st.selectbox(
                "Target tax bracket to fill",
                options=[0.10, 0.12, 0.22, 0.24],
                format_func=lambda x: f"{x:.0%} bracket",
                index=2,  # Default to 22%
                help="Convert enough to fill up to this marginal tax bracket.",
            )
            roth_max_conversion = st.number_input(
                "Max annual conversion",
                min_value=10000,
                max_value=200000,
                value=50000,
                step=5000,
                help="Maximum amount to convert per year.",
            )
            roth_start_age = st.number_input(
                "Start conversions at age",
                min_value=current_age,
                max_value=72,
                value=max(soft_retirement_age, current_age),
                help="Typically start when income drops (soft retirement).",
            )
            roth_stop_age = st.number_input(
                "Stop conversions at age",
                min_value=roth_start_age + 1,
                max_value=73,
                value=72,
                help="Stop before RMDs begin at 73.",
            )

            from core.roth_conversion import RothConversionStrategy
            roth_conversion_strategy = RothConversionStrategy(
                enabled=True,
                target_bracket=roth_target_bracket,
                max_annual_conversion=float(roth_max_conversion),
                start_age=roth_start_age,
                stop_age=roth_stop_age,
            )

            st.caption(
                f"Conversions will occur annually from age {roth_start_age} to {roth_stop_age}, "
                f"converting up to ${roth_max_conversion:,.0f}/year to fill the {roth_target_bracket:.0%} bracket."
            )

        # NIIT info
        st.caption(
            "Note: Net Investment Income Tax (3.8%) automatically applies to investment income "
            "above $200k (single) / $250k (married)."
        )

        tax_config: TaxConfig | None = TaxConfig(
            enabled=True,
            filing_status=filing_status_code,
            cost_basis_ratio=cost_basis_ratio,
            enforce_rmd=enforce_rmd,
            enforce_contribution_limits=enforce_contribution_limits,
            tax_efficient_withdrawal=tax_efficient_withdrawal,
            irs_limits=CURRENT_IRS_LIMITS,
            roth_conversion=roth_conversion_strategy,
        )
    else:
        tax_config = None

    st.header("Inflation")
    inflation_rate = st.slider(
        "Annual inflation", min_value=0.0, max_value=0.05, value=0.025, step=0.005
    )
    adjust_for_inflation = st.toggle("Show inflation-adjusted results", value=True)

    st.header("Return Modeling")
    use_fat_tails = st.toggle(
        "Use fat-tailed returns",
        value=False,
        help="Use Student-t distribution instead of historical bootstrap. "
        "Better captures extreme market events.",
    )

    fat_tail_df = 5  # Default degrees of freedom
    if use_fat_tails:
        fat_tail_df = st.slider(
            "Tail fatness (degrees of freedom)",
            min_value=3,
            max_value=30,
            value=5,
            help="Lower = fatter tails (more extreme events). "
            "5 is typical for equities, 30 is nearly normal.",
        )

    use_valuation_adjustment = st.toggle(
        "Adjust for current valuations",
        value=False,
        help="Reduce expected equity returns when CAPE is elevated above historical median.",
    )

    valuation_adjustment = None
    if use_valuation_adjustment:
        from core.returns import ValuationAdjustment

        st.caption("**US Market (S&P 500)**")
        cape_cols = st.columns(2)
        with cape_cols[0]:
            us_cape = st.number_input(
                "US CAPE",
                min_value=10.0,
                max_value=50.0,
                value=30.0,
                step=1.0,
                help="Current Shiller P/E for S&P 500. Historical median ~16.",
            )
        with cape_cols[1]:
            us_cape_median = st.number_input(
                "US historical median",
                min_value=10.0,
                max_value=25.0,
                value=16.0,
                step=0.5,
                help="Long-term median CAPE for US market.",
            )

        st.caption("**International Market (EAFE)**")
        intl_cols = st.columns(2)
        with intl_cols[0]:
            intl_cape = st.number_input(
                "Intl CAPE",
                min_value=8.0,
                max_value=40.0,
                value=16.0,
                step=1.0,
                help="Current CAPE for international developed markets. Historical median ~14.",
            )
        with intl_cols[1]:
            intl_cape_median = st.number_input(
                "Intl historical median",
                min_value=8.0,
                max_value=20.0,
                value=14.0,
                step=0.5,
                help="Long-term median CAPE for international markets.",
            )

        valuation_factor = st.slider(
            "Adjustment strength",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = no adjustment, 1 = full adjustment based on CAPE premium.",
        )

        valuation_adjustment = ValuationAdjustment(
            enabled=True,
            us_cape=us_cape,
            us_cape_median=us_cape_median,
            intl_cape=intl_cape,
            intl_cape_median=intl_cape_median,
            adjustment_factor=valuation_factor,
        )

        us_annual_drag = valuation_adjustment.calculate_us_monthly_drag() * 12
        intl_annual_drag = valuation_adjustment.calculate_intl_monthly_drag() * 12

        st.caption(
            f"US equity return reduced by {us_annual_drag:.1%}/year | "
            f"Intl equity return reduced by {intl_annual_drag:.1%}/year"
        )

    st.header("Visualization")
    display_months = st.slider(
        "Months to display",
        min_value=12,
        max_value=years * 12,
        value=years * 12,
        step=12,
    )
    percentile_options = ["5th", "25th", "50th", "75th", "95th"]
    selected_percentiles = st.multiselect(
        "Percentile lines",
        options=percentile_options,
        default=["50th", "25th", "75th"],
    )
    if not selected_percentiles:
        selected_percentiles = ["50th"]
    show_retirement_marker = st.toggle("Highlight retirement month", value=True)
    show_soft_retirement_marker = st.toggle("Highlight soft retirement month", value=True)
    show_percentile_bands = st.toggle("Show percentile bands", value=True)
    show_account_area = st.toggle("Show account breakdown", value=True)
    show_success_chart = st.toggle("Show success over time", value=True)
    show_risk_metrics = st.toggle("Show detailed risk metrics", value=True)

# =============================================================================
# VALIDATION
# =============================================================================

allocation = allocation_from_dict(
    {"US": us_alloc / 100, "VXUS": vxus_alloc / 100, "SGOV": sgov_alloc / 100}
)

balances_dict = {"401k": balance_401k, "roth": balance_roth, "taxable": balance_taxable}
contributions_dict = {
    "401k": contrib_401k,
    "roth": contrib_roth,
    "taxable": contrib_taxable,
    "employer_match_rate": match_rate,
    "employer_match_cap": match_cap,
}

validation_result = validate_all(
    current_age=current_age,
    retirement_age=retirement_age,
    soft_retirement_age=soft_retirement_age,
    balances=balances_dict,
    contributions=contributions_dict,
    allocation=allocation,
    years=years,
    n_simulations=n_simulations,
    withdrawal_rate=withdrawal_rate,
    expense_ratio=expense_ratio,
)

if not validation_result.is_valid():
    st.error("Please fix the following input errors:")
    for error_msg in validation_result.error_messages():
        st.warning(error_msg)
    st.stop()

# =============================================================================
# DATA LOADING WITH ERROR HANDLING
# =============================================================================

# Check cache status
cache_status = get_cache_status(CACHE_DIR / "returns.json")
data_warning = None

try:
    returns_df = get_monthly_returns(TICKERS, CACHE_DIR, EMBEDDED_PATH)

    # Validate data quality
    data_validation = validate_returns_data(returns_df, TICKERS)
    if not data_validation.is_valid():
        for error in data_validation.errors:
            st.error(f"Data error: {error}")
        st.warning("Using embedded fallback data due to data quality issues.")
        returns_df = load_embedded_returns(EMBEDDED_PATH)
    elif data_validation.has_warnings():
        data_warning = data_validation.warnings

except DataFetchError as e:
    st.error(f"Could not load market data: {e}")
    st.info("Using embedded historical data as fallback.")
    returns_df = load_embedded_returns(EMBEDDED_PATH)
except Exception as e:
    logger.error(f"Unexpected error loading data: {e}")
    st.error("An error occurred loading market data. Using embedded fallback.")
    returns_df = load_embedded_returns(EMBEDDED_PATH)

# Show data status in sidebar expander
with st.sidebar.expander("Data Status", expanded=False):
    if cache_status.exists:
        st.write(f"Cache age: {cache_status.age_days:.1f} days")
        st.write(f"Cache size: {cache_status.size_kb:.1f} KB")
        st.write(f"Status: {'Fresh' if cache_status.is_fresh else 'Stale'}")
    else:
        st.write("Cache: Not available")
        st.write("Using: Embedded data")

    # Show historical data summary
    hist_summary = get_historical_summary(returns_df)
    if "error" not in hist_summary:
        st.caption(f"**Data range**: {hist_summary['start_date']} to {hist_summary['end_date']}")
        st.caption(f"**History**: {hist_summary['n_years']:.1f} years ({hist_summary['n_months']} months)")
        included_events = []
        if hist_summary.get("includes_gfc"):
            included_events.append("GFC 2008")
        if hist_summary.get("includes_dotcom"):
            included_events.append("Dot-com crash")
        if hist_summary.get("includes_covid"):
            included_events.append("COVID 2020")
        if included_events:
            st.caption(f"**Includes**: {', '.join(included_events)}")

        # Show expected returns per ticker
        if "ticker_stats" in hist_summary:
            st.caption("**Expected annual returns**:")
            for ticker, stats in hist_summary["ticker_stats"].items():
                display_name = TICKER_DISPLAY_NAMES.get(ticker, ticker)
                ann_return = stats["annualized_return"]
                ann_vol = stats["annualized_volatility"]
                st.caption(f"  {display_name}: {ann_return:.1%} (±{ann_vol:.1%})")

    if data_warning:
        st.warning("Data warnings:")
        for warning in data_warning:
            st.caption(f"- {warning}")

# =============================================================================
# SIMULATION
# =============================================================================

# Get historical data summary
historical_summary = get_historical_summary(returns_df)

# Sample returns based on selected method
if use_fat_tails:
    from core.returns import sample_returns_student_t
    asset_returns = sample_returns_student_t(
        returns_df,
        n_months=years * 12,
        n_simulations=n_simulations,
        degrees_of_freedom=fat_tail_df,
    )
else:
    # Use block bootstrap from FULL historical data
    asset_returns = sample_returns(
        returns_df,
        n_months=years * 12,
        n_simulations=n_simulations,
        block_size=12,
    )

# Apply valuation adjustment if enabled
if use_valuation_adjustment and valuation_adjustment is not None:
    from core.returns import apply_valuation_adjustment
    asset_returns = apply_valuation_adjustment(
        asset_returns,
        valuation_adjustment,
        us_column=0,     # US stocks (SPY)
        intl_column=1,   # International (EFA)
    )

retirement_months = max((retirement_age - current_age) * 12, 0)
soft_retirement_months = max((soft_retirement_age - current_age) * 12, 0)

# Run stress test simulations if enabled
stress_test_results: dict = {}
if run_stress_tests and selected_stress_tests:
    for test_name in selected_stress_tests:
        test = get_stress_test_by_name(test_name)
        if test:
            # Apply stress test to a copy of the returns
            stressed_returns = apply_stress_test(
                asset_returns,
                test,
                retirement_month=retirement_months,
            )
            # Run simulation with stressed returns
            stress_result = run_simulation(
                initial_balances=balances_dict,
                monthly_contributions=contributions_dict,
                allocation=allocation,
                years=years,
                n_simulations=n_simulations,
                scenario="historical",  # Backward compat
                asset_returns=stressed_returns,
                retirement_months=retirement_months,
                soft_retirement_months=soft_retirement_months,
                soft_contribution_factor=soft_contribution_factor,
                soft_contribution_factors={
                    "401k": soft_contribution_factor if continue_401k else 0.0,
                    "roth": soft_contribution_factor if continue_roth else 0.0,
                    "taxable": soft_contribution_factor if continue_taxable else 0.0,
                },
                withdrawal_rate=withdrawal_rate,
                expense_ratio=expense_ratio,
                guardrails=guardrails,
                glide_path=glide_path,
                current_age=current_age,
                tax_config=tax_config,
            )
            stress_test_results[test.id] = {
                "test": test,
                "result": stress_result,
            }

# Run main simulation with full historical data
result = run_simulation(
    initial_balances=balances_dict,
    monthly_contributions=contributions_dict,
    allocation=allocation,
    years=years,
    n_simulations=n_simulations,
    scenario="historical",  # Always use full history
    asset_returns=asset_returns,
    retirement_months=retirement_months,
    soft_retirement_months=soft_retirement_months,
    soft_contribution_factor=soft_contribution_factor,
    soft_contribution_factors={
        "401k": soft_contribution_factor if continue_401k else 0.0,
        "roth": soft_contribution_factor if continue_roth else 0.0,
        "taxable": soft_contribution_factor if continue_taxable else 0.0,
    },
    withdrawal_rate=withdrawal_rate,
    expense_ratio=expense_ratio,
    guardrails=guardrails,
    glide_path=glide_path,
    current_age=current_age,
    tax_config=tax_config,
)

# Calculate enhanced risk metrics
initial_balance = balance_401k + balance_roth + balance_taxable
annual_withdrawal = initial_balance * withdrawal_rate
risk_metrics = calculate_risk_metrics(
    total_paths=result.total_paths,
    ending_balances=result.ending_balances,
    annual_withdrawal=annual_withdrawal,
    initial_balance=initial_balance,
    years=years,
)

# =============================================================================
# DATA PROCESSING
# =============================================================================

months = np.arange(0, years * 12 + 1)
display_mask = months <= display_months
ages = current_age + months / 12
display_age_axis = ages[display_mask]

if adjust_for_inflation and inflation_rate > 0:
    monthly_inflation = annual_to_monthly_rate(inflation_rate)
    inflation_factors = (1 + monthly_inflation) ** months
else:
    inflation_factors = np.ones_like(months, dtype=float)

# =============================================================================
# CHARTS
# =============================================================================

fan_chart = go.Figure()
percentile_series = {
    "p95": np.array(result.percentiles.p95) / inflation_factors,
    "p75": np.array(result.percentiles.p75) / inflation_factors,
    "p50": np.array(result.percentiles.p50) / inflation_factors,
    "p25": np.array(result.percentiles.p25) / inflation_factors,
    "p5": np.array(result.percentiles.p5) / inflation_factors,
}
percentile_lookup = {
    "5th": "p5",
    "25th": "p25",
    "50th": "p50",
    "75th": "p75",
    "95th": "p95",
}
percentile_styles = {
    "5th": {"color": "#6c757d", "dash": "dot"},
    "25th": {"color": "#adb5bd", "dash": "dash"},
    "50th": {"color": "#0d6efd", "dash": "solid"},
    "75th": {"color": "#adb5bd", "dash": "dash"},
    "95th": {"color": "#6c757d", "dash": "dot"},
}

for label in selected_percentiles:
    percentile_key = percentile_lookup[label]
    style = percentile_styles[label]
    fan_chart.add_trace(
        go.Scatter(
            x=display_age_axis,
            y=percentile_series[percentile_key][display_mask],
            line=dict(
                color=style["color"],
                dash=style["dash"],
                width=3 if percentile_key == "p50" else 2,
            ),
            name=label,
        )
    )

if show_percentile_bands:
    fan_chart.add_traces(
        [
            go.Scatter(
                x=display_age_axis,
                y=percentile_series["p95"][display_mask],
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=display_age_axis,
                y=percentile_series["p75"][display_mask],
                fill="tonexty",
                fillcolor="rgba(0, 123, 255, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="75-95%",
            ),
            go.Scatter(
                x=display_age_axis,
                y=percentile_series["p25"][display_mask],
                fill="tonexty",
                fillcolor="rgba(0, 123, 255, 0.25)",
                line=dict(color="rgba(0,0,0,0)"),
                name="25-75%",
            ),
            go.Scatter(
                x=display_age_axis,
                y=percentile_series["p5"][display_mask],
                fill="tonexty",
                fillcolor="rgba(0, 123, 255, 0.35)",
                line=dict(color="rgba(0,0,0,0)"),
                name="5-25%",
            ),
        ]
    )

# Add stress test median lines
if stress_test_results:
    for test_id, stress_data in stress_test_results.items():
        test = stress_data["test"]
        stress_result = stress_data["result"]

        # Get median (50th percentile) path for stress test
        stress_median = np.array(stress_result.percentiles.p50) / inflation_factors

        fan_chart.add_trace(
            go.Scatter(
                x=display_age_axis,
                y=stress_median[display_mask],
                line=dict(
                    color=test.color,
                    dash="dash",
                    width=2,
                ),
                name=f"{test.name} (median)",
            )
        )

if show_retirement_marker and retirement_months <= display_months:
    fan_chart.add_vline(
        x=current_age + retirement_months / 12,
        line_dash="dot",
        line_color="gray",
        annotation_text="Retirement",
        annotation_position="top left",
    )

if (
    show_soft_retirement_marker
    and soft_retirement_months <= display_months
    and soft_retirement_months < retirement_months
):
    fan_chart.add_vline(
        x=current_age + soft_retirement_months / 12,
        line_dash="dash",
        line_color="#6f42c1",
        annotation_text="Soft retirement",
        annotation_position="top left",
    )

stress_test_info = ""
if stress_test_results:
    stress_test_info = f" + {len(stress_test_results)} stress test(s)"

fan_chart.update_layout(
    title=f"Portfolio Value Percentiles (Full Historical Data{stress_test_info})",
    xaxis_title="Age",
    yaxis_title="Balance",
    hovermode="x unified",
    xaxis=dict(dtick=5, tickformat=".1f"),
)

ending_balances = result.ending_balances / inflation_factors[-1]

histogram = go.Figure(
    data=[go.Histogram(x=ending_balances, nbinsx=30, marker_color="#0d6efd")]
)
histogram.update_layout(
    title="Ending Balance Distribution", xaxis_title="Balance", yaxis_title="Count"
)

account_paths = result.account_paths.mean(axis=0) / inflation_factors[:, None]
account_chart = go.Figure()
if show_account_area:
    account_chart.add_traces(
        [
            go.Scatter(
                x=display_age_axis,
                y=account_paths[:, 0][display_mask],
                stackgroup="one",
                name="401k",
            ),
            go.Scatter(
                x=display_age_axis,
                y=account_paths[:, 1][display_mask],
                stackgroup="one",
                name="Roth IRA",
            ),
            go.Scatter(
                x=display_age_axis,
                y=account_paths[:, 2][display_mask],
                stackgroup="one",
                name="After-tax",
            ),
        ]
    )
else:
    account_chart.add_traces(
        [
            go.Scatter(
                x=display_age_axis,
                y=account_paths[:, 0][display_mask],
                name="401k",
            ),
            go.Scatter(
                x=display_age_axis,
                y=account_paths[:, 1][display_mask],
                name="Roth IRA",
            ),
            go.Scatter(
                x=display_age_axis,
                y=account_paths[:, 2][display_mask],
                name="After-tax",
            ),
        ]
    )
account_chart.update_layout(
    title="Account Growth (Average)",
    xaxis_title="Age",
    yaxis_title="Balance",
    xaxis=dict(dtick=5, tickformat=".1f"),
)

total_paths = result.total_paths / inflation_factors[None, :]
success_over_time = (total_paths > 0).mean(axis=0)
success_chart = go.Figure(
    data=[
        go.Scatter(
            x=display_age_axis,
            y=success_over_time[display_mask] * 100,
            line=dict(color="#198754", width=3),
            name="Success rate",
        )
    ]
)
success_chart.update_layout(
    title="Probability of Success Over Time",
    xaxis_title="Age",
    yaxis_title="Success Rate (%)",
    yaxis_range=[0, 100],
    xaxis=dict(dtick=5, tickformat=".1f"),
)

# Drawdown chart
drawdown_years = list(range(1, years + 1))
ruin_probs = [p * 100 for p in risk_metrics.probability_of_ruin_by_year]
ruin_chart = go.Figure(
    data=[
        go.Bar(
            x=[current_age + y for y in drawdown_years],
            y=ruin_probs,
            marker_color="#dc3545",
            name="Ruin probability",
        )
    ]
)
ruin_chart.update_layout(
    title="Probability of Ruin by Age",
    xaxis_title="Age",
    yaxis_title="Ruin Probability (%)",
    yaxis_range=[0, max(100, max(ruin_probs) * 1.1) if ruin_probs else 100],
    xaxis=dict(dtick=5),
)

# =============================================================================
# DISPLAY
# =============================================================================

# Summary metrics - top row
summary_cols = st.columns(4)
summary_cols[0].metric("Success rate", f"{result.success_rate * 100:.1f}%")
summary_cols[1].metric("Median ending balance", format_currency(np.median(ending_balances)))
summary_cols[2].metric(
    "5th percentile ending", format_currency(np.percentile(ending_balances, 5))
)
if retirement_months < len(percentile_series["p50"]):
    median_at_retirement = percentile_series["p50"][retirement_months]
    summary_cols[3].metric(
        "Median at retirement",
        format_currency(median_at_retirement),
        help="Median projected balance at the retirement month.",
    )
else:
    summary_cols[3].metric("Median at retirement", "N/A")

# Risk metrics - second row (if enabled)
if show_risk_metrics:
    risk_cols = st.columns(4)
    risk_cols[0].metric(
        "Max Drawdown (median)",
        f"{risk_metrics.max_drawdown_median * 100:.1f}%",
        help="Median maximum peak-to-trough decline across simulations.",
    )
    risk_cols[1].metric(
        "Max Drawdown (worst 5%)",
        f"{risk_metrics.max_drawdown_worst * 100:.1f}%",
        help="95th percentile of maximum drawdowns (worst case scenarios).",
    )
    risk_cols[2].metric(
        "Years of income remaining",
        f"{risk_metrics.median_years_of_income:.1f}",
        help="Median ending balance divided by annual withdrawal amount.",
    )
    if risk_metrics.perfect_withdrawal_rate > 0:
        risk_cols[3].metric(
            "Safe withdrawal rate (est.)",
            f"{risk_metrics.perfect_withdrawal_rate * 100:.2f}%",
            help="Estimated highest withdrawal rate with 95% success.",
        )
    else:
        risk_cols[3].metric("Safe withdrawal rate", "N/A")

    # Enhanced metrics - third row
    enhanced_cols = st.columns(4)
    enhanced_cols[0].metric(
        "CVaR 95%",
        format_currency(risk_metrics.cvar_95 / inflation_factors[-1]) if adjust_for_inflation else format_currency(risk_metrics.cvar_95),
        help="Expected Shortfall: Average ending balance in worst 5% of scenarios.",
    )
    enhanced_cols[1].metric(
        "CVaR 99%",
        format_currency(risk_metrics.cvar_99 / inflation_factors[-1]) if adjust_for_inflation else format_currency(risk_metrics.cvar_99),
        help="Expected Shortfall: Average ending balance in worst 1% of scenarios.",
    )
    if risk_metrics.legacy_metrics:
        enhanced_cols[2].metric(
            "P(Leave $500k+)",
            f"{risk_metrics.legacy_metrics.prob_leave_500k * 100:.0f}%",
            help="Probability of leaving at least $500k to heirs.",
        )
        enhanced_cols[3].metric(
            "Expected Legacy",
            format_currency(risk_metrics.legacy_metrics.expected_legacy / inflation_factors[-1]) if adjust_for_inflation else format_currency(risk_metrics.legacy_metrics.expected_legacy),
            help="Expected (average) ending balance across successful simulations.",
        )

    # Spending flexibility info
    if risk_metrics.spending_flexibility and risk_metrics.spending_flexibility.improvement > 0:
        st.caption(
            f"**Spending flexibility bonus:** If you can cut spending by 10% during downturns, "
            f"success rate could improve by ~{risk_metrics.spending_flexibility.improvement * 100:.1f} percentage points."
        )

# Strategy indicators
active_strategies = []
if use_glide_path:
    active_strategies.append(f"Glide path ({glide_path_type})")
if use_guardrails:
    active_strategies.append(f"Guardrails (ceiling: {gr_ceiling:.0%}, floor: {gr_floor:.0%})")
if enable_tax_modeling:
    tax_features = []
    if enforce_contribution_limits:
        tax_features.append("limits")
    if enforce_rmd:
        tax_features.append("RMD")
    if tax_efficient_withdrawal:
        tax_features.append("tax-efficient")
    if 'enable_roth_conversion' in dir() and enable_roth_conversion:
        tax_features.append("Roth conversion")
    if tax_features:
        active_strategies.append(f"Tax modeling ({', '.join(tax_features)})")
if use_fat_tails:
    active_strategies.append(f"Fat-tailed returns (df={fat_tail_df})")
if use_valuation_adjustment:
    active_strategies.append("Valuation-adjusted returns")

if active_strategies:
    st.info(f"**Active strategies:** {' | '.join(active_strategies)}")

# Stress test info
if stress_test_results:
    stress_names = [data["test"].name for data in stress_test_results.values()]
    st.info(f"**Stress tests shown:** {', '.join(stress_names)} (dashed lines on chart)")

# Tabs
tabs = st.tabs(["Overview", "Distribution", "Accounts", "Risk Analysis", "Assumptions"])

with tabs[0]:
    st.plotly_chart(fan_chart, use_container_width=True)
    if show_success_chart:
        st.plotly_chart(success_chart, use_container_width=True)
    allocation_chart = go.Figure(
        data=[
            go.Pie(
                labels=["US (VTI)", "International (VXUS)", "Treasuries (SGOV)"],
                values=[us_alloc, vxus_alloc, sgov_alloc],
                hole=0.4,
            )
        ]
    )
    allocation_chart.update_layout(
        title="Asset Allocation" + (" (Starting)" if use_glide_path else ""),
        legend_title="Assets",
    )
    st.plotly_chart(allocation_chart, use_container_width=True)

with tabs[1]:
    st.plotly_chart(histogram, use_container_width=True)

with tabs[2]:
    st.plotly_chart(account_chart, use_container_width=True)

with tabs[3]:
    st.subheader("Risk Analysis")
    st.plotly_chart(ruin_chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Drawdown Statistics")
        st.write(
            {
                "Median max drawdown": f"{risk_metrics.max_drawdown_median:.1%}",
                "Worst-case max drawdown (5%)": f"{risk_metrics.max_drawdown_worst:.1%}",
            }
        )

    with col2:
        st.subheader("Ending Balance Analysis")
        st.write(
            {
                "Best case (95th percentile)": format_currency(
                    risk_metrics.best_case_ending / inflation_factors[-1]
                ),
                "Worst case (5th percentile)": format_currency(
                    risk_metrics.worst_case_ending / inflation_factors[-1]
                ),
                "Avg shortfall (if failed)": format_currency(
                    risk_metrics.average_shortfall / inflation_factors[-1]
                )
                if risk_metrics.average_shortfall > 0
                else "N/A",
            }
        )

    # CVaR Analysis
    st.subheader("Tail Risk Analysis (CVaR / Expected Shortfall)")
    cvar_cols = st.columns(2)
    with cvar_cols[0]:
        st.write(
            {
                "CVaR 95% (worst 5% average)": format_currency(
                    risk_metrics.cvar_95 / inflation_factors[-1]
                ),
                "CVaR 99% (worst 1% average)": format_currency(
                    risk_metrics.cvar_99 / inflation_factors[-1]
                ),
            }
        )
        st.caption(
            "CVaR (Conditional Value at Risk) shows the average ending balance in the worst scenarios. "
            "Unlike VaR, it captures the severity of tail losses, not just the threshold."
        )

    with cvar_cols[1]:
        if risk_metrics.spending_flexibility:
            st.write(
                {
                    "Base success rate": f"{risk_metrics.spending_flexibility.base_success_rate:.1%}",
                    "With 10% spending flexibility": f"{risk_metrics.spending_flexibility.flexible_success_rate:.1%}",
                    "Improvement": f"+{risk_metrics.spending_flexibility.improvement:.1%}",
                }
            )
            st.caption(
                "Spending flexibility shows how much your success rate improves if you can reduce "
                "spending by 10% during market downturns."
            )

    # Legacy Analysis
    if risk_metrics.legacy_metrics:
        st.subheader("Legacy / Bequest Analysis")
        legacy_cols = st.columns(3)
        with legacy_cols[0]:
            st.metric(
                "P(Leave $100k+)",
                f"{risk_metrics.legacy_metrics.prob_leave_100k:.0%}",
            )
        with legacy_cols[1]:
            st.metric(
                "P(Leave $500k+)",
                f"{risk_metrics.legacy_metrics.prob_leave_500k:.0%}",
            )
        with legacy_cols[2]:
            st.metric(
                "P(Leave $1M+)",
                f"{risk_metrics.legacy_metrics.prob_leave_1m:.0%}",
            )

        st.write(
            {
                "Expected legacy (mean)": format_currency(
                    risk_metrics.legacy_metrics.expected_legacy / inflation_factors[-1]
                ),
                "Median legacy": format_currency(
                    risk_metrics.legacy_metrics.median_legacy / inflation_factors[-1]
                ),
                "Conservative legacy (10th percentile)": format_currency(
                    risk_metrics.legacy_metrics.legacy_at_risk / inflation_factors[-1]
                ),
            }
        )
        st.caption(
            "Legacy metrics show the probability and expected amount you would leave to heirs. "
            "The 10th percentile is a conservative estimate for planning purposes."
        )

with tabs[4]:
    st.subheader("Simulation settings")

    # Historical data info
    if "error" not in historical_summary:
        st.caption(
            f"Using {historical_summary['n_years']:.1f} years of market history "
            f"({historical_summary['start_date']} to {historical_summary['end_date']})"
        )

    st.write(
        {
            "Years": years,
            "Simulations": n_simulations,
            "Return sampling": "Full historical (block bootstrap)",
            "Withdrawal rate": f"{withdrawal_rate:.2%}",
            "Expense ratio": f"{expense_ratio:.3%}",
            "Inflation": f"{inflation_rate:.2%}" if adjust_for_inflation else "Not applied",
            "Soft retirement age": soft_retirement_age,
            "Glide path": glide_path_type if use_glide_path else "Disabled",
            "Guardrails": "Enabled" if use_guardrails else "Disabled",
        }
    )

    # Show expected returns per ticker in assumptions
    if "error" not in historical_summary and "ticker_stats" in historical_summary:
        st.subheader("Expected Returns (Historical)")
        ticker_returns_data = {}
        for ticker, stats in historical_summary["ticker_stats"].items():
            display_name = TICKER_DISPLAY_NAMES.get(ticker, ticker)
            ticker_returns_data[display_name] = {
                "Annual return": f"{stats['annualized_return']:.1%}",
                "Volatility": f"{stats['annualized_volatility']:.1%}",
            }
        st.write(ticker_returns_data)
        st.caption(
            "Returns are based on historical data and may not reflect future performance. "
            "Volatility shown as annualized standard deviation."
        )

    # Show stress test details if enabled
    if stress_test_results:
        st.subheader("Stress Tests")
        for test_id, stress_data in stress_test_results.items():
            test = stress_data["test"]
            stress_result = stress_data["result"]
            st.write(
                {
                    "Name": test.name,
                    "Description": test.description,
                    "Timing": f"Month {test.apply_at}" if isinstance(test.apply_at, int) else test.apply_at.title(),
                    "Shock magnitude": f"{test.shock_magnitude:.0%}",
                    "Duration": f"{test.duration_months} months",
                    "Success rate": f"{stress_result.success_rate * 100:.1f}%",
                }
            )
    st.subheader("Contribution summary")
    st.write(
        {
            "401k": format_currency(contrib_401k),
            "Roth IRA": format_currency(contrib_roth),
            "After-tax": format_currency(contrib_taxable),
            "Post-soft retirement contributions": (
                f"401k: {soft_contribution_factor:.0%}" if continue_401k else "401k: 0%"
            )
            + (f" | Roth IRA: {soft_contribution_factor:.0%}" if continue_roth else " | Roth IRA: 0%")
            + (
                f" | After-tax: {soft_contribution_factor:.0%}"
                if continue_taxable
                else " | After-tax: 0%"
            ),
            "Employer match": f"{match_rate:.0%} up to {format_currency(match_cap)}",
        }
    )

    if use_glide_path and glide_path is not None:
        st.subheader("Glide Path Details")
        st.write(
            {
                "Starting equity": f"{glide_path.start_equity:.0%}",
                "Retirement equity": f"{glide_path.retirement_equity:.0%}",
                "Ending equity": f"{glide_path.end_equity:.0%}",
                "International ratio": f"{glide_path.international_ratio:.0%}",
            }
        )

    if use_guardrails and guardrails is not None:
        st.subheader("Guardrails Details")
        st.write(
            {
                "Spending ceiling": f"{guardrails.ceiling:.0%}",
                "Spending floor": f"{guardrails.floor:.0%}",
                "Upper threshold": f"{guardrails.upper_threshold:.0%}",
                "Lower threshold": f"{guardrails.lower_threshold:.0%}",
            }
        )

    if enable_tax_modeling and tax_config is not None:
        st.subheader("Tax Modeling Details")
        st.write(
            {
                "Filing status": filing_status,
                "Cost basis ratio": f"{cost_basis_ratio:.0%}",
                "Contribution limits": "Enforced" if enforce_contribution_limits else "Disabled",
                "RMD enforcement": "Enabled" if enforce_rmd else "Disabled",
                "Tax-efficient withdrawal": "Enabled" if tax_efficient_withdrawal else "Disabled",
                "401k annual limit": format_currency(CURRENT_IRS_LIMITS.get_401k_limit(current_age)),
                "Roth IRA annual limit": format_currency(CURRENT_IRS_LIMITS.get_roth_ira_limit(current_age)),
                "RMD start age": CURRENT_IRS_LIMITS.rmd_start_age,
            }
        )
