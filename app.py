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
from core.returns import get_monthly_returns, load_embedded_returns, sample_returns
from core.risk_metrics import calculate_risk_metrics
from core.simulator import Guardrails, TaxConfig, run_simulation
from core.tax_config import CURRENT_IRS_LIMITS
from core.validation import validate_all
from utils.helpers import annual_to_monthly_rate, format_currency

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CACHE_DIR = Path("data/cache")
EMBEDDED_PATH = Path("data/historical_returns.json")
TICKERS = ["VTI", "VXUS", "SGOV"]

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
    current_age = st.number_input("Current age", min_value=18, max_value=80, value=35)
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
        value=150_000.0,
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
        value=25_000.0,
        key="balance_taxable",
        help_text="Enter a dollar amount (commas and $ optional).",
    )

    st.header("Monthly contributions")
    contrib_401k = currency_input(
        "401k contribution",
        value=900.0,
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
        value=300.0,
        key="contrib_taxable",
        help_text="Monthly contribution in dollars.",
    )
    st.subheader("Soft retirement contribution choices")
    continue_401k = st.checkbox("Continue 401k contributions", value=True)
    continue_roth = st.checkbox("Continue Roth IRA contributions", value=True)
    continue_taxable = st.checkbox("Continue after-tax contributions", value=True)
    soft_contribution_factor = st.slider(
        "Contributions after soft retirement",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
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
        us_alloc = st.slider("US (VTI)", min_value=0, max_value=100, value=60)
        vxus_alloc = st.slider("International (VXUS)", min_value=0, max_value=100, value=30)
        sgov_alloc = st.slider("Treasuries (SGOV)", min_value=0, max_value=100, value=10)

    st.header("Simulation")
    years = st.slider("Years to project", min_value=5, max_value=60, value=30)
    n_simulations = st.select_slider(
        "Number of simulations", options=[250, 500, 1000, 2500, 5000], value=1000
    )
    scenario = st.selectbox(
        "Market scenario",
        ["Historical", "Recession", "Lost Decade", "Bull", "Bear", "High Volatility"],
    )
    volatility_multiplier = st.slider(
        "High volatility multiplier", min_value=1.0, max_value=2.0, value=1.3, step=0.1
    )
    withdrawal_rate = st.slider(
        "Withdrawal rate", min_value=0.02, max_value=0.08, value=0.04, step=0.005
    )

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

        tax_config: TaxConfig | None = TaxConfig(
            enabled=True,
            filing_status=filing_status_code,
            cost_basis_ratio=cost_basis_ratio,
            enforce_rmd=enforce_rmd,
            enforce_contribution_limits=enforce_contribution_limits,
            tax_efficient_withdrawal=tax_efficient_withdrawal,
            irs_limits=CURRENT_IRS_LIMITS,
        )
    else:
        tax_config = None

    st.header("Inflation")
    inflation_rate = st.slider(
        "Annual inflation", min_value=0.0, max_value=0.05, value=0.025, step=0.005
    )
    adjust_for_inflation = st.toggle("Show inflation-adjusted results", value=True)

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

    if data_warning:
        st.warning("Data warnings:")
        for warning in data_warning:
            st.caption(f"- {warning}")

# =============================================================================
# SIMULATION
# =============================================================================

asset_returns = sample_returns(
    returns_df,
    n_months=years * 12,
    n_simulations=n_simulations,
    scenario=scenario,
    block_size=12,
    volatility_multiplier=volatility_multiplier,
)

retirement_months = max((retirement_age - current_age) * 12, 0)
soft_retirement_months = max((soft_retirement_age - current_age) * 12, 0)

result = run_simulation(
    initial_balances=balances_dict,
    monthly_contributions=contributions_dict,
    allocation=allocation,
    years=years,
    n_simulations=n_simulations,
    scenario=scenario,
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

fan_chart.update_layout(
    title="Portfolio Value Percentiles",
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
    if tax_features:
        active_strategies.append(f"Tax modeling ({', '.join(tax_features)})")

if active_strategies:
    st.info(f"**Active strategies:** {' | '.join(active_strategies)}")

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

with tabs[4]:
    st.subheader("Scenario settings")
    st.write(
        {
            "Years": years,
            "Simulations": n_simulations,
            "Scenario": scenario,
            "Withdrawal rate": f"{withdrawal_rate:.2%}",
            "Expense ratio": f"{expense_ratio:.3%}",
            "Inflation": f"{inflation_rate:.2%}" if adjust_for_inflation else "Not applied",
            "Soft retirement age": soft_retirement_age,
            "Glide path": glide_path_type if use_glide_path else "Disabled",
            "Guardrails": "Enabled" if use_guardrails else "Disabled",
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
