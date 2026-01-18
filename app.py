from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.portfolio import allocation_from_dict
from core.returns import get_monthly_returns, sample_returns
from core.simulator import run_simulation
from utils.helpers import annual_to_monthly_rate, format_currency


CACHE_DIR = Path("data/cache")
EMBEDDED_PATH = Path("data/historical_returns.json")
TICKERS = ["VTI", "VXUS", "SGOV"]

st.set_page_config(page_title="Retirement Calculator", layout="wide")

st.title("Retirement Calculator with Monte Carlo Simulation")


def parse_currency(value: str, fallback: float) -> float:
    cleaned = value.replace("$", "").replace(",", "").strip()
    if not cleaned:
        return fallback
    try:
        return float(cleaned)
    except ValueError:
        return fallback


def currency_input(label: str, value: float, key: str, help_text: str | None = None) -> float:
    if key not in st.session_state:
        st.session_state[key] = format_currency(value)
    raw_value = st.text_input(label, key=key, help=help_text)
    parsed_value = parse_currency(raw_value, value)
    formatted_value = format_currency(parsed_value)
    if raw_value != formatted_value:
        st.session_state[key] = formatted_value
    return parsed_value

with st.sidebar:
    st.header("Profile")
    current_age = st.number_input("Current age", min_value=18, max_value=80, value=35)
    retirement_age = st.number_input("Retirement age", min_value=current_age + 1, max_value=90, value=65)
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
    soft_contribution_factor = st.slider(
        "Contributions after soft retirement",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Scale contributions after the soft retirement age (1.0 keeps contributions unchanged).",
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
    us_alloc = st.slider("US (VTI)", min_value=0, max_value=100, value=60)
    vxus_alloc = st.slider("International (VXUS)", min_value=0, max_value=100, value=30)
    sgov_alloc = st.slider("Treasuries (SGOV)", min_value=0, max_value=100, value=10)

    st.header("Simulation")
    years = st.slider("Years to project", min_value=5, max_value=60, value=30)
    n_simulations = st.select_slider("Number of simulations", options=[250, 500, 1000, 2500, 5000], value=1000)
    scenario = st.selectbox(
        "Market scenario",
        ["Historical", "Recession", "Lost Decade", "Bull", "Bear", "High Volatility"],
    )
    volatility_multiplier = st.slider("High volatility multiplier", min_value=1.0, max_value=2.0, value=1.3, step=0.1)
    withdrawal_rate = st.slider("Withdrawal rate", min_value=0.02, max_value=0.08, value=0.04, step=0.005)

    st.header("Inflation")
    inflation_rate = st.slider("Annual inflation", min_value=0.0, max_value=0.05, value=0.025, step=0.005)
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

allocation = allocation_from_dict({"US": us_alloc / 100, "VXUS": vxus_alloc / 100, "SGOV": sgov_alloc / 100})

returns_df = get_monthly_returns(TICKERS, CACHE_DIR, EMBEDDED_PATH)
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
    initial_balances={"401k": balance_401k, "roth": balance_roth, "taxable": balance_taxable},
    monthly_contributions={
        "401k": contrib_401k,
        "roth": contrib_roth,
        "taxable": contrib_taxable,
        "employer_match_rate": match_rate,
        "employer_match_cap": match_cap,
    },
    allocation=allocation,
    years=years,
    n_simulations=n_simulations,
    scenario=scenario,
    asset_returns=asset_returns,
    retirement_months=retirement_months,
    soft_retirement_months=soft_retirement_months,
    soft_contribution_factor=soft_contribution_factor,
    withdrawal_rate=withdrawal_rate,
)

months = np.arange(0, years * 12 + 1)
display_mask = months <= display_months
ages = current_age + months / 12
display_age_axis = ages[display_mask]

if adjust_for_inflation and inflation_rate > 0:
    monthly_inflation = annual_to_monthly_rate(inflation_rate)
    inflation_factors = (1 + monthly_inflation) ** months
else:
    inflation_factors = np.ones_like(months, dtype=float)

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
histogram.update_layout(title="Ending Balance Distribution", xaxis_title="Balance", yaxis_title="Count")

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

total_paths = result.account_paths.sum(axis=2) / inflation_factors[None, :]
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

summary_cols = st.columns(4)
summary_cols[0].metric("Success rate", f"{result.success_rate * 100:.1f}%")
summary_cols[1].metric("Median ending balance", format_currency(np.median(ending_balances)))
summary_cols[2].metric("5th percentile ending", format_currency(np.percentile(ending_balances, 5)))
if retirement_months < len(percentile_series["p50"]):
    median_at_retirement = percentile_series["p50"][retirement_months]
    summary_cols[3].metric(
        "Median at retirement",
        format_currency(median_at_retirement),
        help="Median projected balance at the retirement month.",
    )
else:
    summary_cols[3].metric("Median at retirement", "N/A")

tabs = st.tabs(["Overview", "Distribution", "Accounts", "Assumptions"])
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
    allocation_chart.update_layout(title="Asset Allocation", legend_title="Assets")
    st.plotly_chart(allocation_chart, use_container_width=True)

with tabs[1]:
    st.plotly_chart(histogram, use_container_width=True)

with tabs[2]:
    st.plotly_chart(account_chart, use_container_width=True)

with tabs[3]:
    st.subheader("Scenario settings")
    st.write(
        {
            "Years": years,
            "Simulations": n_simulations,
            "Scenario": scenario,
            "Withdrawal rate": f"{withdrawal_rate:.2%}",
            "Inflation": f"{inflation_rate:.2%}" if adjust_for_inflation else "Not applied",
            "Soft retirement age": soft_retirement_age,
        }
    )
    st.subheader("Contribution summary")
    st.write(
        {
            "401k": format_currency(contrib_401k),
            "Roth IRA": format_currency(contrib_roth),
            "After-tax": format_currency(contrib_taxable),
            "Post-soft retirement factor": f"{soft_contribution_factor:.0%}",
            "Employer match": f"{match_rate:.0%} up to {format_currency(match_cap)}",
        }
    )
