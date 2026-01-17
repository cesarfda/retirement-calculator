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

with st.sidebar:
    st.header("Profile")
    current_age = st.number_input("Current age", min_value=18, max_value=80, value=35)
    retirement_age = st.number_input("Retirement age", min_value=current_age + 1, max_value=90, value=65)

    st.header("Current balances")
    balance_401k = st.number_input("401k balance", value=150_000.0, step=5_000.0)
    balance_roth = st.number_input("Roth IRA balance", value=40_000.0, step=2_500.0)
    balance_taxable = st.number_input("After-tax balance", value=25_000.0, step=2_500.0)

    st.header("Monthly contributions")
    contrib_401k = st.number_input("401k contribution", value=900.0, step=50.0)
    contrib_roth = st.number_input("Roth IRA contribution", value=400.0, step=25.0)
    contrib_taxable = st.number_input("After-tax contribution", value=300.0, step=25.0)

    st.header("Employer match")
    match_rate = st.slider("Match rate", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    match_cap = st.number_input("Match cap (monthly)", value=500.0, step=25.0)

    st.header("Asset allocation")
    us_alloc = st.slider("US (VTI)", min_value=0, max_value=100, value=60)
    vxus_alloc = st.slider("International (VXUS)", min_value=0, max_value=100, value=30)
    sgov_alloc = st.slider("Treasuries (SGOV)", min_value=0, max_value=100, value=10)

    st.header("Simulation")
    years = st.slider("Years to project", min_value=5, max_value=60, value=30)
    n_simulations = st.select_slider("Number of simulations", options=[250, 500, 1000, 2500, 5000], value=1000)
    scenario = st.selectbox(
        "Market scenario",
        ["Historical", "Bull", "Bear", "High Volatility"],
    )
    volatility_multiplier = st.slider("High volatility multiplier", min_value=1.0, max_value=2.0, value=1.3, step=0.1)
    withdrawal_rate = st.slider("Withdrawal rate", min_value=0.02, max_value=0.08, value=0.04, step=0.005)

    st.header("Inflation")
    inflation_rate = st.slider("Annual inflation", min_value=0.0, max_value=0.05, value=0.025, step=0.005)
    adjust_for_inflation = st.toggle("Show inflation-adjusted results", value=True)

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
    withdrawal_rate=withdrawal_rate,
)

months = np.arange(0, years * 12 + 1)

if adjust_for_inflation and inflation_rate > 0:
    monthly_inflation = annual_to_monthly_rate(inflation_rate)
    inflation_factors = (1 + monthly_inflation) ** months
else:
    inflation_factors = np.ones_like(months, dtype=float)

fan_chart = go.Figure()
fan_chart.add_traces(
    [
        go.Scatter(
            x=months,
            y=np.array(result.percentiles.p95) / inflation_factors,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(
            x=months,
            y=np.array(result.percentiles.p75) / inflation_factors,
            fill="tonexty",
            fillcolor="rgba(0, 123, 255, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="75-95%",
        ),
        go.Scatter(
            x=months,
            y=np.array(result.percentiles.p25) / inflation_factors,
            fill="tonexty",
            fillcolor="rgba(0, 123, 255, 0.25)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25-75%",
        ),
        go.Scatter(
            x=months,
            y=np.array(result.percentiles.p5) / inflation_factors,
            fill="tonexty",
            fillcolor="rgba(0, 123, 255, 0.35)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5-25%",
        ),
        go.Scatter(
            x=months,
            y=np.array(result.percentiles.p50) / inflation_factors,
            line=dict(color="#0d6efd", width=3),
            name="Median",
        ),
    ]
)
fan_chart.update_layout(
    title="Portfolio Value Percentiles",
    xaxis_title="Months",
    yaxis_title="Balance",
    hovermode="x unified",
)

ending_balances = result.ending_balances / inflation_factors[-1]

histogram = go.Figure(
    data=[go.Histogram(x=ending_balances, nbinsx=30, marker_color="#0d6efd")]
)
histogram.update_layout(title="Ending Balance Distribution", xaxis_title="Balance", yaxis_title="Count")

account_paths = result.account_paths.mean(axis=0) / inflation_factors[:, None]
account_chart = go.Figure(
    data=[
        go.Scatter(
            x=months,
            y=account_paths[:, 0],
            stackgroup="one",
            name="401k",
        ),
        go.Scatter(
            x=months,
            y=account_paths[:, 1],
            stackgroup="one",
            name="Roth IRA",
        ),
        go.Scatter(
            x=months,
            y=account_paths[:, 2],
            stackgroup="one",
            name="After-tax",
        ),
    ]
)
account_chart.update_layout(title="Account Growth (Average)", xaxis_title="Months", yaxis_title="Balance")

summary_cols = st.columns(3)
summary_cols[0].metric("Success rate", f"{result.success_rate * 100:.1f}%")
summary_cols[1].metric("Median ending balance", format_currency(np.median(ending_balances)))
summary_cols[2].metric("5th percentile ending", format_currency(np.percentile(ending_balances, 5)))

st.plotly_chart(fan_chart, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(histogram, use_container_width=True)
with col2:
    st.plotly_chart(account_chart, use_container_width=True)
