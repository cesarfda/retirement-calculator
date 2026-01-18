# Retirement Calculator

A comprehensive retirement planning tool with Monte Carlo simulation, tax modeling, and advanced withdrawal strategies.

## Features

### Monte Carlo Simulation
- **Block Bootstrap Sampling**: Preserves autocorrelation in market returns by sampling 12-month blocks
- **Multiple Scenarios**: Historical, Recession (2007-2009), Lost Decade (2000-2009), Bull, Bear, High Volatility
- **Configurable Simulations**: 250 to 5,000 simulation paths

### Account Types
- **401(k)**: Pre-tax contributions with employer matching
- **Roth IRA**: After-tax contributions, tax-free growth and withdrawals
- **Taxable Brokerage**: After-tax with capital gains treatment

### Tax Modeling
- **IRS Contribution Limits**: Enforces annual 401(k) and Roth IRA limits with catch-up contributions for age 50+
- **Required Minimum Distributions (RMDs)**: Mandatory 401(k) withdrawals starting at age 73
- **Tax-Efficient Withdrawal Order**: Optimizes withdrawal sequence (taxable → 401k → Roth)
- **Federal Tax Brackets**: 2024 brackets for single and married filing jointly

### Risk Management
- **Glide Path / Bond Tent**: Dynamic allocation that reduces equity near retirement, then increases post-retirement
- **Guardrails Strategy**: Adjusts spending based on portfolio performance (ceiling/floor)
- **Sequence of Returns Risk (SORR)**: Withdrawals applied before returns in retirement phase
- **Expense Ratio Modeling**: Accounts for fund fees

### Risk Metrics
- Success rate (probability of not running out of money)
- Maximum drawdown (median and worst-case)
- Safe withdrawal rate estimation
- Years of income remaining
- Probability of ruin by year

## Quickstart

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Configuration

### Profile Settings
| Setting | Description | Range |
|---------|-------------|-------|
| Current age | Your current age | 18-80 |
| Retirement age | When you stop working | > current age |
| Soft retirement age | When you reduce contributions | current age to retirement |

### Account Balances
Enter current balances for each account type:
- **401k balance**: Traditional pre-tax retirement account
- **Roth IRA balance**: After-tax retirement account
- **After-tax balance**: Taxable brokerage account

### Contributions
- **Monthly amounts**: How much you contribute each month
- **Soft retirement factor**: Contribution reduction after soft retirement (0-100%)
- **Per-account toggles**: Continue or stop each account's contributions

### Employer Match
- **Match rate**: Percentage of your contribution matched (0-100%)
- **Match cap**: Maximum monthly employer match

### Asset Allocation
Choose allocation or use glide path:
- **Static allocation**: Fixed percentages for US, International, and Treasuries
- **Glide path**: Dynamic allocation that changes over time
  - Default: 90% → 50% → 60% equity
  - Aggressive: Higher equity throughout
  - Conservative: Lower equity throughout
  - Custom: Set your own percentages

### Tax Modeling (Optional)
- **Filing status**: Single or Married Filing Jointly
- **Cost basis ratio**: Percentage of taxable account that is original investment
- **Contribution limits**: Enforce IRS annual limits
- **RMD enforcement**: Require minimum distributions at age 73+
- **Tax-efficient withdrawal**: Optimize withdrawal order

### Guardrails (Optional)
- **Ceiling**: Maximum spending increase when portfolio is up (e.g., 110%)
- **Floor**: Maximum spending decrease when portfolio is down (e.g., 95%)
- **Thresholds**: Portfolio ratios that trigger ceiling/floor

## Data Sources

Historical returns are fetched from Yahoo Finance for:
- **VTI**: Vanguard Total Stock Market ETF (US)
- **VXUS**: Vanguard Total International Stock ETF
- **SGOV**: iShares 0-3 Month Treasury Bond ETF

Data is cached locally and refreshed every 30 days. If fetching fails, embedded historical data is used as fallback.

## Project Structure

```
retirement-calculator/
├── app.py                 # Streamlit web application
├── core/
│   ├── accounts.py        # Account types and employer match
│   ├── data_validation.py # Data quality checks
│   ├── exceptions.py      # Custom exceptions
│   ├── glide_path.py      # Bond tent / glide path logic
│   ├── portfolio.py       # Asset allocation
│   ├── returns.py         # Historical data fetching
│   ├── risk_metrics.py    # Risk calculations
│   ├── simulator.py       # Monte Carlo engine
│   ├── tax_config.py      # IRS limits and RMD tables
│   ├── taxes.py           # Tax bracket calculations
│   └── validation.py      # Input validation
├── data/
│   ├── cache/             # Cached market data
│   └── historical_returns.json  # Embedded fallback data
├── docs/
│   └── ASSUMPTIONS.md     # Model assumptions
├── tests/                 # Pytest test suite
├── utils/
│   └── helpers.py         # Utility functions
└── requirements.txt
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

## Limitations

See [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) for detailed model assumptions and limitations.

Key limitations:
- Federal taxes only (no state taxes)
- No Social Security income modeling
- No healthcare cost modeling
- Simplified RMD calculation (uses current balance, not prior year-end)
- No Roth conversion optimization

## License

MIT License
