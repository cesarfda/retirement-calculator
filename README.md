# Retirement Calculator

Streamlit-based retirement calculator with Monte Carlo simulation and historical bootstrap scenarios.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data

Historical monthly returns are cached under `data/cache/`. If the cache is unavailable or stale,
the simulator falls back to the embedded dataset in `data/historical_returns.json`.
