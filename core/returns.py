from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


def load_embedded_returns(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    frames = []
    for ticker, data in payload.items():
        frame = pd.DataFrame({"date": data["dates"], ticker: data["returns"]})
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner")

    merged = merged.set_index("date").sort_index()
    return merged


def _cache_is_fresh(cache_path: Path, max_age_days: int) -> bool:
    if not cache_path.exists():
        return False
    max_age = datetime.utcnow() - timedelta(days=max_age_days)
    return datetime.utcfromtimestamp(cache_path.stat().st_mtime) >= max_age


def _save_cache(data: dict[str, dict[str, list[float]]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)


def _fetch_yfinance_returns(tickers: Iterable[str]) -> pd.DataFrame:
    price_data = yf.download(
        tickers=list(tickers),
        auto_adjust=True,
        progress=False,
        interval="1mo",
        period="max",
    )["Close"]
    price_data = price_data.dropna(how="all")
    returns = np.log(price_data / price_data.shift(1)).dropna()
    return returns


def get_monthly_returns(
    tickers: Iterable[str],
    cache_dir: Path,
    embedded_path: Path,
    max_age_days: int = 30,
) -> pd.DataFrame:
    cache_path = cache_dir / "returns.json"
    if _cache_is_fresh(cache_path, max_age_days):
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return load_embedded_returns_from_payload(payload)

    try:
        returns = _fetch_yfinance_returns(tickers)
    except Exception:
        return load_embedded_returns(embedded_path)
    if returns.empty:
        return load_embedded_returns(embedded_path)

    payload = {}
    for ticker in returns.columns:
        payload[ticker] = {
            "dates": [d.strftime("%Y-%m-%d") for d in returns.index],
            "returns": returns[ticker].fillna(0).tolist(),
        }

    _save_cache(payload, cache_path)
    return returns


def load_embedded_returns_from_payload(payload: dict[str, dict[str, list[float]]]) -> pd.DataFrame:
    frames = []
    for ticker, data in payload.items():
        frame = pd.DataFrame({"date": data["dates"], ticker: data["returns"]})
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="inner")

    merged = merged.set_index("date").sort_index()
    return merged


def sample_returns(
    returns: pd.DataFrame,
    n_months: int,
    n_simulations: int,
    scenario: str,
    block_size: int = 12,
    volatility_multiplier: float = 1.0,
) -> np.ndarray:
    scenario = scenario.lower()
    returns_matrix = returns.copy()

    if scenario in {"recession", "lost decade"}:
        returns_matrix = _filter_historical_scenario(returns_matrix, scenario)

    if scenario in {"bull", "bear"}:
        score = returns_matrix.mean(axis=1)
        median = score.median()
        if scenario == "bull":
            returns_matrix = returns_matrix.loc[score >= median]
        else:
            returns_matrix = returns_matrix.loc[score < median]

    values = returns_matrix.values
    n_available = values.shape[0]
    if n_available == 0:
        n_assets = values.shape[1] if values.ndim > 1 else 0
        return np.zeros((n_simulations, n_months, n_assets))

    effective_block_size = min(block_size, n_available)
    blocks_per_path = int(np.ceil(n_months / effective_block_size))

    indices = np.random.randint(
        0, n_available - effective_block_size + 1, size=(n_simulations, blocks_per_path)
    )
    sampled = np.zeros((n_simulations, blocks_per_path * effective_block_size, values.shape[1]))

    for sim in range(n_simulations):
        blocks = [values[start : start + effective_block_size] for start in indices[sim]]
        sampled[sim] = np.concatenate(blocks, axis=0)

    sampled = sampled[:, :n_months, :]

    if scenario == "high volatility":
        sampled = sampled * volatility_multiplier

    return sampled


def _filter_historical_scenario(returns: pd.DataFrame, scenario: str) -> pd.DataFrame:
    scenario = scenario.lower()
    ranges = {
        "recession": ("2007-01-01", "2009-12-31"),
        "lost decade": ("2000-01-01", "2009-12-31"),
    }
    if scenario not in ranges:
        return returns

    start, end = ranges[scenario]
    filtered = returns.loc[start:end]
    if filtered.shape[0] < 12:
        return returns
    return filtered
