"""Historical returns data fetching and caching."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from core.exceptions import DataFetchError

# Configure module logger
logger = logging.getLogger(__name__)


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
    """Fetch historical returns from yfinance."""
    ticker_list = list(tickers)
    logger.info(f"Fetching returns from yfinance for: {ticker_list}")

    try:
        price_data = yf.download(
            tickers=ticker_list,
            auto_adjust=True,
            progress=False,
            interval="1mo",
            period="max",
        )["Close"]
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        raise DataFetchError("yfinance", str(e)) from e

    price_data = price_data.dropna(how="all")
    returns = np.log(price_data / price_data.shift(1)).dropna()

    logger.info(f"Fetched {len(returns)} months of data from yfinance")
    return returns


def get_monthly_returns(
    tickers: Iterable[str],
    cache_dir: Path,
    embedded_path: Path,
    max_age_days: int = 30,
) -> pd.DataFrame:
    """
    Get monthly returns data, using cache or fetching fresh data.

    Args:
        tickers: List of ticker symbols to fetch
        cache_dir: Directory for caching data
        embedded_path: Path to embedded fallback data
        max_age_days: Maximum cache age in days

    Returns:
        DataFrame with monthly returns for each ticker
    """
    ticker_list = list(tickers)
    cache_path = cache_dir / "returns.json"

    # Try to use cached data
    if _cache_is_fresh(cache_path, max_age_days):
        logger.info(f"Using cached returns from {cache_path}")
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return load_embedded_returns_from_payload(payload)
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    # Try to fetch fresh data
    try:
        returns = _fetch_yfinance_returns(ticker_list)
        if returns.empty:
            logger.warning("yfinance returned empty data, using embedded fallback")
            return load_embedded_returns(embedded_path)
    except Exception as e:
        logger.warning(f"yfinance fetch failed: {e}, using embedded fallback")
        return load_embedded_returns(embedded_path)

    # Save to cache
    payload = {}
    for ticker in returns.columns:
        payload[ticker] = {
            "dates": [d.strftime("%Y-%m-%d") for d in returns.index],
            "returns": returns[ticker].fillna(0).tolist(),
        }

    try:
        _save_cache(payload, cache_path)
        logger.info(f"Saved {len(returns)} months to cache at {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return returns


def refresh_cache(
    tickers: Iterable[str],
    cache_dir: Path,
) -> bool:
    """
    Force refresh the returns cache.

    Args:
        tickers: List of ticker symbols to fetch
        cache_dir: Directory for caching data

    Returns:
        True if refresh succeeded, False otherwise
    """
    logger.info("Force refreshing returns cache")
    cache_path = cache_dir / "returns.json"

    try:
        returns = _fetch_yfinance_returns(tickers)
        if returns.empty:
            logger.error("yfinance returned empty data during refresh")
            return False

        payload = {}
        for ticker in returns.columns:
            payload[ticker] = {
                "dates": [d.strftime("%Y-%m-%d") for d in returns.index],
                "returns": returns[ticker].fillna(0).tolist(),
            }

        _save_cache(payload, cache_path)
        logger.info(f"Cache refreshed successfully with {len(returns)} months")
        return True

    except Exception as e:
        logger.error(f"Cache refresh failed: {e}")
        return False


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
    block_size: int = 12,
    scenario: str | None = None,
    volatility_multiplier: float = 1.0,
) -> np.ndarray:
    """
    Sample returns using block bootstrap for Monte Carlo simulation.

    Always samples from the FULL historical data to capture the complete
    distribution of market conditions (crashes, bull markets, recoveries).
    This naturally includes tail events like 2008 GFC and dot-com crash.

    For stress testing specific scenarios, use the stress_tests module to
    apply overlays to the sampled returns.

    Args:
        returns: Historical returns DataFrame
        n_months: Number of months to simulate
        n_simulations: Number of simulation paths
        block_size: Size of blocks for bootstrap (preserves autocorrelation)
        scenario: DEPRECATED - ignored, kept for backward compatibility
        volatility_multiplier: DEPRECATED - use stress tests instead

    Returns:
        3D array of shape (n_simulations, n_months, n_assets)
    """
    logger.debug(
        f"Sampling returns: {n_simulations} sims, {n_months} months, block_size={block_size}"
    )

    if scenario is not None and scenario.lower() != "historical":
        logger.warning(
            f"Scenario '{scenario}' is deprecated. Using full historical data. "
            "Use stress tests for scenario analysis."
        )

    # Always use full historical data
    values = returns.values
    n_available = values.shape[0]

    if n_available == 0:
        n_assets = values.shape[1] if values.ndim > 1 else 0
        return np.zeros((n_simulations, n_months, n_assets))

    # Block bootstrap from full history
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

    return sampled


def get_historical_summary(returns: pd.DataFrame) -> dict:
    """
    Get summary statistics about the historical data being used.

    Returns:
        Dict with date range, notable periods, and statistics
    """
    if returns.empty:
        return {"error": "No data available"}

    start_date = returns.index.min()
    end_date = returns.index.max()
    n_months = len(returns)

    # Calculate annualized returns and volatility per ticker
    ticker_stats = {}
    for ticker in returns.columns:
        mean_monthly = returns[ticker].mean()
        std_monthly = returns[ticker].std()
        # Annualize using compound returns
        annualized_return = (1 + mean_monthly) ** 12 - 1
        annualized_vol = std_monthly * np.sqrt(12)
        ticker_stats[ticker] = {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "mean_monthly": mean_monthly,
        }

    # Overall portfolio stats (equal weighted for summary)
    mean_monthly = returns.mean().mean()
    std_monthly = returns.std().mean()
    annualized_return = (1 + mean_monthly) ** 12 - 1
    annualized_vol = std_monthly * np.sqrt(12)

    # Identify worst drawdown periods
    total_return = returns.sum(axis=1)
    cumulative = total_return.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    worst_drawdown = drawdown.min()
    worst_drawdown_date = drawdown.idxmin()

    return {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "n_months": n_months,
        "n_years": n_months / 12,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "ticker_stats": ticker_stats,
        "worst_drawdown": worst_drawdown,
        "worst_drawdown_date": worst_drawdown_date.strftime("%Y-%m-%d") if hasattr(worst_drawdown_date, 'strftime') else str(worst_drawdown_date),
        "includes_gfc": start_date.year <= 2007 and end_date.year >= 2009,
        "includes_dotcom": start_date.year <= 2000 and end_date.year >= 2002,
        "includes_covid": end_date.year >= 2020,
    }
