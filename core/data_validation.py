"""Data quality validation for retirement calculator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class DataValidationResult:
    """Result of data validation containing any issues found."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add a critical error."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a non-critical warning."""
        self.warnings.append(message)

    def is_valid(self) -> bool:
        """Return True if no critical errors."""
        return len(self.errors) == 0

    def has_warnings(self) -> bool:
        """Return True if there are warnings."""
        return len(self.warnings) > 0

    def all_messages(self) -> list[str]:
        """Return all errors and warnings."""
        return self.errors + self.warnings


def validate_returns_data(
    df: pd.DataFrame,
    expected_tickers: list[str],
    min_months: int = 120,
    max_monthly_return: float = 0.5,
) -> DataValidationResult:
    """
    Validate historical returns data quality.

    Args:
        df: DataFrame with returns data (columns = tickers, rows = months)
        expected_tickers: List of ticker symbols expected in data
        min_months: Minimum months of history required (default 10 years)
        max_monthly_return: Maximum reasonable monthly return (default 50%)

    Returns:
        DataValidationResult with errors and warnings
    """
    result = DataValidationResult()

    if df is None or df.empty:
        result.add_error("Returns data is empty or None")
        return result

    # Check for missing tickers
    available_tickers = set(df.columns)
    expected_set = set(expected_tickers)
    missing = expected_set - available_tickers

    if missing:
        result.add_error(f"Missing tickers: {missing}")

    # Check for sufficient history
    n_months = len(df)
    if n_months < min_months:
        result.add_warning(
            f"Limited history: {n_months} months (recommended: {min_months}+)"
        )

    if n_months < 24:
        result.add_error(f"Insufficient history: {n_months} months (need at least 24)")

    # Check for extreme values (likely data errors)
    for ticker in expected_tickers:
        if ticker not in df.columns:
            continue

        ticker_returns = df[ticker]

        # Check for extreme positive returns
        extreme_high = ticker_returns > max_monthly_return
        if extreme_high.any():
            n_extreme = extreme_high.sum()
            max_val = ticker_returns.max()
            result.add_warning(
                f"{ticker}: {n_extreme} extreme positive returns (max: {max_val:.1%})"
            )

        # Check for extreme negative returns
        extreme_low = ticker_returns < -max_monthly_return
        if extreme_low.any():
            n_extreme = extreme_low.sum()
            min_val = ticker_returns.min()
            result.add_warning(
                f"{ticker}: {n_extreme} extreme negative returns (min: {min_val:.1%})"
            )

        # Check for missing/NaN values
        n_missing = ticker_returns.isna().sum()
        if n_missing > 0:
            result.add_warning(f"{ticker}: {n_missing} missing values")

        # Check for suspiciously constant values
        if ticker_returns.std() < 0.001:
            result.add_warning(f"{ticker}: Very low volatility (std < 0.1%)")

    # Check date range
    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
        try:
            start_date = df.index.min()
            end_date = df.index.max()

            # Warn if data is stale (more than 60 days old)
            if hasattr(end_date, 'date'):
                days_old = (datetime.now().date() - end_date.date()).days
                if days_old > 60:
                    result.add_warning(
                        f"Data may be stale: last date is {end_date.date()} "
                        f"({days_old} days ago)"
                    )
        except Exception:
            pass  # Date parsing failed, skip this check

    return result


@dataclass
class CacheStatus:
    """Status of the data cache."""

    exists: bool
    age_days: float | None
    size_kb: float | None
    is_fresh: bool
    last_modified: datetime | None


def get_cache_status(
    cache_path: Path,
    max_age_days: int = 30,
) -> CacheStatus:
    """
    Check the status of the data cache.

    Args:
        cache_path: Path to the cache file
        max_age_days: Maximum age in days before cache is considered stale

    Returns:
        CacheStatus with cache information
    """
    if not cache_path.exists():
        return CacheStatus(
            exists=False,
            age_days=None,
            size_kb=None,
            is_fresh=False,
            last_modified=None,
        )

    stat = cache_path.stat()
    last_modified = datetime.fromtimestamp(stat.st_mtime)
    age = datetime.now() - last_modified
    age_days = age.total_seconds() / (24 * 60 * 60)
    size_kb = stat.st_size / 1024

    return CacheStatus(
        exists=True,
        age_days=age_days,
        size_kb=size_kb,
        is_fresh=age_days <= max_age_days,
        last_modified=last_modified,
    )


def validate_simulation_inputs(
    n_simulations: int,
    years: int,
    asset_returns_shape: tuple[int, ...],
) -> DataValidationResult:
    """
    Validate simulation input consistency.

    Args:
        n_simulations: Number of simulations requested
        years: Number of years to simulate
        asset_returns_shape: Shape of the asset returns array

    Returns:
        DataValidationResult with any issues
    """
    result = DataValidationResult()

    expected_months = years * 12

    if len(asset_returns_shape) != 3:
        result.add_error(
            f"Asset returns must be 3D (sims, months, assets), got shape {asset_returns_shape}"
        )
        return result

    actual_sims, actual_months, actual_assets = asset_returns_shape

    if actual_sims != n_simulations:
        result.add_error(
            f"Asset returns has {actual_sims} simulations, expected {n_simulations}"
        )

    if actual_months < expected_months:
        result.add_error(
            f"Asset returns has {actual_months} months, need {expected_months} for {years} years"
        )

    if actual_assets < 1:
        result.add_error("Asset returns must have at least 1 asset")

    return result
