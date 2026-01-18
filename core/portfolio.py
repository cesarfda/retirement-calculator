from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Allocation:
    """
    Basic 3-asset allocation (US stocks, International stocks, Treasuries).

    This is the default allocation class for backward compatibility.
    For more asset classes, use ExtendedAllocation.
    """

    us: float
    vxus: float
    sgov: float

    def as_array(self) -> np.ndarray:
        return np.array([self.us, self.vxus, self.sgov])

    def normalized(self) -> "Allocation":
        total = self.us + self.vxus + self.sgov
        if total == 0:
            return Allocation(0.0, 0.0, 0.0)
        return Allocation(self.us / total, self.vxus / total, self.sgov / total)


@dataclass
class ExtendedAllocation:
    """
    Extended allocation supporting additional asset classes.

    Supports 6 asset classes:
    - US Total Market (VTI/SPY)
    - International Developed (VXUS/EFA)
    - Short-term Treasuries (SGOV/SHY)
    - REITs (VNQ)
    - TIPS (VTIP)
    - Small Cap Value (VBR)

    All weights should sum to 1.0 (or will be normalized).
    """

    us: float = 0.0            # US Total Market
    vxus: float = 0.0          # International Developed
    sgov: float = 0.0          # Short-term Treasuries
    reits: float = 0.0         # Real Estate (VNQ)
    tips: float = 0.0          # Inflation-Protected (VTIP)
    small_cap_value: float = 0.0  # Small Cap Value (VBR)

    def as_array(self) -> np.ndarray:
        """Return weights as array in order: US, VXUS, SGOV, REITs, TIPS, SCV."""
        return np.array([
            self.us,
            self.vxus,
            self.sgov,
            self.reits,
            self.tips,
            self.small_cap_value,
        ])

    def total(self) -> float:
        """Sum of all allocations."""
        return (
            self.us
            + self.vxus
            + self.sgov
            + self.reits
            + self.tips
            + self.small_cap_value
        )

    def normalized(self) -> "ExtendedAllocation":
        """Return normalized allocation (weights sum to 1)."""
        total = self.total()
        if total == 0:
            return ExtendedAllocation()
        return ExtendedAllocation(
            us=self.us / total,
            vxus=self.vxus / total,
            sgov=self.sgov / total,
            reits=self.reits / total,
            tips=self.tips / total,
            small_cap_value=self.small_cap_value / total,
        )

    def to_basic(self) -> Allocation:
        """
        Convert to basic 3-asset Allocation.

        Maps extended assets to basic categories:
        - REITs and Small Cap Value -> US
        - TIPS -> SGOV (bonds)
        """
        return Allocation(
            us=self.us + self.reits + self.small_cap_value,
            vxus=self.vxus,
            sgov=self.sgov + self.tips,
        ).normalized()

    @property
    def equity_ratio(self) -> float:
        """Percentage allocated to equities (US + VXUS + REITs + SCV)."""
        total = self.total()
        if total == 0:
            return 0.0
        equity = self.us + self.vxus + self.reits + self.small_cap_value
        return equity / total

    @property
    def bond_ratio(self) -> float:
        """Percentage allocated to bonds (SGOV + TIPS)."""
        total = self.total()
        if total == 0:
            return 0.0
        bonds = self.sgov + self.tips
        return bonds / total


# Asset class display names
ASSET_CLASS_NAMES: Dict[str, str] = {
    "us": "US Total Market (VTI)",
    "vxus": "International (VXUS)",
    "sgov": "Treasuries (SGOV)",
    "reits": "REITs (VNQ)",
    "tips": "TIPS (VTIP)",
    "small_cap_value": "Small Cap Value (VBR)",
}


def allocation_from_dict(allocation: Dict[str, float]) -> Allocation:
    """Create basic Allocation from dictionary."""
    return Allocation(
        us=allocation.get("US", 0.0),
        vxus=allocation.get("VXUS", 0.0),
        sgov=allocation.get("SGOV", 0.0),
    ).normalized()


def extended_allocation_from_dict(allocation: Dict[str, float]) -> ExtendedAllocation:
    """Create ExtendedAllocation from dictionary."""
    return ExtendedAllocation(
        us=allocation.get("US", allocation.get("us", 0.0)),
        vxus=allocation.get("VXUS", allocation.get("vxus", 0.0)),
        sgov=allocation.get("SGOV", allocation.get("sgov", 0.0)),
        reits=allocation.get("REITS", allocation.get("reits", 0.0)),
        tips=allocation.get("TIPS", allocation.get("tips", 0.0)),
        small_cap_value=allocation.get("SCV", allocation.get("small_cap_value", 0.0)),
    ).normalized()


def portfolio_returns(asset_returns: np.ndarray, allocation: Allocation) -> np.ndarray:
    """
    Calculate portfolio returns from asset returns and allocation.

    Args:
        asset_returns: Array of shape (n_sims, n_months, n_assets)
        allocation: Allocation weights (basic 3-asset)

    Returns:
        Array of shape (n_sims, n_months) with portfolio returns
    """
    weights = allocation.as_array()
    return (asset_returns * weights).sum(axis=2)


def extended_portfolio_returns(
    asset_returns: np.ndarray,
    allocation: ExtendedAllocation,
) -> np.ndarray:
    """
    Calculate portfolio returns with extended allocation.

    Args:
        asset_returns: Array of shape (n_sims, n_months, 6) with returns for
                      [US, VXUS, SGOV, REITs, TIPS, SCV]
        allocation: ExtendedAllocation weights

    Returns:
        Array of shape (n_sims, n_months) with portfolio returns
    """
    weights = allocation.as_array()

    # Handle case where asset_returns has fewer columns than weights
    n_assets = asset_returns.shape[2]
    if n_assets < len(weights):
        # Truncate weights to match available assets
        weights = weights[:n_assets]

    return (asset_returns[:, :, :len(weights)] * weights).sum(axis=2)
