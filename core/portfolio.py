from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class Allocation:
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


def allocation_from_dict(allocation: Dict[str, float]) -> Allocation:
    return Allocation(
        us=allocation.get("US", 0.0),
        vxus=allocation.get("VXUS", 0.0),
        sgov=allocation.get("SGOV", 0.0),
    ).normalized()


def portfolio_returns(asset_returns: np.ndarray, allocation: Allocation) -> np.ndarray:
    weights = allocation.as_array()
    return (asset_returns * weights).sum(axis=2)
