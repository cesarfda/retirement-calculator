from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Percentiles:
    p5: list[float]
    p25: list[float]
    p50: list[float]
    p75: list[float]
    p95: list[float]


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def annual_to_monthly_rate(annual_rate: float) -> float:
    return (1 + annual_rate) ** (1 / 12) - 1


def weighted_sum(values: Iterable[float], weights: Iterable[float]) -> float:
    return sum(v * w for v, w in zip(values, weights))
