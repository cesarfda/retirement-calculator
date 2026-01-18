"""Custom exceptions for the retirement calculator."""

from __future__ import annotations


class RetirementCalculatorError(Exception):
    """Base exception for retirement calculator errors."""

    pass


class ValidationError(RetirementCalculatorError):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class DataFetchError(RetirementCalculatorError):
    """Raised when fetching market data fails."""

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        self.message = message
        super().__init__(f"Failed to fetch data from {source}: {message}")


class SimulationError(RetirementCalculatorError):
    """Raised when simulation encounters numerical or logical issues."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
