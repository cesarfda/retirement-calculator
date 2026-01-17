from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AccountContributions:
    monthly_401k: float
    monthly_roth: float
    monthly_taxable: float
    employer_match_rate: float
    employer_match_cap: float


@dataclass
class AccountBalances:
    balance_401k: float
    balance_roth: float
    balance_taxable: float


def apply_employer_match(contribution: float, match_rate: float, match_cap: float) -> float:
    eligible = min(contribution, match_cap)
    return eligible * match_rate


def total_balance(balances: AccountBalances) -> float:
    return balances.balance_401k + balances.balance_roth + balances.balance_taxable
