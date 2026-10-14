"""Canonical schema, column aliases, category taxonomy and shared defaults.

Design notes
------------
* Canonical ``amount`` is **signed**: negative means money leaving the account
  (a debit / expense), positive means money arriving. Bank exports disagree on
  this convention, so :mod:`expense_ml.ingest` normalises into it.
* Column matching is alias-based and tolerant of the messy headers real banks
  emit ("Booking date", "Dr/Cr", "Memo", ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# --- canonical output columns ------------------------------------------------
TXN_ID = "transaction_id"
DATE = "date"
DESC = "description"
AMOUNT = "amount"
CATEGORY = "category"
ACCOUNT = "account"
CURRENCY = "currency"
DIRECTION = "direction"

CANONICAL_COLUMNS: tuple[str, ...] = (
    TXN_ID, DATE, DESC, AMOUNT, CATEGORY, ACCOUNT, CURRENCY,
)

# --- category taxonomy -------------------------------------------------------
# "Income"/"transfer" style categories are included on purpose: a real export
# contains them, and a model that can only ever say "Groceries" is useless.
CATEGORIES: tuple[str, ...] = (
    "Groceries",
    "Dining",
    "Transport",
    "Housing & Rent",
    "Utilities",
    "Shopping",
    "Entertainment",
    "Subscriptions",
    "Health & Fitness",
    "Travel",
    "Education",
    "Insurance",
    "Salary & Income",
    "Transfers",
    "Fees & Interest",
    "Personal Care",
    "Gifts & Donations",
)

#: Categories that are money *arriving* rather than discretionary spending.
INCOME_CATEGORIES: frozenset[str] = frozenset({"Salary & Income"})
NON_SPEND_CATEGORIES: frozenset[str] = frozenset(
    {"Salary & Income", "Transfers", "Fees & Interest"}
)

UNCATEGORIZED = "Uncategorized"

# --- column aliases ----------------------------------------------------------
# Keys are canonical names, values are normalised aliases (lowercased,
# punctuation stripped) checked against the incoming header.
COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    DATE: (
        "date", "transactiondate", "postingdate", "postdate", "bookdate",
        "bookingdate", "valuedate", "settlementdate", "datetime", "timestamp",
        "transdate", "createdat", "completeddate",
    ),
    DESC: (
        "description", "memo", "memonote", "narrative", "payee", "details",
        "reference", "transactiondescription", "merchant", "name", "concept",
        "remarks", "particulars", "info", "transactiondetails",
    ),
    AMOUNT: (
        "amount", "value", "transactionamount", "sum", "total", "amt",
        "transactionvalue",
    ),
    "debit": (
        "debit", "debitamount", "moneyout", "withdrawal", "withdrawals",
        "paidout", "charges", "out",
    ),
    "credit": (
        "credit", "creditamount", "moneyin", "deposit", "deposits",
        "paidin", "credits", "in",
    ),
    ACCOUNT: (
        "account", "accountname", "accountnumber", "accountid", "bankaccount",
        "card", "cardnumber",
    ),
    CURRENCY: ("currency", "currencycode", "ccy", "curr"),
    CATEGORY: (
        "category", "merchantcategory", "mcc", "spendcategory", "budgetcategory",
    ),
    # A "type"/"direction" column tells us whether an unsigned amount is money
    # in or out -- very common in card exports.
    DIRECTION: (
        "type", "direction", "transactiontype", "drcr", "debitcredit",
        "transactiondirection", "crdr", "indicator",
    ),
}

_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d-%m-%Y",
    "%Y%m%d",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
)

_DATETIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%.f",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
)


def normalise_header(name: object) -> str:
    """Normalise a raw column header for alias matching."""
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def resolve_columns(columns: Sequence[str]) -> dict[str, str]:
    """Map canonical field names onto the real column names of a file.

    The first matching alias wins, so callers get deterministic behaviour when
    a file has both ``date`` and ``transaction_date``.
    """
    normalised = {c: normalise_header(c) for c in columns}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            for original, norm in normalised.items():
                if norm == alias:
                    resolved[canonical] = original
                    break
            if canonical in resolved:
                break
    return resolved


@dataclass(slots=True)
class IngestReport:
    """What happened while reading a file -- surfaced in the UI, never hidden."""

    rows_in: int = 0
    rows_kept: int = 0
    dropped_bad_date: int = 0
    dropped_bad_amount: int = 0
    resolved_columns: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    dropped_examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_kept

    @property
    def drop_rate(self) -> float:
        return self.rows_dropped / self.rows_in if self.rows_in else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_kept": self.rows_kept,
            "rows_dropped": self.rows_dropped,
            "drop_rate": self.drop_rate,
            "dropped_bad_date": self.dropped_bad_date,
            "dropped_bad_amount": self.dropped_bad_amount,
            "resolved_columns": self.resolved_columns,
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class FeatureConfig:
    """Tunable knobs for feature engineering, kept together for reproducibility."""

    burst_window_hours: int = 2
    short_window_days: int = 7
    medium_window_days: int = 30
    amount_clip_percentile: float = 0.999


DEFAULT_FEATURE_CONFIG = FeatureConfig()
