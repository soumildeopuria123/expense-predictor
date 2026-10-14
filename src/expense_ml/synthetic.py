"""Synthetic bank-transaction generator.

Produces a realistic labelled dataset so the whole pipeline runs out of the box
and the anomaly detector can be scored against ground truth. Real bank data is
not reproducible or shareable, and an anomaly detector you cannot evaluate is
just a number generator.

Two kinds of label are attached:

* ``category``            -- the true spend category, for classifier metrics;
* ``is_injected_anomaly`` -- whether the row was deliberately corrupted, which
  gives us precision/recall for the unsupervised detector. This column is a
  *target only*; the feature builder must never see it.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import polars as pl

from expense_ml.config import ACCOUNT, AMOUNT, CATEGORY, CURRENCY, DATE, DESC, TXN_ID

ANOMALY_LABEL = "is_injected_anomaly"
ANOMALY_TYPE = "anomaly_type"


@dataclass(frozen=True, slots=True)
class CategorySpec:
    name: str
    merchants: tuple[str, ...]
    cities: tuple[str, ...]
    median_amount: float
    sigma: float = 0.45          # lognormal spread
    share: float = 0.05          # relative share of all rows
    frequency: str = "random"    # random | monthly | biweekly | weekend
    day_of_month: int | None = None
    hour_low: int = 8
    hour_high: int = 21
    inflow: bool = False


_CITIES = (
    "SAN FRANCISCO CA", "OAKLAND CA", "SEATTLE WA", "AUSTIN TX",
    "CHICAGO IL", "BROOKLYN NY", "DENVER CO", "PORTLAND OR",
    "BOSTON MA", "MIAMI FL", "PHOENIX AZ", "ATLANTA GA",
)

SPECS: tuple[CategorySpec, ...] = (
    CategorySpec("Groceries", ("WHOLE FOODS", "TRADER JOE'S", "SAFETY", "KROGER", "ALDI", "COSTCO", "SPROUTS FARMERS", "PCC COMMUNITY"), _CITIES, 68, 0.5, 0.11, "random", hour_low=8, hour_high=20),
    CategorySpec("Dining", ("CHIPOTLE", "STARBUCKS", "MCDONALD'S", "PANERA BREAD", "TACO BELL", "PIZZA HUT", "DOORDASH", "UBER EATS", "KFC", "SWEETGREEN"), _CITIES, 17, 0.6, 0.16, "random", hour_low=7, hour_high=23),
    CategorySpec("Transport", ("UBER TRIP", "LYFT RIDE", "SHELL OIL", "CHEVRON", "METRO CARD", "SP+ PARKING", "BAY AREA RAPID", "EXXONMOBIL"), _CITIES, 26, 0.7, 0.11, "random", hour_low=5, hour_high=23),
    CategorySpec("Housing & Rent", ("SUNSET APARTMENT MGMT", "RENT PAYMENT", "OAKWOOD RESIDENCES", "PROPERTY MGMT"), _CITIES, 1850, 0.12, 0.03, "monthly", day_of_month=1, hour_low=0, hour_high=23),
    CategorySpec("Utilities", ("PG&E", "COMCAST CABLE", "AT&T WIRELESS", "VERIZON FIOS", "SEATTLE CITY LIGHT", "WATER DISTRICT"), _CITIES, 124, 0.45, 0.05, "monthly", day_of_month=12, hour_low=0, hour_high=23),
    CategorySpec("Shopping", ("AMAZON", "TARGET", "WALMART", "BEST BUY", "IKEA", "COSTCO WHOLESALE", "ETSY", "NORDSTROM"), _CITIES, 64, 0.8, 0.12, "random"),
    CategorySpec("Entertainment", ("AMC THEATRES", "STEAM GAMES", "TICKMASTER", "BOWLERO", "LIVE NATION", "REGAL CINEMAS"), _CITIES, 32, 0.7, 0.05, "random", hour_low=10, hour_high=23),
    CategorySpec("Subscriptions", ("NETFLIX", "SPOTIFY", "ADOBE CC", "ICLOUD+", "DISNEY+", "GYM MEMBERSHIP", "NYTIMES.COM", "HULU"), _CITIES, 15, 0.4, 0.06, "monthly", day_of_month=15, hour_low=0, hour_high=23),
    CategorySpec("Health & Fitness", ("WALGREENS", "CVS PHARMACY", "24 HOUR FITNESS", "DR SMITH DENTAL", "QUEST DIAGNOSTICS", "CVS/PHARMACY"), _CITIES, 48, 0.7, 0.05, "random"),
    CategorySpec("Travel", ("DELTA AIR LINES", "MARRIOTT", "AIRBNB", "HERTZ RENTAL", "UNITED AIRLINES", "HILTON HOTELS", "AMTRAK"), _CITIES, 385, 0.7, 0.03, "random"),
    CategorySpec("Education", ("COURSERA", "UNIVERSITY TUITION", "UDEMY", "BARNES & NOBLE", "CHEGG", "SKILLSHARE"), _CITIES, 95, 0.8, 0.02, "random"),
    CategorySpec("Insurance", ("GEICO AUTO", "STATE FARM", "BLUE CROSS", "PROGRESSIVE"), _CITIES, 142, 0.3, 0.03, "monthly", day_of_month=8, hour_low=0, hour_high=23),
    CategorySpec("Salary & Income", ("ACME CORP PAYROLL", "ACME CORP DIRECT DEP", "ACME CORP SALARY"), _CITIES, 3200, 0.08, 0.05, "biweekly", inflow=True, hour_low=0, hour_high=23),
    CategorySpec("Transfers", ("TRANSFER TO SAVINGS", "ZELLE TRANSFER", "VENMO", "PAYPAL XFER", "TRANSFER FROM CHK"), _CITIES, 250, 0.9, 0.05, "random", hour_low=0, hour_high=23),
    CategorySpec("Fees & Interest", ("OVERDRAFT FEE", "ATM SURCHARGE", "WIRE FEE", "INTEREST PAID"), _CITIES, 28, 0.8, 0.02, "random", hour_low=0, hour_high=23),
    CategorySpec("Personal Care", ("SUPERCUTS", "ULTA BEAUTY", "SEPHORA", "NAIL SALON", "TIDE CLEANERS"), _CITIES, 42, 0.6, 0.03, "random"),
    CategorySpec("Gifts & Donations", ("GOODWILL", "RED CROSS", "BIRTHDAY GIFT", "SALVATION ARMY", "ST JUDE"), _CITIES, 55, 0.8, 0.02, "random"),
)

_BY_NAME = {s.name: s for s in SPECS}

_SUFFIXES = (
    "POS {n}", "CARD {n}", "AUTH {n}", "ORDER {n}", "REF {n}", "TP {n}",
    "ID {n}", "",
)


def _weighted_specs(rng: np.random.Generator) -> list[CategorySpec]:
    shares = np.array([s.share for s in SPECS], dtype=float)
    return list(rng.choice(np.array(SPECS, dtype=object), size=1, p=shares / shares.sum())[0]) \
        if False else list(SPECS)


def _pick_time(rng: np.random.Generator, spec: CategorySpec, start: dt.datetime, end: dt.datetime) -> dt.datetime:
    span_days = max((end - start).days, 1)
    if spec.frequency == "monthly":
        # spread across months, pinned to one day of the month
        n_months = max(span_days // 30, 1)
        month_off = int(rng.integers(0, n_months))
        base = start + dt.timedelta(days=30 * month_off + (spec.day_of_month or 1))
        if base > end:
            base = end - dt.timedelta(days=int(rng.integers(0, 28)))
    elif spec.frequency == "biweekly":
        n = max(span_days // 14, 1)
        base = start + dt.timedelta(days=14 * int(rng.integers(0, n)) + 1)
        if base > end:
            base = end - dt.timedelta(days=int(rng.integers(0, 13)))
    else:
        day = int(rng.integers(0, span_days))
        base = start + dt.timedelta(days=day)
        if spec.frequency == "weekend":
            while base.weekday() >= 5:
                base += dt.timedelta(days=1)
            if base > end:
                base = end
    hour = int(rng.integers(spec.hour_low, spec.hour_high + 1))
    return base.replace(hour=hour, minute=int(rng.integers(0, 60)), second=int(rng.integers(0, 60)))


def _describe(rng: np.random.Generator, spec: CategorySpec) -> str:
    merchant = str(rng.choice(np.array(spec.merchants, dtype=object)))
    city = str(rng.choice(np.array(_CITIES, dtype=object)))
    tail = str(rng.choice(np.array(_SUFFIXES, dtype=object)))
    tail = tail.format(n=int(rng.integers(1000, 99999)))
    parts = [merchant]
    # roughly a third of transactions omit the location, as in real exports
    if rng.random() < 0.34:
        parts.append(city)
    if tail:
        parts.append(tail)
    return " ".join(parts)


def make_transactions(
    n_rows: int = 12000,
    *,
    seed: int = 42,
    start: dt.date | None = None,
    end: dt.date | None = None,
    anomaly_rate: float = 0.018,
    accounts: Sequence[str] = ("Chase Checking", "Amex Blue Cash"),
) -> pl.DataFrame:
    """Generate a labelled transaction history.

    ``anomaly_rate`` controls how many rows are deliberately corrupted, across
    four realistic patterns (amount spike, odd-hour, velocity burst, unknown
    merchant). These are the signals a fraud model should plausibly catch.
    """
    rng = np.random.default_rng(seed)
    start_dt = dt.datetime.combine(start or dt.date(2024, 1, 1), dt.time(0, 0))
    end_dt = dt.datetime.combine(end or dt.date(2025, 9, 30), dt.time(23, 59, 59))

    shares = np.array([s.share for s in SPECS], dtype=float)
    probs = shares / shares.sum()
    picks = rng.choice(len(SPECS), size=n_rows, p=probs)

    records: list[dict[str, object]] = []
    for idx in picks:
        spec = SPECS[int(idx)]
        when = _pick_time(rng, spec, start_dt, end_dt)
        # lognormal centred on the spec median
        amount = float(np.exp(rng.normal(math.log(spec.median_amount), spec.sigma)))
        sign = 1.0 if spec.inflow else -1.0
        # transfers are genuinely mixed in sign
        if spec.name == "Transfers":
            sign = 1.0 if rng.random() < 0.45 else -1.0
        records.append(
            {
                DATE: when,
                DESC: _describe(rng, spec),
                AMOUNT: sign * amount,
                CATEGORY: spec.name,
                ACCOUNT: str(rng.choice(np.array(accounts, dtype=object))),
                CURRENCY: "USD",
                ANOMALY_LABEL: False,
                ANOMALY_TYPE: "",
            }
        )

    # --- inject anomalies ---------------------------------------------------
    n_bad = int(round(n_rows * anomaly_rate))
    if n_bad > 0 and records:
        idxs = rng.choice(len(records), size=n_bad, replace=False)
        kinds = rng.choice(
            ["amount_spike", "odd_hour", "unknown_merchant", "velocity_burst"],
            size=n_bad,
            p=[0.45, 0.2, 0.2, 0.15],
        )
        unknown_seed = 0
        for i, kind in zip(idxs, kinds):
            rec = records[int(i)]
            if kind == "amount_spike":
                # 8-25x the usual amount for that merchant's category
                mult = float(rng.uniform(8, 25))
                rec[AMOUNT] = float(np.sign(rec[AMOUNT])) * abs(float(rec[AMOUNT])) * mult
            elif kind == "odd_hour":
                t = rec[DATE]
                assert isinstance(t, dt.datetime)
                rec[DATE] = t.replace(hour=int(rng.integers(1, 5)), minute=int(rng.integers(0, 60)))
            elif kind == "unknown_merchant":
                unknown_seed += 1
                rec[DESC] = f"XYZONLINEPAY {unknown_seed} INTL AUTH {int(rng.integers(1000, 9999))}"
                rec[AMOUNT] = -abs(float(rng.uniform(400, 2600)))
            else:  # velocity_burst: several near-identical charges within minutes
                t = rec[DATE]
                assert isinstance(t, dt.datetime)
                base_amount = abs(float(rec[AMOUNT]))
                rec[DATE] = t.replace(minute=int(rng.integers(0, 20)))
                burst = []
                for k in range(1, int(rng.integers(3, 6))):
                    dup = dict(rec)
                    dup[DATE] = rec[DATE] + dt.timedelta(minutes=k * int(rng.integers(2, 9)))
                    dup[AMOUNT] = -base_amount * float(rng.uniform(0.95, 1.05))
                    dup[ANOMALY_LABEL] = True
                    dup[ANOMALY_TYPE] = "velocity_burst"
                    burst.append(dup)
                records.extend(burst)
            rec[ANOMALY_LABEL] = True
            rec[ANOMALY_TYPE] = str(kind)

    df = pl.DataFrame(records, schema={
        DATE: pl.Datetime("us"),
        DESC: pl.Utf8,
        AMOUNT: pl.Float64,
        CATEGORY: pl.Utf8,
        ACCOUNT: pl.Utf8,
        CURRENCY: pl.Utf8,
        ANOMALY_LABEL: pl.Boolean,
        ANOMALY_TYPE: pl.Utf8,
    }).sort([DATE])

    n = df.height
    df = df.with_columns(
        ((pl.int_range(0, n, eager=True) + 1)).alias(TXN_ID),
        (pl.col(AMOUNT) < 0).alias("is_debit"),
        pl.col(AMOUNT).abs().alias("abs_amount"),
    )
    return df.select(
        TXN_ID, DATE, DESC, AMOUNT, CATEGORY, ACCOUNT, CURRENCY,
        "is_debit", "abs_amount", ANOMALY_LABEL, ANOMALY_TYPE,
    )


def write_csv(df: pl.DataFrame, path: str) -> str:
    """Write a bank-export-shaped CSV (date, description, amount)."""
    out = df.select(
        pl.col(DATE).dt.strftime("%Y-%m-%d").alias("Date"),
        pl.col(DESC).alias("Description"),
        pl.col(AMOUNT).round(2).alias("Amount"),
        pl.col(ACCOUNT).alias("Account"),
        pl.col(CURRENCY).alias("Currency"),
    )
    out.write_csv(path)
    return path
