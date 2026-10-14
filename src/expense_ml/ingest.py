"""Bank-export ingestion built on Polars.

Responsibilities
----------------
* Accept a path, raw bytes or a file-like object (Streamlit uploads hand us
  ``BytesIO``).
* Resolve messy real-world headers onto the canonical schema.
* Normalise amount semantics to **signed** values (negative = money out),
  covering three common export shapes:
    1. a single signed ``amount`` column,
    2. separate ``debit`` / ``credit`` columns,
    3. an unsigned amount plus a ``Dr``/``Cr`` direction column.
* Parse dates across a spread of formats, including ISO with time and Unix
  epochs.
* Quarantine unparseable rows rather than silently coercing them, and report
  exactly what was dropped so the UI can show it.

Everything is read as string first (``infer_schema_length=0``); letting Polars
infer types is what produces those "amount parsed as string" failures on
exports with stray currency symbols.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import IO, Any, Union

import polars as pl

from expense_ml.config import (
    ACCOUNT,
    AMOUNT,
    CANONICAL_COLUMNS,
    CATEGORY,
    CURRENCY,
    DATE,
    DESC,
    DIRECTION,
    IngestReport,
    TXN_ID,
    _DATETIME_FORMATS,
    _DATE_FORMATS,
    resolve_columns,
)

Source = Union[str, Path, bytes, bytearray, IO[bytes]]

_CURRENCY_JUNK = r"[^0-9.\-+]"
_EPOCH_START = 315532800  # 1980-01-01, in seconds; filters out tiny integers


class IngestError(ValueError):
    """Raised when a file cannot be mapped onto the canonical schema."""


def _as_reader(source: Source) -> IO[bytes]:
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(bytes(source))
    if isinstance(source, Path):
        return source.open("rb")
    if isinstance(source, str):
        # A str is ambiguous: it may be a path or the CSV itself. Newlines
        # decide it -- a filesystem path can never contain one, and inline CSV
        # virtually always does. Keeps `load_transactions(csv_text)` working
        # alongside `load_transactions("export.csv")`.
        if "\n" in source:
            return io.BytesIO(source.encode("utf-8"))
        p = Path(source)
        if p.exists():
            return p.open("rb")
        raise IngestError(f"No such file: {source}")
    if hasattr(source, "read"):
        return source  # type: ignore[return-value]
    raise IngestError(f"Unsupported source type: {type(source)!r}")


def _parse_datetime(col: str) -> pl.Expr:
    """Try explicit formats first, then ISO inference, then Unix epochs."""
    base = pl.col(col).str.strip_chars()
    parsed: pl.Expr | None = None
    for fmt in (*_DATETIME_FORMATS, *_DATE_FORMATS):
        attempt = base.str.to_datetime(fmt, strict=False, cache=False)
        parsed = attempt if parsed is None else pl.coalesce([parsed, attempt])

    # Bare ISO dates/timestamps that the explicit formats above already cover
    # mostly resolve here; this also catches things like "2024-1-5".
    inferred = base.str.to_datetime(None, strict=False, cache=False)

    # Epoch seconds. Guarded so we never treat a bare integer ID as a date.
    as_int = base.cast(pl.Int64, strict=False)
    epoch = (
        pl.when(as_int.is_not_null() & (as_int.abs() > pl.lit(_EPOCH_START)))
        .then(pl.from_epoch(as_int, time_unit="us"))
        .otherwise(None)
    )

    out = pl.coalesce([parsed, inferred, epoch])
    return out.cast(pl.Datetime("us"))


def _parse_amount_parts(col: str) -> tuple[pl.Expr, pl.Expr]:
    """Return ``(value_expr, used_european_separator_expr)``.

    Money strings are genuinely ambiguous: "1,234" is one thousand two hundred
    thirty-four in the US and one point two three four in much of Europe, and
    getting it wrong silently corrupts every downstream figure. So the rules are
    explicit and reported rather than guessed at:

    * both separators present -> the later one is the decimal point;
    * one separator, appearing once with exactly 3 trailing digits -> treated
      as a thousands separator (US convention, matching our USD default);
    * any other single separator -> decimal point (covers "12,5" and "12.5").

    The second return value flags rows where the European reading was taken, so
    the ingest report can surface them.
    """
    s = pl.col(col).str.strip_chars()
    body = s.str.replace_all(r"[^0-9.,-]", "")

    last_sep = body.str.extract(r"^(.*[0-9])([.,])([0-9]*)$", 2)
    tail_len = (
        body.str.extract(r"^(.*[0-9])([.,])([0-9]*)$", 3)
        .str.len_chars()
        .fill_null(0)
    )
    n_dot = body.str.count_matches(r"\.")
    n_comma = body.str.count_matches(r",")

    has_both = (n_dot > 0) & (n_comma > 0)
    comma_as_thousands = (n_comma == 1) & (n_dot == 0) & (tail_len == 3)
    dot_as_thousands = (n_dot == 1) & (n_comma == 0) & (tail_len == 3)
    plain = (n_dot == 0) & (n_comma == 0)

    is_dot_decimal = (
        (has_both & (last_sep == "."))
        | (~has_both & ~comma_as_thousands & ~dot_as_thousands & ~plain & (n_dot > 0))
    )
    is_comma_decimal = (
        (has_both & (last_sep == ","))
        | (~has_both & ~comma_as_thousands & ~dot_as_thousands & ~plain & (n_comma > 0))
    )

    normalised = (
        pl.when(is_dot_decimal)
        .then(body.str.replace_all(",", ""))                 # dots stay as decimal
        .when(is_comma_decimal)
        .then(                                              # commas become decimal
            body.str.replace_all(r"\.", "").str.replace_all(",", ".")
        )
        .otherwise(body.str.replace_all(r"[.,]", ""))       # all separators are thousands
    )

    paren_negative = s.str.starts_with("(") & s.str.ends_with(")")
    minus_negative = s.str.starts_with("-")
    unsigned = normalised.str.replace_all(r"^[+-]", "")
    sign = (
        pl.when(paren_negative | minus_negative).then(pl.lit("-")).otherwise(pl.lit(""))
    )
    signed = (sign + unsigned).str.replace_all(r"^-$", "")
    return signed.cast(pl.Float64, strict=False), is_comma_decimal


def _parse_amount(col: str) -> pl.Expr:
    """Parse a money string into a signed float (see :func:`_parse_amount_parts`)."""
    return _parse_amount_parts(col)[0]


def load_transactions(
    source: Source,
    *,
    skip_rows: int = 0,
    default_currency: str = "USD",
    max_reported_drops: int = 5,
) -> tuple[pl.DataFrame, IngestReport]:
    """Load a bank export into the canonical schema.

    Returns a ``(frame, report)`` pair; the report is deliberately separate so
    a caller can show data-quality problems instead of swallowing them.
    """
    raw = pl.read_csv(
        _as_reader(source),
        infer_schema_length=0,          # read everything as string
        skip_rows=skip_rows,
        ignore_errors=False,
        null_values=["", "NA", "N/A", "null", "NULL", "None"],
        truncate_ragged_lines=True,
    )

    if raw.height == 0:
        raise IngestError("The file has a header but no data rows.")
    if raw.width == 0:
        raise IngestError("The file appears to be empty.")

    report = IngestReport(rows_in=raw.height)
    resolved = resolve_columns(raw.columns)
    report.resolved_columns = resolved

    missing = [f for f in (DATE, DESC) if f not in resolved]
    has_amount = AMOUNT in resolved
    has_pair = {"debit", "credit"} <= set(resolved)
    if missing or not (has_amount or has_pair):
        raise IngestError(
            "Could not map this file onto the expected schema.\n"
            f"  Found columns: {raw.columns}\n"
            f"  Matched: {resolved}\n"
            f"  Missing required: {missing or '(need amount or debit/credit)'}\n"
            "Expected at least a date column, a description column, and either "
            "an amount column or a debit/credit pair."
        )

    # --- date ---------------------------------------------------------------
    date_col = resolved[DATE]
    frame = raw.with_columns(_parse_datetime(date_col).alias("_date"))

    # --- amount -------------------------------------------------------------
    frame = frame.with_columns(
        _parse_amount(resolved[AMOUNT]).alias("_amount_signed")
        if has_amount
        else pl.lit(None, dtype=pl.Float64).alias("_amount_signed")
    )
    if has_pair:
        debit = _parse_amount(resolved["debit"])
        credit = _parse_amount(resolved["credit"])
        # A row whose debit *and* credit are both empty carries no amount at all.
        # Filling each with 0.0 would invent a $0.00 transaction (and would
        # override a real `amount` column via the coalesce below), so keep it
        # null and let the quarantine drop it.
        both_missing = debit.is_null() & credit.is_null()
        computed = credit.fill_null(0.0) - debit.fill_null(0.0)
        frame = frame.with_columns(
            pl.when(both_missing)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(computed)
            .alias("_amount_pair")
        )
    else:
        frame = frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("_amount_pair")
        )

    # Direction column overrides the sign for unsigned amounts.
    if DIRECTION in resolved and has_amount:
        d = pl.col(resolved[DIRECTION]).str.to_lowercase().str.strip_chars()
        is_out = d.is_in(
            ["debit", "dr", "out", "d", "withdrawal", "paid out", "paidout", "-"]
        )
        is_in = d.is_in(["credit", "cr", "in", "c", "deposit", "paid in", "paidin", "+"])
        base = pl.col("_amount_signed").abs()
        frame = frame.with_columns(
            pl.when(is_out & ~is_in)
            .then(-base)
            .when(is_in & ~is_out)
            .then(base)
            .otherwise(pl.col("_amount_signed"))
            .alias("_amount_signed")
        )

    if has_amount:
        _, euro_expr = _parse_amount_parts(resolved[AMOUNT])
        frame = frame.with_columns(euro_expr.alias("_euro_sep"))
        n_euro = int(frame["_euro_sep"].sum())
        if n_euro:
            report.warnings.append(
                f"{n_euro} amount(s) parsed using European decimal separators "
                "(e.g. '1.234,56'). Confirm the expected convention if totals "
                "look wrong."
            )
    frame = frame.with_columns(
        pl.coalesce([pl.col("_amount_pair"), pl.col("_amount_signed")]).alias(
            "_amount"
        )
    )

    # --- metadata -----------------------------------------------------------
    desc_col = resolved[DESC]
    frame = frame.with_columns(
        pl.col(desc_col).str.strip_chars().fill_null("").alias("_desc"),
        (
            pl.col(resolved[ACCOUNT]).str.strip_chars().alias("_account")
            if ACCOUNT in resolved
            else pl.lit("default").alias("_account")
        ),
        (
            pl.col(resolved[CURRENCY]).str.to_uppercase().str.strip_chars().alias(
                "_currency"
            )
            if CURRENCY in resolved
            else pl.lit(default_currency).alias("_currency")
        ),
        (
            pl.col(resolved[CATEGORY]).str.strip_chars().alias("_category")
            if CATEGORY in resolved
            else pl.lit(None, dtype=pl.Utf8).alias("_category")
        ),
    )

    # --- quarantine ---------------------------------------------------------
    bad_date = frame.filter(pl.col("_date").is_null())
    bad_amount = frame.filter(pl.col("_date").is_not_null() & pl.col("_amount").is_null())
    report.dropped_bad_date = bad_date.height
    report.dropped_bad_amount = bad_amount.height

    for bad, label in ((bad_date, "unparseable date"), (bad_amount, "unparseable amount")):
        for row in bad.head(max_reported_drops).iter_rows(named=True):
            report.dropped_examples.append(
                {
                    "reason": label,
                    "date": row.get(date_col),
                    "description": row.get(desc_col),
                    "amount": row.get(resolved[AMOUNT]) if has_amount else None,
                }
            )

    clean = frame.filter(pl.col("_date").is_not_null() & pl.col("_amount").is_not_null())

    if report.dropped_bad_date:
        report.warnings.append(
            f"{report.dropped_bad_date} row(s) dropped: date could not be parsed."
        )
    if report.dropped_bad_amount:
        report.warnings.append(
            f"{report.dropped_bad_amount} row(s) dropped: amount could not be parsed."
        )
    if clean.height == 0:
        raise IngestError(
            "Every row was rejected -- the date/amount columns were not recognised "
            f"as such. Matched columns: {resolved}"
        )

    # --- tidy canonical output ---------------------------------------------
    out = clean.select(
        (pl.int_range(0, clean.height, eager=True).cast(pl.Int64) + 1).alias(TXN_ID),
        pl.col("_date").alias(DATE),
        pl.col("_desc").alias(DESC),
        pl.col("_amount").cast(pl.Float64).alias(AMOUNT),
        pl.col("_category").alias(CATEGORY),
        pl.col("_account").alias(ACCOUNT),
        pl.col("_currency").alias(CURRENCY),
    ).with_columns(
        (pl.col(AMOUNT) < 0).alias("is_debit"),
        pl.col(AMOUNT).abs().alias("abs_amount"),
        pl.col(DESC).str.to_lowercase().alias("_desc_lower"),
    )
    out = out.sort([DATE, TXN_ID])

    report.rows_kept = out.height
    return out, report


def canonical_to_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return only the documented canonical columns, in order."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Frame is missing canonical columns: {missing}")
    return df.select(list(CANONICAL_COLUMNS))
