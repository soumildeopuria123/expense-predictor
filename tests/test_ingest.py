import polars as pl
import pytest

from expense_ml.ingest import IngestError, load_transactions

HEADER = "Date,Memo,Amount,Account,Currency\n"


def test_signed_amount_parsing():
    csv = HEADER + "2024-01-05,COFFEE,-4.50,Chase,USD\n"
    df, rep = load_transactions(csv)
    assert df["amount"][0] == -4.50
    assert rep.rows_kept == 1 and rep.rows_dropped == 0


def test_accounting_parentheses_and_currency_symbols():
    csv = HEADER + (
        "2024-01-06,FEE,(45.00),Chase,USD\n"
        "2024-01-07,BIG,\"$1,234.56\",Chase,USD\n"
        "2024-01-08,PAYROLL,USD 3200.00,Chase,USD\n"
    )
    df, _ = load_transactions(csv)
    assert df["amount"].to_list() == [-45.0, 1234.56, 3200.0]


def test_european_decimal_separators_are_correct_not_silently_wrong():
    """'1.234,56' is one thousand two hundred 34.56 -- not 1.23456 million."""
    csv = HEADER + "2024-01-09,EUR,\"1.234,56\",Chase,USD\n2024-01-10,EUR2,\"-2 000,00\",Chase,USD\n"
    df, rep = load_transactions(csv)
    assert df["amount"].to_list() == [1234.56, -2000.0]
    assert any("European" in w for w in rep.warnings), "convention warning missing"


def test_unparseable_rows_are_quarantined_and_reported():
    csv = HEADER + "2024-01-05,OK,-1.00,Chase,USD\nnot-a-date,BAD,-2.00,Chase,USD\n2024-01-06,BAD,xyz,Chase,USD\n"
    df, rep = load_transactions(csv)
    assert df.height == 1, "bad rows must not silently coerce"
    assert rep.dropped_bad_date == 1 and rep.dropped_bad_amount == 1
    assert rep.rows_dropped == 2 and rep.drop_rate == pytest.approx(2 / 3)
    assert len(rep.dropped_examples) == 2
    assert any("date could not be parsed" in w for w in rep.warnings)


def test_alias_resolution_and_messy_headers():
    csv = "Booking Date,Narrative,Value,Account No,Dr/Cr,Currency\n2024-03-01,ACME,-5.00,123,DR,USD\n"
    df, rep = load_transactions(csv)
    assert rep.resolved_columns["date"] == "Booking Date"
    assert rep.resolved_columns["direction"] == "Dr/Cr"
    assert df["amount"][0] == -5.0


def test_debit_credit_columns():
    csv = "Date,Memo,Debit,Credit\n2024-02-01,PAYROLL,0.00,3200.00\n2024-02-02,RENT,1800.00,0.00\n"
    df, _ = load_transactions(csv)
    assert df["amount"].to_list() == [3200.0, -1800.0]
    assert df["is_debit"].to_list() == [False, True]


def test_unsigned_amount_with_direction_column():
    csv = "Date,Details,Amount,Dr/Cr\n2024-03-01,GROCERY,54.20,DR\n2024-03-02,REFUND,12.00,CR\n"
    df, _ = load_transactions(csv)
    assert df["amount"].to_list() == [-54.20, 12.00]


def test_datetime_with_time_is_preserved():
    csv = HEADER + "2024-04-05 13:22:11,POS,-1.00,Chase,USD\n"
    df, _ = load_transactions(csv)
    assert df["date"][0].hour == 13 and df["date"][0].minute == 22


def test_unmappable_file_raises_with_diagnostics():
    with pytest.raises(IngestError) as e:
        load_transactions("foo,bar\n1,2\n")
    assert "Found columns" in str(e.value)


def test_header_only_file_raises():
    with pytest.raises(IngestError):
        load_transactions(HEADER)


def test_row_with_no_amount_at_all_is_quarantined_not_zeroed():
    """Both debit and credit empty must not become a fabricated $0.00 txn.

    A $0.00 row would otherwise count as a real transaction in the velocity
    features and appear in the dashboard's flagged/spend tables.
    """
    csv = (
        "Date,Memo,Debit,Credit\n"
        "2024-04-01,COFFEE,4.50,\n"
        "2024-04-02,BROKEN ROW,,\n"
        "2024-04-03,LUNCH,12.00,\n"
    )
    df, rep = load_transactions(csv)
    assert df.height == 2, "an amount-less row must not be kept as $0.00"
    assert df["amount"].to_list() == [-4.5, -12.0]
    assert rep.dropped_bad_amount == 1
    assert any(r["reason"] == "unparseable amount" for r in rep.dropped_examples)


def test_amount_column_wins_when_debit_and_credit_are_empty():
    """A populated amount column must not be overridden by empty pair columns."""
    csv = (
        "Date,Memo,Amount,Debit,Credit\n"
        "2024-05-01,GROCERY,-42.50,,\n"
        "2024-05-02,PAYROLL,1000.00,,\n"
    )
    df, _ = load_transactions(csv)
    assert df["amount"].to_list() == [-42.5, 1000.0]
