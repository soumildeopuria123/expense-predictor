#!/usr/bin/env python
"""Regenerate the bundled sample dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_ml.synthetic import ANOMALY_LABEL, make_transactions, write_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "sample_transactions.csv")
    args = ap.parse_args()

    df = make_transactions(args.rows, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(df, args.out)

    print(f"wrote {args.out} ({df.height:,} rows)")
    print(f"  categories    : {df['category'].n_unique()}")
    print(f"  span           : {df['date'].min()} -> {df['date'].max()}")
    print(f"  injected anomalies: {int(df[ANOMALY_LABEL].sum()):,} "
          f"({df[ANOMALY_LABEL].mean():.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
