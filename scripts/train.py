#!/usr/bin/env python
"""Train the expense/transaction models and write artifacts/.

    python scripts/train.py --rows 5000 --categorizer lightgbm --method ensemble
    python scripts/train.py --csv data/raw/my_export.csv --no-labels   # score only

Ground truth comes from the bundled generator unless ``--csv`` is given; real
bank exports have neither category labels nor fraud labels, so they can be
scored but not evaluated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expense_ml.anomaly import ALL_METHODS                      # noqa: E402
from expense_ml.categorize import available_backends            # noqa: E402
from expense_ml.ingest import load_transactions                 # noqa: E402
from expense_ml.pipeline import (ExpenseModel, TrainConfig,     # noqa: E402
                                 save_metrics, train_model)
from expense_ml.synthetic import make_transactions, write_csv  # noqa: E402

MODEL = ROOT / "artifacts" / "expense_model.joblib"


def metrics_path_for(model_path: Path) -> Path:
    """Keep metrics beside the model, so ``--out`` cannot split them apart."""
    return Path(model_path).resolve().parent / "metrics.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=5000, help="synthetic rows to generate")
    ap.add_argument("--seed", type=int, default=7, help="generator seed")
    ap.add_argument("--csv", type=Path, help="score an existing export instead of training")
    ap.add_argument("--categorizer", default="lightgbm",
                    choices=sorted(available_backends()))
    ap.add_argument("--method", default="ensemble", choices=list(ALL_METHODS))
    ap.add_argument("--contamination", type=float, default=0.03)
    ap.add_argument("--folds", type=int, default=3, help="out-of-fold folds")
    ap.add_argument("--out", type=Path, default=MODEL, help="where to write the model")
    ap.add_argument("--skip-rows", type=int, default=0)
    args = ap.parse_args()

    cfg = TrainConfig(
        categorizer=args.categorizer,
        anomaly_method=args.method,
        contamination=args.contamination,
        oof_folds=args.folds,
    )

    t0 = time.time()
    model = ExpenseModel(cfg)

    if args.csv:
        df, report = load_transactions(args.csv, skip_rows=args.skip_rows)
        print(f"read {report.rows_kept:,} rows from {args.csv} "
              f"({report.rows_dropped:,} quarantined)")
        for w in report.warnings:
            print(f"  warning: {w}")
        if "category" not in df.columns or df["category"].null_count() == df.height:
            print("--no-labels: no category column; scoring only.", file=sys.stderr)
            model = ExpenseModel.load(MODEL) if MODEL.exists() else None
            if model is None:
                print("no saved model to score with; train first.", file=sys.stderr)
                return 2
            out = model.score(df)
            dest = args.out.with_name("scored.csv")
            out.drop([c for c in ("_desc_lower",) if c in out.columns]).write_csv(dest)
            print(f"wrote {dest} ({out.height:,} rows)")
            return 0
    else:
        df = make_transactions(args.rows, seed=args.seed)
        write_csv(df, ROOT / "data" / "raw" / "sample_transactions.csv")
        print(f"generated {df.height:,} rows across {df['category'].n_unique()} categories")

    result = train_model(df, cfg, verbose=True)
    path = result.model.save(args.out)
    metrics_path = metrics_path_for(args.out)
    save_metrics(result.metrics, metrics_path)

    m = result.metrics
    print(f"\ntrained in {result.elapsed_s:.1f}s -> {path}")
    print(f"  categorizer : acc={m['categorization']['accuracy']:.4f} "
          f"macroF1={m['categorization']['macro_f1']:.4f}")
    a = m["anomaly"]
    if a.get("ground_truth"):
        print(f"  anomaly     : AP={a['average_precision']:.3f} "
              f"P={a['precision']:.3f} R={a['recall']:.3f} F1={a['f1']:.3f} "
              f"lift={a['lift']:.1f}x (base rate {a['random_baseline_ap']:.3f})")
        for name, v in (a.get("detection_by_type") or {}).items():
            print(f"      {name:20} {v['catch_rate']:5.1%} of {v['n']}")
    print(f"  metrics     -> {metrics_path}")
    print(json.dumps(m["backends"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
