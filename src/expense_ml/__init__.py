"""Smart Expense & Financial Anomalies Predictor.

Tabular ML pipeline that ingests bank exports, auto-categorises spending with
text classification, and flags anomalous transactions with unsupervised
outlier detection.
"""

from expense_ml.config import CATEGORIES, IngestReport

__all__ = ["CATEGORIES", "IngestReport"]
__version__ = "0.1.0"
