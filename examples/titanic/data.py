"""
data.py — LOCKED FILE. The agent must NOT modify this file.

Public contract:
  load_raw()        -> (train_df, test_df)
  evaluate(pipeline_fn) -> float   (mean AUC-ROC, 5-fold stratified CV)

Changing this file invalidates all previous experiments.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ── Locked configuration ───────────────────────────────────────────────────
DATA_DIR   = pathlib.Path(__file__).parent / "data"
TRAIN_FILE = DATA_DIR / "train.csv"
TARGET_COL = "Survived"

RANDOM_SEED = 42
N_FOLDS     = 5


# ── Public API ─────────────────────────────────────────────────────────────

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train_df, test_df) with no preprocessing."""
    if not TRAIN_FILE.exists():
        sys.exit(
            f"ERROR: {TRAIN_FILE} not found.\n"
            "Download the dataset:\n"
            "  kaggle competitions download -c titanic -p data/\n"
            "Or manually place train.csv in the data/ folder."
        )
    train = pd.read_csv(TRAIN_FILE)
    # test.csv is optional — return empty df if missing
    test_file = DATA_DIR / "test.csv"
    test = pd.read_csv(test_file) if test_file.exists() else pd.DataFrame()
    return train, test


def evaluate(pipeline_fn) -> float:
    """
    Evaluates pipeline_fn with 5-fold Stratified CV.

    pipeline_fn contract:
        pipeline_fn(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    X_val:   pd.DataFrame) -> np.ndarray  (probabilities, shape [n,])

    Returns mean AUC-ROC across 5 folds. Higher is better.

    IMPORTANT: this function defines the official experiment metric.
    The agent cannot change it. Doing so invalidates historical comparison.
    """
    train, _ = load_raw()

    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    aucs: list[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr  = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr  = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]

        try:
            y_proba = pipeline_fn(X_tr, y_tr, X_val)
        except Exception as exc:
            sys.exit(f"ERROR in fold {fold_idx}: {exc}")

        y_proba = np.asarray(y_proba, dtype=float)

        if y_proba.shape != (len(y_val),):
            sys.exit(
                f"ERROR fold {fold_idx}: pipeline_fn must return array of shape "
                f"({len(y_val)},), got {y_proba.shape}"
            )

        aucs.append(roc_auc_score(y_val, y_proba))

    return float(np.mean(aucs))


# ── Script mode: verify environment ───────────────────────────────────────

if __name__ == "__main__":
    print("Verifying data.py...")
    train, _ = load_raw()
    print(f"  train shape : {train.shape}")
    print(f"  target dist : {train[TARGET_COL].value_counts().to_dict()}")
    print(f"  null counts : {train.isnull().sum()[train.isnull().sum() > 0].to_dict()}")
    print("data.py OK")