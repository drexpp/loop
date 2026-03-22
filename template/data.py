"""
data.py — LOCKED FILE. The agent must NOT modify this file.

Public contract:
  load_raw()             -> (train_df, test_df)
  evaluate(pipeline_fn)  -> float   (higher is always better)

HOW TO ADAPT THIS FILE TO YOUR PROBLEM:
  1. Set TRAIN_FILE to your training CSV path
  2. Set TARGET_COL to the column you want to predict
  3. Choose the right metric and CV strategy (see options below)

Once the loop starts, do not change this file.
Changing it invalidates all previous experiments.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

# ── STEP 1: Point to your data ─────────────────────────────────────────────
DATA_DIR   = pathlib.Path(__file__).parent / "data"
TRAIN_FILE = DATA_DIR / "train.csv"   # ← change filename if needed

# ── STEP 2: Set your target column ────────────────────────────────────────
TARGET_COL = "target"                  # ← change to your target column name

# ── Locked configuration (do not change after first run) ──────────────────
RANDOM_SEED = 42
N_FOLDS     = 5


# ── Public API ─────────────────────────────────────────────────────────────

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train_df, test_df) with no preprocessing."""
    if not TRAIN_FILE.exists():
        sys.exit(
            f"ERROR: {TRAIN_FILE} not found.\n"
            f"Place your training CSV at {TRAIN_FILE}"
        )
    train     = pd.read_csv(TRAIN_FILE)
    test_file = DATA_DIR / "test.csv"
    test      = pd.read_csv(test_file) if test_file.exists() else pd.DataFrame()
    return train, test


def evaluate(pipeline_fn) -> float:
    """
    Evaluates pipeline_fn with cross-validation.
    Returns a scalar — higher is always better.

    pipeline_fn contract:
        pipeline_fn(X_train, y_train, X_val) -> np.ndarray

    ── STEP 3: Choose your metric and CV strategy ──────────────────────────

    BINARY CLASSIFICATION (e.g. fraud, churn, survival):
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        cv  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        # pipeline_fn returns probabilities: model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)

    MULTI-CLASS CLASSIFICATION:
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold
        cv  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        # pipeline_fn returns predicted classes: model.predict(X_val)
        score = f1_score(y_val, y_pred, average='macro')

    REGRESSION (e.g. prices, demand, scores):
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import KFold
        cv  = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        # pipeline_fn returns predicted values: model.predict(X_val)
        score = -mean_squared_error(y_val, y_pred, squared=False)  # negative RMSE

    TIME SERIES (e.g. forecasting — use this if data has a time column):
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import TimeSeriesSplit
        cv  = TimeSeriesSplit(n_splits=N_FOLDS)
        # pipeline_fn returns predicted values
        score = -mean_absolute_error(y_val, y_pred)  # negative MAE
    ────────────────────────────────────────────────────────────────────────
    """
    # ── Default: binary classification with AUC-ROC ───────────────────────
    # Replace this block with the appropriate option from above
    from sklearn.metrics import roc_auc_score          # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

    train, _ = load_raw()
    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    cv     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores: list[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr  = X.iloc[tr_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr  = y.iloc[tr_idx]
        y_val = y.iloc[val_idx]

        try:
            y_pred = pipeline_fn(X_tr, y_tr, X_val)
        except Exception as exc:
            sys.exit(f"ERROR in fold {fold_idx}: {exc}")

        y_pred = np.asarray(y_pred, dtype=float)
        scores.append(roc_auc_score(y_val, y_pred))

    return float(np.mean(scores))


# ── Script mode: verify environment ───────────────────────────────────────

if __name__ == "__main__":
    print("Verifying data.py...")
    train, _ = load_raw()
    print(f"  train shape  : {train.shape}")
    print(f"  target col   : {TARGET_COL}")
    print(f"  target dist  : {train[TARGET_COL].value_counts().head().to_dict()}")
    print(f"  null counts  : {train.isnull().sum()[train.isnull().sum() > 0].to_dict()}")
    print("data.py OK")