"""
solve.py — THE ONLY FILE THE AGENT EDITS.

Contract (defined by data.py — do not change the interface):
  1. Function pipeline(X_train, y_train, X_val) -> np.ndarray
     Returns probabilities (classification) or predicted values (regression)

  2. When run as __main__, prints exactly:
       val_metric: X.XXXXXX
     (the agent greps this line — do not change the format)

  3. Do not import data.py internals. Only use: evaluate.
  4. Do not add dependencies outside requirements.txt.

Baseline: Logistic Regression with basic numeric features.
Replace with the appropriate baseline for your problem type — see comments.
"""

import numpy as np
import pandas as pd

from data import evaluate


# ── Feature engineering ────────────────────────────────────────────────────
# This is where the agent will make most of its changes.
# Start simple — the agent will improve this.

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw DataFrame into numeric features.
    The agent modifies this function to improve the metric.
    """
    df = df.copy()

    # Select only numeric columns as a safe starting point
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Basic imputation — fill nulls with column median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # TODO: the agent will add feature engineering here
    # Examples of what the agent might add:
    #   - Encode categorical columns
    #   - Create interaction features
    #   - Extract features from text/dates
    #   - Bin continuous variables

    return df[numeric_cols]


# ── Pipeline ───────────────────────────────────────────────────────────────

def pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
) -> np.ndarray:
    """
    Receives raw data from one CV fold.
    Returns predictions — probabilities for classification, values for regression.

    The agent rewrites this function to improve the metric.

    ── Baseline options by problem type ──────────────────────────────────────

    BINARY CLASSIFICATION (current — returns probabilities):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_tr, y_train)
        return model.predict_proba(X_v)[:, 1]

    MULTI-CLASS CLASSIFICATION (returns predicted classes):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_tr, y_train)
        return model.predict(X_v)

    REGRESSION (returns predicted values):
        from sklearn.linear_model import Ridge
        model = Ridge()
        model.fit(X_tr, y_train)
        return model.predict(X_v)
    ─────────────────────────────────────────────────────────────────────────
    """
    from sklearn.linear_model import Ridge

    X_tr = engineer_features(X_train)
    X_v  = engineer_features(X_val)

    model = Ridge(random_state=42)
    model.fit(X_tr, y_train)
    return model.predict(X_v)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    score = evaluate(pipeline)
    print(f"val_metric: {score:.6f}")