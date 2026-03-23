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

    # Add TotalSF feature (classic House Prices feature)
    if all(c in df.columns for c in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df['TotalSF'] = df['TotalBsmtSF'].fillna(0) + df['1stFlrSF'].fillna(0) + df['2ndFlrSF'].fillna(0)

    # Ordinal encoding for quality metrics
    qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    for col in qual_cols:
        if col in df.columns:
            df[col + '_num'] = df[col].fillna('None').map(qual_map).fillna(0)

            
    # Add age features
    if all(c in df.columns for c in ['YrSold', 'YearBuilt', 'YearRemodAdd']):
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    # Add bathroom count
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(c in df.columns for c in bath_cols):
        df['TotalBathrooms'] = (df['FullBath'].fillna(0) + df['BsmtFullBath'].fillna(0) 
                                + 0.5 * df['HalfBath'].fillna(0) + 0.5 * df['BsmtHalfBath'].fillna(0))

    # Total porch SF
    porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'WoodDeckSF']
    available_porch = [c for c in porch_cols if c in df.columns]
    if available_porch:
        df['TotalPorchSF'] = df[available_porch].fillna(0).sum(axis=1)

    # Select only numeric columns as a safe starting point
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Basic imputation — fill nulls with column median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

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
    from xgboost import XGBRegressor

    # Filter outliers in training data (standard House Prices refinement)
    if 'GrLivArea' in X_train.columns:
        train_mask = X_train['GrLivArea'] < 4000
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

    X_tr = engineer_features(X_train)
    X_v  = engineer_features(X_val)

    # Target encode key categoricals (computed from train, applied to val)
    te_cols = ['Neighborhood', 'MSZoning', 'SaleCondition', 'Exterior1st', 
               'Foundation', 'BldgType', 'HouseStyle', 'Condition1',
               'LotConfig', 'GarageType', 'Exterior2nd', 'RoofStyle',
               'LotShape', 'LandContour', 'SaleType']
    global_mean = y_train.mean()
    alpha = 10  # Smoothing parameter
    for col in te_cols:
        if col in X_train.columns:
            stats = X_train.assign(_y=y_train.values).groupby(col)['_y'].agg(['mean', 'count'])
            # Smoothed mean: (mean * count + global_mean * alpha) / (count + alpha)
            smoothed = (stats['mean'] * stats['count'] + global_mean * alpha) / (stats['count'] + alpha)
            X_tr[col + '_te'] = X_train[col].map(smoothed).fillna(global_mean)
            X_v[col + '_te'] = X_val[col].map(smoothed).fillna(global_mean)

    model = XGBRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    model.fit(X_tr, y_train)
    return model.predict(X_v)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    score = evaluate(pipeline)
    print(f"val_metric: {score:.6f}")