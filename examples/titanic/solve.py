"""
solve.py — THE ONLY FILE THE AGENT EDITS.

Contract (defined by data.py — do not change):
  1. Function pipeline(X_train, y_train, X_val) -> np.ndarray
     Returns class-1 probabilities, shape (n_val,)

  2. When run as __main__, prints exactly:
       val_metric: X.XXXXXX
     (the agent greps this line — do not change the format)

  3. Do not import data.py internals. Only use: evaluate, load_raw.
  4. Do not add dependencies outside requirements.txt.

Baseline: Random Forest + basic features. Expected AUC ~0.845
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data import evaluate


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Sex"]      = (df["Sex"] == "female").astype(int)
    df["Age"]      = df["Age"].fillna(df["Age"].median())
    df["AgeBin"]   = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    
    df["Fare"]     = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FamilyGroup"] = df["FamilySize"].map(lambda s: 0 if s == 1 else (1 if s <= 4 else 2))

    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
    df["Title"] = df["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df["Title"] = df["Title"].replace('Mlle', 'Miss')
    df["Title"] = df["Title"].replace('Ms', 'Miss')
    df["Title"] = df["Title"].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df["Title"] = df["Title"].map(title_mapping).fillna(0)

    df["Pclass_Age"] = df["Pclass"] * df["AgeBin"]
    df["Sex_Pclass"] = df["Sex"] * df["Pclass"]

    df["Sex_Family"] = df["Sex"] * df["FamilyGroup"]

    df["IsMaster"] = (df["Title"] == 4).astype(int)

    df["Pclass_Family"] = df["Pclass"] * df["FamilyGroup"]
    df["Age_Fare"] = df["Age"] * df["Fare"]

    return df[["Pclass", "Sex", "AgeBin", "Fare", "FamilyGroup", "Embarked", "Title", "Pclass_Age", "Sex_Pclass", "Sex_Family", "IsMaster", "Pclass_Family", "Age_Fare"]]


def pipeline(X_train: pd.DataFrame,
             y_train: pd.Series,
             X_val:   pd.DataFrame) -> np.ndarray:

    X_tr = engineer_features(X_train)
    X_v  = engineer_features(X_val)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_tr, y_train)
    return model.predict_proba(X_v)[:, 1]


if __name__ == "__main__":
    auc = evaluate(pipeline)
    print(f"val_metric: {auc:.6f}")