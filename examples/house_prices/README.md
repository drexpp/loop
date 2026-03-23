# House Prices — Regression

**Task:** predict the sale price of residential homes in Ames, Iowa  
**Metric:** negative RMSE on log1p(SalePrice) — higher is better (5-fold KFold, seed=42)  
**Dataset:** 1,460 rows, 79 features

---

## Results

24 experiments. No human intervention after setup. A laptop.

```
id  val_metric  status     description
0   -0.159835   baseline   Ridge regression, numeric features only
1   -0.159835   discarded  Added TotalSF (collinear for Ridge)
2   -0.148687   keep       Switched to RandomForestRegressor          ← +0.011
3   -0.144822   keep       Added ordinal quality encoding and TotalSF ← +0.004
4   -0.139446   keep       Switched to HistGradientBoostingRegressor  ← +0.005
5   -0.137906   keep       Added HouseAge and RemodAge features       ← +0.002
6   -0.138535   discarded  Label encoding for categoricals (worse)
7   -0.139506   discarded  OverallQual*TotalSF interaction (worse for HGBR)
8   -0.142386   discarded  HGBR lr=0.05 max_iter=500 depth=5 (underfitting)
9   -0.136153   keep       Switched to XGBRegressor                  ← +0.002
10  -0.134515   keep       Target encode Neighborhood                 ← +0.002
11  -0.131370   keep       Target encode 7 categoricals               ← +0.003
12  -0.129294   keep       TotalBathrooms + TotalPorchSF features     ← +0.002
13  -0.128881   keep       XGB n_estimators=500 lr=0.05 reg_alpha=0.1
14  -0.127755   keep       XGB n=800 lr=0.03 depth=3                 ← +0.001
15  -0.125866   keep       Target encode 15 categoricals             ← +0.002
16  -0.126661   discarded  XGB+LGBM blend (50/50)
17  -0.123048   keep       Remove outliers (GrLivArea > 4000)        ← +0.003
18  -0.123048   discarded  Log transform skewed features (no change)
19  -0.125619   discarded  Treat MSSubClass and MoSold as categorical
20  -0.125689   discarded  Polynomial features (OverallQual², GrLivArea²)
21  -0.125681   discarded  Binary indicators for amenities
22  -0.124394   discarded  XGB tuning (n=2000, lr=0.01, subsample=0.7)
23  -0.127177   discarded  Label encode all strings
24  -0.124267   discarded  Smoothed target encoding
```

**RMSE −0.160 → −0.123** across 24 experiments.

```
Kaggle leaderboard reference:
  Top 10%  → RMSE ~0.115   (score ~ −0.115)
  Top 25%  → RMSE ~0.130   (score ~ −0.130)  ← crossed at exp 11
  Baseline → RMSE ~0.160   (score ~ −0.160)
```

---

## What the agent discovered on its own

**TotalSF is model-dependent.**  
Experiment 1 discarded TotalSF because it hurt Ridge (collinearity).
The agent added it back in experiment 3 only after switching to a tree-based model
that handles correlated features natively. It found the interaction between
feature and model type without being told.

**Target encoding is the key to unlocking categoricals.**  
Experiment 6 tried label encoding all categoricals — it made things worse.
The agent discarded it, then found a better approach: target encoding individual
columns one by one, starting with `Neighborhood` (exp 10, +0.002) and expanding
to 15 columns by exp 15 (+0.002 more). Systematic exploration, not random search.

**Blending only works with diverse models.**  
Experiment 16 tried XGBoost + LightGBM blend (50/50) — no improvement.
The agent correctly identified this as redundant: both models have the same
feature set and make correlated errors. It discarded the blend and moved on.

**Outlier removal is a structural fix, not a tweak.**  
Experiment 17 removed houses with GrLivArea > 4000 — the biggest single
improvement in the second half of the run (+0.003). These are anomalous
data points that skew the loss function. The agent found this without
any domain knowledge hint.

**The plateau is real and informative.**  
Experiments 18-24 all discarded. The agent explored log transforms,
categorical encodings, polynomial features, and tuning variants — none improved.
This is the agent confirming that the current feature set and model are
near the optimum for this approach. Further gains require architectural changes
(stacking, external data) rather than incremental improvements.

---

## How to run it

```bash
# 1. Get the data
# Download from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
# Place train.csv in data/train.csv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the environment
python data.py
# Should print: data.py OK

# 4. Check the current best model
python solve.py
# Should print: val_metric: -0.123048

# 5. Point your agent at program.md
# "Read program.md and follow the instructions exactly."
```

---

## File structure

```
house_prices/
├── program.md       ← instructions for the agent
├── data.py          ← locked evaluator (do not modify)
├── solve.py         ← current best model (the agent improves this)
├── results.tsv      ← full experiment history
├── requirements.txt
└── data/
    └── train.csv    ← download from Kaggle
```

---

## The solve.py the agent built

The final `solve.py` contains techniques the agent discovered through iteration:

- **Outlier removal** — filters `GrLivArea > 4000` from training data
- **Ordinal quality encoding** — maps Ex/Gd/TA/Fa/Po/None to 5/4/3/2/1/0 for 9 quality columns
- **Age features** — `HouseAge = YrSold − YearBuilt`, `RemodAge = YrSold − YearRemodAdd`
- **Bathroom aggregation** — `TotalBathrooms` combining full, half, basement baths
- **Porch aggregation** — `TotalPorchSF` summing all porch/deck areas
- **Target encoding with smoothing** — 15 categorical columns encoded using
  smoothed target means (α=10) computed from training fold only, never leaking val data
- **XGBoost** — n=800, lr=0.03, depth=3, with L1 regularization

None of these techniques were suggested in `program.md`.
The agent found them by exploring the feature and model space from first principles.