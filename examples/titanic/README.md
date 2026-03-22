# Titanic — Binary Classification

**Task:** predict passenger survival on the Titanic  
**Metric:** AUC-ROC (5-fold stratified CV, seed=42)  
**Dataset:** 891 rows, 12 columns

---

## Results

32 experiments. No human intervention after setup. A laptop.

```
id  val_metric  status     description
0   0.866731    baseline   Random Forest, basic features
1   0.864204    discarded  Extract Title from Name
2   0.864469    discarded  Pclass * Sex interaction
3   0.874242    keep       RF max_depth=5                    ← +0.008
4   0.868534    discarded  Titles + max_depth=5
5   0.875129    keep       Age binning                       ← +0.001
6   0.872022    discarded  Fare binning (qcut)
7   0.871637    discarded  Simplified title grouping
8   0.870072    discarded  Sex * AgeBin interaction
9   0.877818    keep       Switch to XGBoost                 ← +0.003
10  0.881658    keep       FamilySize and IsAlone            ← +0.004
11  0.884463    keep       Simplified Titles + XGBoost       ← +0.003
12  0.884791    keep       XGBoost hyperparameter tuning
13  0.885167    keep       Pclass * AgeBin interaction
14  0.873497    discarded  Pclass * Age (raw) interaction
15  0.000000    crash      Sex * Pclass interaction — missing feature
16  0.886128    keep       Sex * Pclass interaction (fixed)  ← +0.001
17  0.000000    crash      FamilySize grouping — missing features
18  0.887030    keep       FamilySize grouping (fixed)       ← +0.001
19  0.887645    keep       Sex * FamilyGroup interaction
20  0.881020    discarded  Ticket prefix extraction
21  0.882601    discarded  Fare per person
22  0.887645    discarded  Log transformation of Fare
23  0.888350    keep       XGBoost max_depth=4               ← +0.001
24  0.872922    discarded  5-bin Fare grouping
25  0.886122    discarded  HasCabin indicator
26  0.887881    discarded  Switch to LightGBM
27  0.888350    discarded  Sex * IsElderly interaction
28  0.888630    keep       IsMaster binary indicator
29  0.891664    keep       Pclass * FamilyGroup interaction   ← +0.003
30  0.000000    crash      LogFarePerPerson — missing feature
31  0.889003    discarded  LogFarePerPerson (fixed)
32  0.000000    crash      Title * Pclass — missing feature
```

**AUC 0.867 → 0.892** across 32 experiments.

---

## What the agent discovered on its own

**Title only helps with XGBoost, not Random Forest.**  
Experiments 1 and 7 discarded Title-based features. Only in experiment 11,
after switching to XGBoost in experiment 9, did Title produce a real improvement.
The agent had to find the interaction between feature and model by itself.

**Interaction features matter more than raw features.**  
`Sex × Pclass`, `Pclass × FamilyGroup` — the biggest single improvements came
from interactions, not from new raw features. The agent explored this space
systematically after exhausting simple features.

**Crashes are the agent mapping the edges of known space.**  
Experiments 15, 17, 30, 32 crashed because the agent tried to combine
features from previous experiments that had since been discarded.
This is not a failure — it's the agent reasoning about what *should* work
based on its history, and finding the boundaries.

**The git ratchet never lost a good result.**  
Despite 4 crashes and 17 discarded experiments, the AUC never dropped
below the best value found at any point. Every improvement was locked in.

---

## How to run it

```bash
# 1. Get the data
# Download train.csv from https://www.kaggle.com/c/titanic/data
# Place it in data/train.csv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the environment
python data.py
# Should print: data.py OK

# 4. Check the current best model
python solve.py
# Should print: val_metric: 0.891664

# 5. Point your agent at program.md
# "Read program.md and follow the instructions exactly."
```

---

## File structure

```
titanic/
├── program.md      ← instructions for the agent
├── data.py         ← locked evaluator (do not modify)
├── solve.py        ← current best model (the agent improves this)
├── results.tsv     ← full experiment history
├── requirements.txt
└── data/
    └── train.csv   ← download from Kaggle
```

---

## The solve.py the agent built

The final `solve.py` contains features the agent invented through iteration:
- `Title` extracted from passenger names (Mr, Mrs, Miss, Master, Rare)
- `FamilyGroup` — Solo (1), Small (2-4), Large (5+)
- `IsMaster` — binary flag for male children
- `Pclass × FamilyGroup` — the last significant improvement (+0.003)
- `Sex × Pclass`, `Sex × FamilyGroup` — interaction features

None of these were suggested in `program.md`.
The agent found them by exploring the feature space from first principles.