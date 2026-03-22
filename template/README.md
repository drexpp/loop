# template — Start here for a new problem

Copy this folder. Edit three files. Point your agent at `program.md`.

---

## Setup in 5 minutes

### 1. Copy this folder

```bash
cp -r template/ my_problem/
cd my_problem/
```

### 2. Add your data

```bash
mkdir data/
cp /path/to/your/train.csv data/train.csv
```

### 3. Edit `data.py` — three lines

Open `data.py` and change:

```python
TRAIN_FILE = DATA_DIR / "train.csv"   # ← your filename
TARGET_COL = "target"                  # ← your target column
```

Then choose the right metric (binary classification, regression, etc.)
by following the comments inside `evaluate()`.

Verify it works:
```bash
python data.py
# Should print: data.py OK
```

### 4. Edit `solve.py` — one line

Change the pipeline function to match your problem type.
The comments show the options: classification, multi-class, regression.

Run the baseline:
```bash
python solve.py
# Should print: val_metric: X.XXXXXX
```

### 5. Edit `program.md` — one section

Fill in the "Problem context" section with your dataset description.
Leave everything else exactly as it is.

### 6. Point your agent at `program.md`

Open the folder in Claude Code, Antigravity, or Cursor.
Give the agent one instruction:

```
Read program.md and follow the instructions exactly.
```

The agent takes over. You go to sleep.

---

## What changes between problems

| File | What to change | What stays the same |
|---|---|---|
| `data.py` | TRAIN_FILE, TARGET_COL, metric function | evaluate() contract, RANDOM_SEED, N_FOLDS |
| `solve.py` | pipeline() baseline model | val_metric print format, function signature |
| `program.md` | Problem context section | Loop structure, constraints, NEVER STOP |

That's it. The loop is the same for every problem.

---

## Choosing your metric

| Problem | What pipeline() returns | Metric |
|---|---|---|
| Binary classification | `predict_proba(X_val)[:, 1]` | AUC-ROC |
| Multi-class | `predict(X_val)` | F1-macro |
| Regression | `predict(X_val)` | −RMSE |
| Time series | `predict(X_val)` | −MAE |

The metric must always be **higher = better**.
For regression, return `-rmse` (negative) so the ratchet works correctly.

---

## Example: adapting to a churn prediction problem

```python
# data.py
TARGET_COL = "churned"          # your target column
TRAIN_FILE = DATA_DIR / "customers.csv"
# metric: roc_auc_score (already the default for binary classification)
```

```python
# solve.py — baseline for churn
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_tr, y_train)
return model.predict_proba(X_v)[:, 1]
```

```markdown
# program.md — problem context section
**Dataset**: Customer churn data — 50,000 rows, 20 columns
**Target**: `churned` (1 = churned, 0 = stayed)
**Key columns**: tenure, monthly_charges, contract_type, ...
```

---

## When the loop finishes

The agent will have committed multiple improvements to `solve.py`.
Your best model is always the latest committed version.

```bash
git log --oneline          # see all experiments
python solve.py            # run the best model
cat results.tsv            # full experiment history
```