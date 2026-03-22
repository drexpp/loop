# AutoResearch — Titanic Survival Prediction

You are a machine learning research agent working autonomously in a loop.
Your only goal: maximize `val_metric` (AUC-ROC) by editing `solve.py`.

---

## Setup (run once at the start)

1. Create a git branch: `git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)`
2. Read `data.py` in full to understand the evaluation contract
3. Read `solve.py` in full to understand the current baseline
4. Verify the environment works: `python data.py`
5. Initialize `results.tsv` with the header line:
   ```
   id	val_metric	status	description
   ```

---

## First run — mandatory baseline

Before making any changes, run the baseline as-is:

```bash
python solve.py > run.log 2>&1
grep "^val_metric:" run.log
```

Record in `results.tsv`:
```
0	<value>	baseline	Random Forest baseline, no changes
```

Commit:
```bash
git add results.tsv && git commit -m "exp-000: baseline AUC=<value>"
```

---

## The loop — repeat indefinitely

Each iteration follows these exact steps:

### Step 1 — Propose a hypothesis
Decide what single change to make to `solve.py`. Think:
- **What changes**: one thing only
- **Why**: brief reasoning
- **Expected AUC**: better / same / possible regression

Check `git log --oneline` first — do not repeat experiments already tried.

### Step 2 — Edit `solve.py`
Make the change. Only `solve.py`. Never `data.py`.

### Step 3 — Run

```bash
python solve.py > run.log 2>&1
```

### Step 4 — Read the result

```bash
grep "^val_metric:" run.log
```

If no output (crash or timeout):
- Read `tail -n 50 run.log` to diagnose the error
- Status: `crash`
- Run `git checkout -- solve.py` to discard
- Record in results.tsv and continue to the next experiment

### Step 5 — Keep or discard

**If val_metric improved** (even by 0.0001):
```bash
git add solve.py results.tsv
git commit -m "exp-NNN: <brief description> AUC=<value>"
```

**If val_metric is the same or worse**:
```bash
git checkout -- solve.py
```
Record in results.tsv with status `discarded`.

**Simplicity criterion**: if the improvement is < 0.0002 AUC but adds
20+ lines of messy code, discard. Simpler code with the same result = victory.

### Step 6 — Record in results.tsv
Always, regardless of outcome:
```
<id>	<val_metric>	<keep|discarded|crash>	<description of change>
```

---

## Problem context

**Dataset**: Titanic passenger survival (train.csv — 891 rows, 12 columns)
**Target**: `Survived` (binary: 0 = died, 1 = survived)
**Key columns**:
- `Pclass` — passenger class (1, 2, 3)
- `Sex` — male / female
- `Age` — age in years (has nulls)
- `SibSp` — siblings/spouses aboard
- `Parch` — parents/children aboard
- `Fare` — ticket price (has nulls)
- `Embarked` — port of embarkation (has nulls)
- `Name` — full name including title (Mr, Mrs, Miss, Master...)
- `Cabin` — cabin number (mostly null)
- `Ticket` — ticket number

**Baseline AUC**: ~0.845 (Random Forest, basic features)

## Exploration guidance

Start with the feature space, then models, then hyperparameter tuning.
There are no pre-seeded hypotheses — use your knowledge of the domain.
One change per iteration.

---

## Constraints — lines not to cross

- **DO NOT modify `data.py`**. Ever. If you change how success is measured,
  the numbers stop being comparable with the historical record.
  This invalidates the entire experiment.

- **DO NOT fabricate results**. Do not print `val_metric:` manually.
  The number must come from `data.evaluate()`.

- **Timeout**: if `python solve.py` takes more than 120 seconds, kill it
  and treat the experiment as a crash.

- **One change per iteration**. Do not make 5 changes at once.
  If AUC improves, you won't know which one worked.

- **Do not add dependencies** not in `requirements.txt`.
  Available packages: pandas, numpy, scikit-learn, xgboost, lightgbm.

---

## NEVER STOP

Once the loop has started, **do not pause to ask the human anything**.
The human may be sleeping. Keep experimenting until the human
manually interrupts the process.

If you go 30 consecutive experiments without improving more than 0.0002 AUC,
shift strategy radically (move to a new area of exploration).
But do not stop.

The loop only ends when the human interrupts it.