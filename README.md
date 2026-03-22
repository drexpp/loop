# loop

**The Karpathy Loop for classical Machine Learning.**  
Three files. Any dataset. Runs overnight on a laptop.

-----

## What just happened

On March 6, 2026, Andrej Karpathy released [autoresearch](https://github.com/karpathy/autoresearch):
an autonomous agent that runs 100+ ML experiments overnight on a GPU, improving a language
model’s training code without any human involvement.

It hit 25,000 GitHub stars in five days. Fortune called it “The Karpathy Loop.”
Shopify’s CEO pointed it at their templating engine and got 53% faster render times
from 93 automated commits.

Every fork that followed adapted it to different hardware — Apple Silicon, Windows RTX —
but kept the same problem: training language models, which requires GPUs, hours of compute,
and specialized knowledge.

**loop asks a different question: what if you pointed the same loop at a CSV file?**

-----

## The result

12 experiments. No human intervention. No GPU. A laptop.

```
id  val_metric  status     description
0   0.866731    baseline   Random Forest, basic features
1   0.864204    discarded  Extract Title from Name
2   0.864469    discarded  Pclass * Sex interaction
3   0.874242    keep       RF max_depth=5              ← +0.008
4   0.868534    discarded  Titles + max_depth=5
5   0.875129    keep       Age binning                ← +0.001
6   0.872022    discarded  Fare binning (qcut)
7   0.871637    discarded  Simplified title grouping
8   0.870072    discarded  Sex * AgeBin interaction
9   0.877818    keep       Switch to XGBoost          ← +0.003
10  0.881658    keep       FamilySize and IsAlone     ← +0.004
11  0.884463    keep       Simplified Titles + XGBoost ← +0.003
12  0.884791    keep       XGBoost hyperparameter tuning
```

AUC 0.867 → 0.885 in 12 experiments. The agent discovered on its own that:

- Title only helps when combined with XGBoost, not Random Forest
- FamilySize matters more than SibSp and Parch separately
- Simpler title grouping outperforms fine-grained title mapping

None of this was told to the agent. It found it.

-----

## Why this is different from AutoML

AutoML searches over a predefined catalog: model × hyperparameter grid.
It cannot invent features. It cannot read a paper and apply it.
It cannot reason about why something worked.

The Karpathy Loop searches over an open space: the agent writes arbitrary Python code,
applies techniques from its pretraining (every ML paper ever written),
and reasons from the git log about what to try next.

The difference is the same as the difference between a search engine and a researcher.

-----

## How it works

Three files. Identical structure to Karpathy’s original.

```
your_problem/
├── program.md    ← instructions for the agent (you write this once)
├── data.py       ← locked evaluator: CV + metric (you write this once)
├── solve.py      ← the only file the agent edits
└── data/
    └── train.csv
```

**`data.py`** is sacred. It defines how success is measured.
The agent cannot touch it. This is the equivalent of Karpathy’s `prepare.py`.
It implements a fixed cross-validation loop that returns a single scalar: higher is better.

**`solve.py`** is the only file the agent modifies.
It starts as the simplest possible baseline. The agent improves it iteratively.
Every change is one hypothesis. Every hypothesis is tested. Kept or discarded. Logged.

**`program.md`** tells the agent the rules: read the git log, propose one change,
run the experiment, keep or revert, never stop.

The agent reads its own history. It never repeats an experiment.
It never loses a good result. The git ratchet is the memory.

-----

## Use it now

**Requirements:** Python 3.10+, an AI coding agent (Claude Code, Antigravity, Codex),
and a dataset.

```bash
git clone https://github.com/you/mlloop
cd mlloop/titanic

pip install -r requirements.txt

# Place your train.csv in data/
# Then point your agent at program.md:
# "Read program.md and follow the instructions exactly."
```

The agent takes over. You go to sleep.

-----

## Adapt it to your problem

To use loop on any supervised ML problem, change three things in `data.py`:

1. **Where are the data?** → `TRAIN_FILE`
1. **What are you predicting?** → `TARGET_COL`
1. **How do you measure success?** → the metric inside `evaluate()`

That’s it. The loop, the git ratchet, the agent instructions — all stay the same.

|Problem type         |Metric    |CV strategy    |
|---------------------|----------|---------------|
|Binary classification|AUC-ROC ↑ |StratifiedKFold|
|Multi-class          |F1-macro ↑|StratifiedKFold|
|Regression           |−RMSE ↑   |KFold          |
|Time series          |−MAE ↑    |TimeSeriesSplit|

-----

## The pattern generalizes further

The loop works on any domain where you can define a computable oracle.
These are unexplored but natural extensions of the same three-file structure:

|Domain                  |Editable file                |Oracle                 |
|------------------------|-----------------------------|-----------------------|
|Classical ML (this repo)|`solve.py` — pipeline        |CV AUC / RMSE          |
|LLM training            |`train.py` — architecture    |val_bpb                |
|SQL optimization        |`query.sql` — query + indexes|execution time         |
|Algo trading            |`strategy.py` — trading rules|Sharpe ratio           |
|Prompt engineering      |`prompt.md` — system prompt  |test suite accuracy    |
|Scientific models       |`model.py` — ODEs            |RMSE vs historical data|
|Infrastructure config   |`config.yaml` — server config|throughput benchmark   |

The pattern is not about ML. It’s about the loop.
Define the arena. Lock the oracle. Let the agent run.

-----

## The philosophical shift

<cite>Karpathy wrote: “One day, frontier AI research used to be done by meat computers
in between eating, sleeping, having other fun. That era is long gone.”</cite>

The role of the human is shifting from **experimenter** to **experimental designer**.
You don’t run the experiments. You design the arena in which they run.

mlloop makes that arena as small as possible for ML:
three files, a CSV, and a clear metric.

The bottleneck was never compute. It was always iteration speed.
A laptop running 60 experiments per hour beats a researcher running 3 experiments per day.

-----

## Examples

- **[Titanic](/titanic/)** — binary classification, 891 rows. AUC 0.867 → 0.885 in 12 experiments.  
  *(more examples coming)*

-----

## Contributing

The most valuable contributions are new `(data.py, solve.py, program.md)` triples
for different domains. If you run the loop on a new dataset and get results,
open a PR with the three files and your `results.tsv`.

The results are the documentation.

-----

## License

MIT

-----

*Built on the insight of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
The loop is his. The arena is yours.*
