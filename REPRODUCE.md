# Reproducing the paper

Every table in the paper is built by `scripts/make_tables.py` from trained models. You can
retrain those models yourself, or check that our released models give the paper's numbers.

## What to expect

Retraining gives the same results in the scientific sense. Each table cell lands within the
spread of its runs, and every conclusion of the paper holds. The digits differ from ours because
training on a GPU is not bit-deterministic.

Our released checkpoints, evaluated on a GPU, give the paper's numbers to the printed digits.
On a CPU the Transformers agree with our GPU logits to about 1e-4. L-GATr-slim, ParT and MIParT
amplify float32 round-off, L-GATr-slim in its layers and ParT and MIParT in the masses of nearly
collinear particle pairs. Single logits can then move by more than one unit, so their metrics can
change in the last printed digit.

## Setup

```bash
pip install -e ".[xformers]"
python scripts/get_data.py          # data/toptagging_full.npz
python scripts/get_checkpoints.py   # runs/<id>/best.pt and data/teachers/, only for checking
```

## Tables

| Paper | Table name | Runs |
|---|---|---|
| Table 1, teachers | `teachers` | 18 teacher runs |
| Table 2, students without a teacher | `students` | 15 |
| Tables 3 and 4, students and teachers | `matrix_auc`, `matrix_rej` | 105 student runs and all teachers |
| Width study | `size` | 36 |
| Depth study | `depth` | 32 |
| Architectures and recipes | `architectures`, `recipes` | none, from the code |
| Appendix: gains, every matrix cell | `gains`, `complete` | as `matrix_auc` |
| Appendix: depth and width with every teacher | `depth_transformer`, `depth_slim`, `size_full` | 32, 32, 72 |

## 1. Retrain

```bash
python scripts/reproduce.py matrix_auc --print-jobs
```

prints every command behind a table, in order. First it trains the teachers and writes their
logits, then it trains the students. Run the commands on any number of GPUs. A run that stops
resumes from its last checkpoint. Pretrained ParT and OmniLearned-S are trained in their own
code bases; [`teachers/README.md`](teachers/README.md) gives the commands and how to import
their logits. The last command rebuilds the table from your models and compares it with ours:

```bash
python scripts/reproduce.py matrix_auc
```

It evaluates every `runs/<id>/best.pt` behind the table on the test set, writes the predictions
and the table under `reproduced/`, and prints the difference to `paper/tables/matrix_auc.tex`.

## 2. Check the released models

With the released checkpoints in `runs/`, the same command evaluates our models:

```bash
python scripts/reproduce.py matrix_auc
```

On a GPU this prints that the table is identical to the paper's. On a CPU pass
`--device cpu --batch-size 100`: without xformers, attention over a batch of jets builds a
dense mask whose size grows with the square of the number of particles in the batch.
`--limit 2000` evaluates only the first test jets, for a quick check.

The teacher metrics come from the teacher checkpoints, except for pretrained ParT and
OmniLearned-S, whose test logits are released in `data/teachers/`.
