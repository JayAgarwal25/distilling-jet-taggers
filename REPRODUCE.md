# Reproducing the paper

Every table in the paper is built by `scripts/make_tables.py` from trained models. You can
retrain those models yourself, or check that our released models give the paper's numbers.

## What to expect

Retraining gives the same results in the scientific sense. Each table cell lands within the
spread of its runs, and every conclusion of the paper holds. The digits differ from ours because
training on a GPU is not bit-deterministic.

Evaluation is deterministic: jetkd evaluates with torch's deterministic algorithms, and the paper's
numbers come from such an evaluation of the released checkpoints. On an RTX 6000 Ada with the versions
in `requirements.txt`, the checkpoints give the paper's numbers to the printed digits. Other GPUs and
CPUs use other kernels, so their logits differ by float32 round-off. For the Transformers this is about
1e-4. L-GATr-slim amplifies round-off in its layers, and ParT and MIParT in the masses of nearly
collinear particle pairs, so a single logit can move by more than one unit. A background rejection can
then change in its last printed digit.

## Setup

```bash
pip install -e ".[xformers]"
python scripts/get_data.py          # data/toptagging_full.npz
python scripts/get_checkpoints.py   # runs/, data/teachers/ and data/external_teachers/, checked by sha256
```

## Tables

| Paper | Table name | Runs |
|---|---|---|
| Table 1, teachers | `teachers` | 18 teacher runs |
| Table 2, students without a teacher | `students` | 15 |
| Tables 3 and 10, students and teachers | `matrix_auc`, `matrix_rej` | 105 |
| Table 4, width study | `size_short` | 36 |
| Tables 6 and 7, architectures and recipes | `architectures`, `recipes` | none, from the code |
| Table 8, teacher ensembles | `teacher_ensembles` | 18 teacher runs |
| Table 9, students of the L-GATr-slim ensemble | `ens_students` | 45 |
| Table 11, width study with L-GATr and OmniLearned-S | `size_other` | 36 |
| Tables 12 and 13, depth study | `depth_main`, `depth_other` | 32, 32 |

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

## Operations per jet

```bash
python scripts/count_flops.py
```

counts the floating-point operations per jet of our models on the first 300 test jets. The counts
do not depend on the hardware and equal ours exactly.
