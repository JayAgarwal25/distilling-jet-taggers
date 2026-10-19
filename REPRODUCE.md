# Reproducing the paper

Every table and figure in the paper is built from the test predictions of trained models, by
`scripts/make_tables.py` and `scripts/make_figures.py`. You can retrain those models yourself, or check
that our released models give the paper's numbers. The inference cost tables and figures come from
`scripts/benchmark.py` on your own hardware (see below).

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
pip install -e ".[xformers,paper]"
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
| Table 14, background jets accepted with and without distillation | `accepted` | 105 and the 12 teacher runs they distil from |
| Tables 5 and 17 to 22, performance and inference cost | `cost`, `cost_<teacher>`, `cost_scratch` | as the matrix and width study, and all teacher runs |
| Tables 15 and 16, execution settings and cost on 50-particle jets | `settings`, `cost_rnd50` | none |

The cost tables also need timings (section 3).

`python scripts/reproduce.py all` evaluates every model once, rebuilds all of these tables and
writes every figure of Sections 4 to 6 and Appendices C to E to `reproduced/figures/`.
`scripts/summarize.py` writes the summary and the analyses of Section 6 (`jetkd/analysis.py`) as
JSON, and `scripts/make_figures.py` draws the figures from them.

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

## 3. Inference cost

```bash
pip install -e ".[benchmark]"
python scripts/count_flops.py                     # results/flops.json
python scripts/benchmark.py --device cuda         # results/benchmark.json
python scripts/benchmark.py --device cpu --core 2
```

`count_flops.py` counts the floating-point operations per jet of the 18 models of the cost tables
on the first 300 test jets. The counts do not depend on the hardware and equal ours exactly.

`benchmark.py` times the same models on your hardware: the GPU with a batch of 512 jets, the CPU
with one jet on one core. Each measurement runs in a fresh process with 100 warm-up and 100 timed
calls of the whole forward pass, from four-momenta in CPU memory to logits, in full 32-bit
precision. Every model runs packed or padded where it accepts both, eager and compiled, on test jets
and on random 50-particle jets, three times. The fastest setting by the median of its three runs
is reported. A full pass takes a few hours per device. Timings depend on the GPU, the CPU and the
software versions. Orderings and rough ratios carry over to other machines; the numbers themselves do not.

`reproduce.py` builds the cost tables from your timings. Our own measurements, on one RTX 6000 Ada
and one Xeon Platinum 8352Y core with the versions in `requirements.txt`, are in
`paper/benchmark.json`; with `--benchmark paper/benchmark.json` the tables come out as printed in
the paper. In our measurements the GPU setting was chosen by one run of every setting and then
timed three times, and FlexAttention was also tried for L-GATr and L-GATr-slim; it was never the
fastest, and `benchmark.py` leaves it out. Pretrained ParT is timed in the architecture of our ParT
teacher and OmniLearned-S in that of its pretrained model, both with random weights.
