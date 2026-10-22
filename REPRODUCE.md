# Retraining the paper

This guide retrains the models of the paper and rebuilds its tables and figures from them. To rebuild
them from the released models instead, see the [README](README.md#reproducing-the-paper).
All commands run from the repository root, in the environment of the [README](README.md#installation),
after `python scripts/get_data.py`.

## What to expect

Retraining gives the same results (in the scientific sense): each table cell lands within the spread of
its three runs and every conclusion of the paper holds. The digits differ from ours because training
on a GPU is not bit-deterministic. Evaluation is deterministic, so evaluating the same checkpoint twice
gives the same numbers.

## Overview

The paper's models are trained in four stages, each needing the one before:

1. **Teachers.** 12 are trained with jetkd (ParT, MIParT, L-GATr and L-GATr-slim, three seeds each).
   Pretrained ParT and OmniLearned-S are fine-tuned in their own code bases, three runs each.
2. **Teacher logits.** Each teacher's logits on the training, validation and test jets go to
   `data/teachers/`. Students read them; they never run the teacher.
3. **Students.** 248 runs, from scratch or distilled from one teacher's logits.
4. **Tables and figures.** `scripts/reproduce.py` evaluates every model behind a table and builds it.

The first two stages can be skipped: `python scripts/get_checkpoints.py teacher_logits` downloads the
logits of every teacher run, and the students then train against our teachers (go to step 4).

Train in a directory without the released checkpoints, or move `runs/` aside first: a training run
writes `runs/<id>/best.pt` and would replace the released one.

### Compute

Training time on one RTX 6000 Ada, from our runs (median per run; the ParT teacher was timed on an
80 GB GPU):

| Models | Runs | Hours per run | GPU hours |
|---|---:|---:|---:|
| L-GATr teacher | 3 | 16 | 48 |
| L-GATr-slim teacher | 3 | 9 | 27 |
| MIParT teacher | 3 | 6 | 18 |
| ParT teacher | 3 | 3 | 8 |
| Transformer student | 24 | 5 | 122 |
| L-GATr-slim student | 24 | 5 | 115 |
| ParT student | 24 | 1 | 26 |
| MIParT student | 24 | 3 | 65 |
| LorentzNet student | 24 | 10 | 250 |
| width study, Transformer (200k, 20k, 2k) | 36 | 4, 4, 7 | 172 |
| width study, L-GATr-slim (200k, 20k, 2k) | 36 | 9, 8, 12 | 332 |
| depth study, Transformer (0 to 8 blocks) | 28 | 1 to 4 | 75 |
| depth study, L-GATr-slim (1 to 10 blocks) | 28 | 4 to 14 | 250 |
| all | 260 | | about 1,500 |

The matrix of students and teachers (`matrix_auc`, `matrix_rej`) alone takes about 510 GPU hours.

## Step 1. List the commands for a table

```bash
python scripts/reproduce.py matrix_auc --print-jobs
```

prints every command behind the table, in order: each teacher's training and logit commands, then
every student, then the command that builds the table. A line starting with `#` names a teacher that is
fine-tuned outside this repository (step 3). The table names are listed in the
[README](README.md#reproducing-the-paper); `all` lists every run of the paper.

## Step 2. Train the teachers and write their logits

For each of the 12 jetkd teachers, train it and write its logits on all three splits:

```bash
python scripts/train.py --run lgatr__scratch__s1001
python scripts/teacher_logits.py --model lgatr --seed 1001 --checkpoint runs/lgatr__scratch__s1001/best.pt
```

The teachers are `part`, `mipart`, `lgatr` and `lgatr_slim`, with seeds 1001, 1002 and 1003. Only
`mipart` run 1001 has students; its other two runs appear only in the teacher tables, which need only
their test logits (`--splits test`, as `--print-jobs` shows). The L-GATr-slim ensemble needs no training:
once the three `lgatr_slim` runs have their logits,

```bash
python scripts/teacher_logits.py --model lgatr_slim_ens --seed 1001
```

writes the mean of their logits as the ensemble teacher.

## Step 3. Pretrained ParT and OmniLearned-S

These two teachers are fine-tuned in clones of their own code bases; their logits are then imported
into this repository. Our three runs of each differ only in the random seed. Below, `DATA_DIR` is any
directory for their datasets, `JETKD` this repository, and `RUN` the run directory the training creates.

**Pretrained ParT**, with [particle_transformer](https://github.com/jet-universe/particle_transformer)
at commit `2925bdb` and [weaver-core](https://github.com/hqucms/weaver-core) 0.4.17. In the
particle_transformer clone:

```bash
./get_datasets.py TopLandscape -d DATA_DIR
./train_TopLandscape.sh ParT-FineTune kin
for split in train val test; do
  weaver --predict --data-test DATA_DIR/TopLandscape/${split}_file.parquet \
    --data-config data/TopLandscape/top_kin.yaml \
    --network-config networks/example_ParticleTransformer_finetune.py \
    --model-prefix training/TopLandscape/ParT-FineTune/RUN/net_best_epoch_state.pt \
    --gpus 0 --batch-size 512 --predict-output pred_${split}.root
done
```

Then in this repository, with `PARTICLE_TRANSFORMER` the clone:

```bash
python scripts/import_logits.py --name part_pretrained --seed 1001 --format weaver \
  --file PARTICLE_TRANSFORMER/training/TopLandscape/ParT-FineTune/RUN/predict_output/pred_{split}.root
```

**OmniLearned-S**, with [OmniLearned](https://github.com/ViniciusMikuni/OmniLearned) at commit `25fef9f`
and [`scripts/omnilearned.diff`](scripts/omnilearned.diff) applied, which adds gradient accumulation and
evaluation on every split. In the OmniLearned clone:

```bash
git apply JETKD/scripts/omnilearned.diff
omnilearned dataloader --dataset top --folder DATA_DIR
omnilearned train -o ./checkpoints --save-tag top_s_run1 --dataset top --path DATA_DIR \
  --size small --fine-tune --pretrain-tag pretrain_s --epoch 10 --lr 5e-6 --lr-factor 5.0 \
  --wd 0.1 --warmup-epoch 1 --interaction --batch 256 --grad-accum 4
for split in train val test; do
  OMNI_EVAL_SPLIT=$split omnilearned evaluate -i ./checkpoints -o ./outputs \
    --save-tag top_s_run1 --dataset top --path DATA_DIR --size small --interaction
  mv outputs/outputs_top_s_run1_top_0.npz outputs/top_s_run1_${split}.npz
done
```

Then in this repository, with `OMNILEARNED` the clone:

```bash
python scripts/import_logits.py --name omnilearned_s --seed 1001 --format omnilearned \
  --file OMNILEARNED/outputs/top_s_run1_{split}.npz
```

For runs 1002 and 1003, train each teacher twice more (for OmniLearned-S with `--save-tag top_s_run2` and
`top_s_run3`) and import their logits with `--seed 1002` and `--seed 1003`. The import checks that every
row matches the dataset.

## Step 4. Train the students

Every student is one command, `python scripts/train.py --run <id>`, and needs its teacher's training
logits in `data/teachers/`. The runs are independent, so spread them over your GPUs, for example four:

```bash
python scripts/reproduce.py matrix_auc --print-jobs | grep '^python scripts/train.py' \
  | grep -v -E -- '--run (part|mipart|lgatr|lgatr_slim)__' > jobs.txt   # students only
for g in 0 1 2 3; do
  awk -v g=$g 'NR % 4 == g' jobs.txt | CUDA_VISIBLE_DEVICES=$g bash > gpu$g.out 2>&1 &
done
```

Each run prints its progress and writes it to `runs/<id>/train.log` (see the
[README](README.md#training)). A run that stops resumes from `runs/<id>/last.pt` when started again, so
rerunning the same commands continues where they stopped.

## Step 5. Build the tables and figures

```bash
python scripts/reproduce.py matrix_auc --out reproduced_retrained
python scripts/reproduce.py all --out reproduced_retrained --benchmark configs/benchmark.json
```

`reproduce.py` evaluates `runs/<id>/best.pt` of every model behind the table on the test set and writes
the predictions, the table and, for `all`, every figure under `--out`. Predictions already there are
reused, so give a new `--out` if you rebuilt tables from the released models before. With the paper's
[arXiv source](https://arxiv.org/src/2610.29439) unpacked, `--paper <source>/tables` prints the difference to each of our tables.

## Timing the models

The inference cost tables and figures come from timings on your own hardware:

```bash
pip install -e ".[benchmark]"
python scripts/count_flops.py                     # results/flops.json
python scripts/benchmark.py --device cuda         # results/benchmark.json
python scripts/benchmark.py --device cpu --core 2
```

`count_flops.py` counts the floating-point operations per jet of the 18 models of the cost tables on
the first 300 test jets. The counts do not depend on the hardware and equal ours exactly.

`benchmark.py` times the same models: the GPU with a batch of 512 jets, the CPU with one jet on one core.
Each measurement runs in a fresh process with 100 warm-up and 100 timed calls of the whole forward pass,
from four-momenta in CPU memory to logits, in full 32-bit precision. Every model runs packed or padded
where it accepts both, eager and compiled (the network compiled, its inputs built eagerly), on test jets
and on random 50-particle jets, three times. The fastest setting by the median of its three runs is
reported. A full pass takes a few hours per device. Timings depend on the GPU, the CPU and the software
versions: orderings and rough ratios carry over to other machines, the numbers themselves do not.

`reproduce.py` builds the cost tables from `results/benchmark.json`. Our own measurements, on one
RTX 6000 Ada and one Xeon Platinum 8352Y core with the versions in `requirements.txt`, are in
`configs/benchmark.json`; with `--benchmark configs/benchmark.json` the tables come out as printed in the
paper. Our CPU timings are those of `benchmark.py --device cpu --core 10`. On the GPU, the setting was
chosen by one run of every setting and then timed three times. Pretrained ParT is
timed in the architecture of our ParT teacher and OmniLearned-S in that of its pretrained model, both
with random weights.
