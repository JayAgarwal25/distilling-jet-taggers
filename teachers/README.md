# Teachers

Distillation reads a teacher only through its logits, one file per split:
`data/teachers/<teacher>_s<seed>_<split>.npz`. We release these files and the teacher
checkpoints, so retraining a teacher is optional. This page shows how each was trained.
Seeds 1001, 1002 and 1003 are the three runs of each teacher. Students distill from the
run with their own seed for ParT, L-GATr and L-GATr-slim and from run 1001 for MIParT,
pretrained ParT and OmniLearned-S. Runs 1002 and 1003 of these three appear only in the
teacher table, so we release only their test logits.

## L-GATr, L-GATr-slim, ParT and MIParT

These train with jetkd under their published recipes (`jetkd/recipes.py`):

```bash
python scripts/train.py --run lgatr__scratch__s1001
python scripts/teacher_logits.py --model lgatr --seed 1001 --checkpoint runs/lgatr__scratch__s1001/best.pt
```

Replace `lgatr` by `lgatr_slim`, `part` or `mipart`. Our ParT and MIParT checkpoints were
trained with the training loop of
[lloca-experiments](https://github.com/heidelberg-hepml/lloca-experiments) under the same
recipes. Their weights load into jetkd's models, but the optimizer and scheduler state they
hold is that loop's, so `scripts/train.py` cannot resume them.

## Pretrained ParT

ParT pretrained on JetClass and fine-tuned on top tagging, from
[particle_transformer](https://github.com/jet-universe/particle_transformer) at commit
`2925bdb` with [weaver-core](https://github.com/hqucms/weaver-core) 0.4.17, unchanged:

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
python scripts/import_logits.py --name part_pretrained --seed 1001 --format weaver \
  --file training/TopLandscape/ParT-FineTune/RUN/predict_output/pred_{split}.root
```

Predictions are made in float32 (no `--use-amp`). Weaver's ParT keeps the 128 leading
constituents of a jet, so a few hundred of the largest jets differ slightly in pT from our
data; the import allows for that and checks every row otherwise.

## OmniLearned-S

[OmniLearned](https://github.com/ViniciusMikuni/OmniLearned) at commit `25fef9f` with
[`omnilearned.diff`](omnilearned.diff) applied. The patch lets `evaluate` run on any split
(`OMNI_EVAL_SPLIT`), adds gradient accumulation (used only for the batch-256 runs in the
appendix) and selects the gloo backend and a free port for single-GPU runs, neither of
which changes the results.

```bash
git apply omnilearned.diff
omnilearned dataloader --dataset top --folder DATA_DIR
omnilearned train -o ./checkpoints --save-tag top_s_run1 --dataset top --path DATA_DIR \
  --size small --fine-tune --pretrain-tag pretrain_s --epoch 10 --lr 5e-6 --lr-factor 5.0 \
  --wd 0.1 --warmup-epoch 1 --interaction
for split in train val test; do
  OMNI_EVAL_SPLIT=$split omnilearned evaluate -i ./checkpoints -o ./outputs \
    --save-tag top_s_run1 --dataset top --path DATA_DIR --size small --interaction
  mv outputs/outputs_top_s_run1_top_0.npz outputs/top_s_run1_${split}.npz
done
python scripts/import_logits.py --name omnilearned_s --seed 1001 --format omnilearned \
  --file outputs/top_s_run1_{split}.npz
```

For the batch-256 runs, add `--batch 256 --grad-accum 4` to `omnilearned train`.

## Any other model

Save its logits and labels for each split as `.npz` with keys `logits` and `labels`, in the
row order of `data/toptagging_full.npz`, then

```bash
python scripts/import_logits.py --name my_teacher --seed 1001 --format logits --file my_logits_{split}.npz
python scripts/train.py --model transformer_200k --teacher my_teacher --teacher-seed 1001
```
