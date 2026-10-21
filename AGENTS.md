# Notes for coding agents

This repository reproduces the paper *How Far Can Foundation Models Be Distilled for Efficient Jet
Tagging?* and lets others distil their own teachers and students. Two things must keep working
through any change: the released checkpoints load, and the paper's tables come out the same.

## Where things are

| To change | Look at |
|---|---|
| a model, or add one | `src/jetkd/models/` (one file per family), registered in `MODELS` in `models/__init__.py` |
| how a model trains | `src/jetkd/recipes.py` (`RECIPES`, `recipe_for`) and `src/jetkd/train.py` |
| the distillation loss | `src/jetkd/distillation.py` |
| the inputs | `src/jetkd/data.py` (dataset, `JetBatch`), `features.py`, `models/inputs.py` |
| the metrics | `src/jetkd/metrics.py` (rejection by linear interpolation of the ROC curve) |
| teacher logits | `src/jetkd/teachers.py`, `scripts/teacher_logits.py`, `scripts/import_logits.py` |
| the paper's runs | `src/jetkd/study.py`, written to `configs/runs.yaml` |
| tables and figures | `scripts/summarize.py`, then `make_tables.py` and `make_figures.py`; `reproduce.py` runs all three |
| progress output | `src/jetkd/log.py` (`log`, `duration`), used by the trainer (also appended to `runs/<id>/train.log`), `model_logits(progress=True)` and the scripts |
| user documentation | `README.md` (install, released models, rebuilding the paper from them, training), `REPRODUCE.md` (retraining, timing) |

Run ids are `<model>__scratch__s<seed>` or `<model>__kd_<teacher>_t<teacher seed>__s<seed>`, for
example `part_200k__kd_lgatr_t1002__s1002`. The paper uses seeds 1001, 1002 and 1003. The id fixes
the model, teacher and seeds; the recipe follows from the model. Each run writes `runs/<id>/`
(`train.log`, `config.json`, `history.json`, `best.pt`, `last.pt`, predictions and `results.json`).

## Commands

On CPU, in seconds, with no dataset needed:

```bash
pip install -e ".[dev]"
ruff check src tests scripts
pytest
python -m jetkd.study --check && python -m jetkd.reference --check
```

A full training run needs a GPU and hours. For a quick check on CPU:

```bash
python scripts/train.py --run transformer_200k_L1__scratch__s1001 --device cpu --workers 0 --limit 2048 --iterations 20
```

## What must not change

- Module and parameter names in `models/`: released checkpoints load with `strict=True`.
  `tests/test_models.py` pins every parameter count, so a changed count is a changed architecture.
- `data.py`, `features.py`, `models/inputs.py`, `distillation.py` and `metrics.py` produce the paper's
  numbers. After a change there, rebuild a table from the released checkpoints with
  `python scripts/reproduce.py <table>` (GPU, see REPRODUCE.md) and check that it is unchanged.
- Evaluation runs inside `jetkd.teachers.deterministic()`. Without it, GPU evaluation of the
  Transformer, MIParT and LorentzNet differs between two calls and rej@0.3 moves in its last digits.
- Teacher logits are in the row order of `data/toptagging_full.npz`. Write them only with
  `scripts/teacher_logits.py` or `scripts/import_logits.py`, which check it.
- `configs/runs.yaml` is written by `python -m jetkd.study`; `configs/models.yaml`, `recipes.yaml` and
  `RUNS.md` by `python -m jetkd.reference`. Never edit them by hand. `runs.yaml` lists the paper's
  runs only: train new experiments with `--model`, not by adding runs.
- `lgatr` and `lloca` are pinned exactly in `pyproject.toml`, as the checkpoints were trained on them.
- Logging only prints: it must not draw random numbers, move tensors or change the order of operations,
  so that a run stays identical to one without it (`tests/test_train.py` checks an exact resume).
- When a command, flag, output file or table name changes, update `README.md` and `REPRODUCE.md`.

## Common tasks

- New student: a module that maps a `JetBatch` to one logit per jet, an entry in `MODELS`, a recipe in
  `RECIPES` reachable from `recipe_for`, its parameter count in `tests/test_models.py`, then
  `python -m jetkd.reference`. `examples/distill_your_own.py` does the same at run time, without
  editing the package.
- New teacher: save its logits per split, `scripts/import_logits.py --format logits`, then
  `scripts/train.py --model <student> --teacher <name>`.
- Tag jets with a checkpoint: `examples/tag_jets.py`.

## Pitfalls

- On CPU without xformers, attention over packed jets builds a dense mask that grows with the square
  of the particles in the batch. Keep CPU batches near 100 jets.
- Logits on another GPU or on CPU differ by float32 round-off; for L-GATr-slim, ParT and MIParT a single
  logit can move by more than one unit. Compare metrics across devices, not logits.
- The pretrained ParT and OmniLearned-S teachers are fine-tuned in their own code bases (REPRODUCE.md),
  not with `scripts/train.py`.
- `scripts/train.py` writes `runs/<id>/best.pt`, so training a run id whose released checkpoint is in
  `runs/` replaces it. Train elsewhere with `--out`.
- `scripts/reproduce.py` reuses test predictions already under its `--out` (default `reproduced/`).
  Rebuilding tables from other checkpoints needs a new `--out`, or it silently reports the old ones.
- `matplotlib` is a core dependency because `scripts/reproduce.py` imports `make_figures.py`.

## Style

Minimal code and one-line docstrings. Comment only where the code differs from its upstream source,
and say that the difference was checked numerically. ruff, line length 100.
