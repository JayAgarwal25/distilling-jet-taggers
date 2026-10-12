# Notes for coding agents

## Layout

- `src/jetkd/`: data, features, models, distillation loss, recipes, trainer, teacher logits,
  summary, the run list (`study.py`) and the reference files (`reference.py`).
- `scripts/`: command-line entry points (`get_data`, `train`, `teacher_logits`,
  `import_logits`, `summarize`).
- `configs/`, `docs/RUNS.md`: generated, see below. `teachers/`: how the teachers were trained.

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

## Rules

- `configs/runs.yaml` is written by `python -m jetkd.study`; `configs/models.yaml`,
  `configs/recipes.yaml` and `docs/RUNS.md` by `python -m jetkd.reference`. Never edit them by hand.
- A run id fixes the model, teacher, teacher seed and seed; the recipe follows from the model.
- Released checkpoints load with `strict=True`, so module and parameter names in
  `jetkd/models/` must not change. `tests/test_models.py` pins every parameter count.
- Teacher logits must be in the dataset's row order; create them with
  `scripts/teacher_logits.py` or `scripts/import_logits.py`, which check it.
- Paper numbers come from `jetkd.metrics` (rejection by linear interpolation of the ROC curve)
  on the test predictions in `runs/<id>/predictions_test.npz`.
