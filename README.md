<div align="center">

## How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?

</div>

This repository contains the code for the paper *How Far Can Foundation Models Be Distilled for
Efficient Jet Tagging?* by Jay Agarwal, Siddharth Khare and Dhruv Kumar. The package `jetkd` trains
every model of the paper on the top tagging dataset, from scratch or distilled from a teacher's
logits, and distils any teacher you provide into any of its students. We release the weights of all
trained models and the logits of all teachers with the
[v1.0.0 release](https://github.com/JayAgarwal25/distilling-jet-taggers/releases/tag/v1.0.0).

This README covers installation and the released models, how to rebuild the paper's tables and figures
from them and how to train the models yourself. [REPRODUCE.md](REPRODUCE.md) explains how to retrain every model of the paper.
If you work on the code with a coding agent, [AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md) gives it
the layout, commands and rules. You can adapt them to your own style.
For any queries, see [Contact](#contact) below.

## Installation

Create a fresh environment with Python 3.10 (the paper's version; newer versions also work), then
install torch for your hardware and the package:

```bash
conda create -n jetkd python=3.10 && conda activate jetkd   # or: python3.10 -m venv .venv && source .venv/bin/activate
git clone https://github.com/JayAgarwal25/distilling-jet-taggers.git && cd distilling-jet-taggers
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU; for CPU only: .../whl/cpu
pip install -e .
pip install -e ".[xformers]"   # optional, GPU only: the attention kernels used for training
```

xformers is optional because each release is built for one torch version: install it after torch, on a
GPU machine. Without it the code builds the attention masks itself. `requirements.txt` lists the exact
versions behind the paper (Python 3.10, torch 2.11, CUDA 12.8). To run the tests,
`pip install -e ".[dev]"` and `pytest`.

## Data

```bash
python scripts/get_data.py    # data/toptagging_full.npz, 1.5 GB
```

This is the top tagging reference sample of Kasieczka et al.
([Zenodo 2603256](https://zenodo.org/records/2603256)) in the single-file layout of L-GATr and
LLoCa: 1,211,000 training, 403,000 validation and 404,000 test jets.

## Released models

```bash
python scripts/get_checkpoints.py                  # all of it, 840 MB
python scripts/get_checkpoints.py teacher_logits   # only the teacher logits, 150 MB
```

| Archive | Unpacks to | Contents |
|---|---|---|
| `students.tar` | `runs/<id>/best.pt` | the 248 student runs of the paper |
| `teachers.tar` | `runs/<id>/best.pt`, `data/external_teachers/` | the 12 teachers we trained (ParT, MIParT, L-GATr and L-GATr-slim, three runs each) and the fine-tuned pretrained ParT and OmniLearned-S, three runs each |
| `teacher_logits.tar` | `data/teachers/` | the logits of every teacher run on the training, validation and test jets |

Each student checkpoint holds the weights and the full training state (optimizer, scheduler and
history). Run ids are
`<model>__scratch__s<seed>` or `<model>__kd_<teacher>_t<teacher seed>__s<seed>`;
[`configs/RUNS.md`](configs/RUNS.md) lists every model, recipe and run. To tag jets with one of them:

```bash
python examples/tag_jets.py --model transformer_200k \
    --checkpoint runs/transformer_200k__kd_part_pretrained_t1001__s1001/best.pt
```

## Distilling your own teacher

Any model can be a teacher. Save its logits and labels per split, in the row order of
`data/toptagging_full.npz`, then

```bash
python scripts/import_logits.py --name my_teacher --format logits --file my_logits_{split}.npz
python scripts/train.py --model lgatr_slim_200k --teacher my_teacher --teacher-seed 1001
```

The import checks that every row matches the dataset. [`examples/distill_your_own.py`](examples/distill_your_own.py)
trains a student of your own, and [REPRODUCE.md](REPRODUCE.md#step-3-pretrained-part-and-omnilearned-s) shows
how we did this for pretrained ParT and OmniLearned-S.

We would love to see the results you get with your own set of runs. Further, if you are interested in distilling the models for any other downstream task apart from jet tagging, please contact us and we can work it out together.

## Reproducing the paper

[REPRODUCE.md](REPRODUCE.md) explains how to reproduce our work from scratch by training the models yourself. This section covers how to reproduce the paper with the released model weights.

With the data and the released models in place, `scripts/reproduce.py` evaluates every model behind a
table on the test set and rebuilds the table:

```bash
python scripts/reproduce.py students                                     # one table
python scripts/reproduce.py all --benchmark configs/benchmark.json       # every table and figure
```

It prints one line per model (`[3/15] <run id>: AUC 0.98...`), writes the test predictions to
`reproduced/runs/` and the tables to `reproduced/tables/<table>.tex`; `all` also writes every figure
to `reproduced/figures/`. Each model takes about 40 s on a GPU and `all` about three hours on one
RTX 6000 Ada. Predictions already in `reproduced/` are reused, so an interrupted call continues where
it stopped. To compare with the paper, download the paper's source from arXiv, unpack it and pass
its `tables/` folder: `--paper <source>/tables` prints the difference to each table or reports it
identical.

| Paper | Table name | Models |
|---|---|---|
| Table 1, teachers | `teachers` | 18 teacher runs |
| Table 2, students without a teacher | `students` | 15 |
| Tables 3 and 11, students and teachers | `matrix_auc`, `matrix_rej` | 105 |
| Table 4, width study | `size_short` | 36 |
| Tables 6 and 7, architectures and recipes | `architectures`, `recipes` | none, from the code |
| Table 8, teachers and LorentzNet against their published performance | `published` | 18 teacher runs and 3 |
| Table 9, teacher ensembles | `teacher_ensembles` | 18 teacher runs |
| Table 10, students of the L-GATr-slim ensemble | `ens_students` | 45 |
| Table 12, width study with L-GATr and OmniLearned-S | `size_other` | 36 |
| Tables 13 and 14, depth study | `depth_main`, `depth_other` | 32, 32 |
| Table 15, background jets accepted with and without distillation | `accepted` | 105 and the 12 teacher runs they distil from |
| Tables 5 and 18 to 23, performance and inference cost | `cost`, `cost_<teacher>`, `cost_scratch` | as the matrix and width study, and all teacher runs |
| Tables 16 and 17, execution settings and cost on 50-particle jets | `settings`, `cost_rnd50` | none |

The cost tables need timings: `--benchmark configs/benchmark.json` uses ours, and
[REPRODUCE.md](REPRODUCE.md#timing-the-models) shows how to measure your own.

On an RTX 6000 Ada with the versions in `requirements.txt`, the released checkpoints give the paper's
numbers to the printed digits: evaluation uses torch's deterministic algorithms. Other GPUs and CPUs
use other kernels, so their logits differ by float32 round-off, and a background rejection can change
in its last printed digit. On a CPU pass `--device cpu --batch-size 100` (without xformers, attention
over a batch of jets builds a dense mask that grows with the square of the particles in the batch);
`--limit 2000` evaluates only the first test jets, for a quick check.

## Training

Every run of the paper has an id in [`configs/runs.yaml`](configs/runs.yaml):

```bash
python scripts/train.py --run transformer_200k__scratch__s1001
python scripts/train.py --run part_200k__kd_lgatr_t1002__s1002    # needs data/teachers/lgatr_s1002_train.npz
```

A distilled run needs its teacher's logits on the training jets in `data/teachers/`. Download ours with
`python scripts/get_checkpoints.py teacher_logits`, or train the teacher and write its logits yourself
([REPRODUCE.md](REPRODUCE.md#step-2-train-the-teachers-and-write-their-logits)). Each model trains under the
recipe its family was published with ([`jetkd/recipes.py`](src/jetkd/recipes.py)). The run prints its
progress and appends it to `runs/<id>/train.log`. It starts with the model, data and recipe. After steps
1, 10, 100 and 1,000 and at every validation it gives the elapsed time and an estimate of the training
time, and at every validation the validation loss and AUC. It ends with the validation and test metrics
of the best checkpoint. `runs/<id>/` then holds

| File | Contents |
|---|---|
| `train.log` | the printed progress |
| `config.json` | the full configuration, parameter count and versions |
| `history.json` | training and validation metrics at every validation |
| `best.pt`, `last.pt` | the checkpoint with the lowest validation loss and the latest one |
| `predictions_val.npz`, `predictions_test.npz` | logits, labels and row indices of the best checkpoint |
| `results.json` | validation and test metrics of the best checkpoint |

A run that stops resumes from `last.pt` when started again. For a quick check on a CPU, add
`--device cpu --workers 0 --limit 2048 --iterations 20`.

## Acknowledgements

This work builds on public code and wouldn't be possible without the HEPML community support. The networks of [lloca](https://github.com/heidelberg-hepml/lloca)
(Transformer) and [lgatr](https://github.com/heidelberg-hepml/lgatr) (L-GATr, L-GATr-slim) are
used as pip dependencies. Our Transformer and L-GATr wrappers and their input pipeline
(`models/transformer.py`, `models/lgatr.py`, `models/inputs.py`) follow the top tagging
wrappers of [tagging-guide](https://github.com/heidelberg-hepml/tagging-guide). The particle
features, the sparse embedding, the Ranger optimizer and the ParT, MIParT and LorentzNet
recipes follow [lloca-experiments](https://github.com/heidelberg-hepml/lloca-experiments)
(`features.py`, `models/inputs.py`, `ranger.py`, `recipes.py`). `models/mipart.py` is adapted
from [MIParT](https://github.com/USST-HEP/MIParT), `models/lorentznet.py` from
[LorentzNet](https://github.com/sdogsq/LorentzNet-release) (both through lloca-experiments)
and `models/particle_transformer.py` from lloca 1.3.5's version of ParT. The size-class recipes
and the width ladders are those of the Economical Jet Taggers
([paper](https://arxiv.org/abs/2512.17011), [code](https://github.com/heidelberg-hepml/tagger-quantization)). The pretrained ParT teacher comes
from [particle_transformer](https://github.com/jet-universe/particle_transformer) with
[weaver-core](https://github.com/hqucms/weaver-core), and OmniLearned-S from
[OmniLearned](https://github.com/ViniciusMikuni/OmniLearned), whose package also builds it for
timing (`models/omnilearned.py`, in the setting of tagging-guide). `models/deepsets.py` adapts the
Deep Sets students of [fm-distillation](https://github.com/stanford-ai4physics/fm-distillation),
which we use only to count operations and time them. We thank their authors.

## Contact

We want to give back to the HEPML community as much as we have learned from it, so we release everything we
have: the code, the model weights and the teacher logits. If anything is unclear or you would like us to add
something, please get in touch. Please report problems with the code as a GitHub issue. You can also
reach us at jay25official@gmail.com.

## Citation

If you find this code useful in your research, please cite our paper

```bibtex
@article{agarwal2026distilling,
  title  = {How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?},
  author = {Agarwal, Jay and Khare, Siddharth and Kumar, Dhruv},
  year   = {2026},
}
```
