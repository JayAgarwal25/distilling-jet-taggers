<div align="center">

## How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?

</div>

This repository contains the code for the paper *How Far Can Foundation Models Be Distilled for
Efficient Jet Tagging?* by Jay Agarwal, Siddharth Khare and Dhruv Kumar. The package `jetkd` trains
every model of the paper on the top tagging dataset, from scratch or distilled from a teacher's
logits, and distils any teacher you provide into any of its students. We release the weights of all
trained models and the logits of all teachers on [Zenodo](https://sandbox.zenodo.org/records/624034).

How to reproduce every table and figure of the paper, from the released weights or by retraining,
and how to time the models on your own hardware is explained in [REPRODUCE.md](REPRODUCE.md).
If you work on the code with a coding agent, [AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md) gives it the layout, commands and rules.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the CPU build
pip install -e .              # the package and its dependencies
pip install -e ".[xformers]"  # GPU attention kernels, as used for training
```

`requirements.txt` lists the exact versions behind the paper (Python 3.10, torch 2.11).

## Data

```bash
python scripts/get_data.py    # data/toptagging_full.npz, 1.5 GB, checked by sha256
```

This is the top tagging reference sample of Kasieczka et al.
([Zenodo 2603256](https://zenodo.org/records/2603256)) in the single-file layout of L-GATr and
LLoCa: 1,211,000 training, 403,000 validation and 404,000 test jets.

## Training

Every run of the paper has an id in [`configs/runs.yaml`](configs/runs.yaml):

```bash
python scripts/train.py --run transformer_200k__scratch__s1001
python scripts/train.py --run part_200k__kd_lgatr_t1002__s1002
```

A distilled run reads the teacher's logits from `data/teachers/`. Each model trains under the
recipe its family was published with ([`jetkd/recipes.py`](src/jetkd/recipes.py)). The run
writes its checkpoints, history, validation and test predictions and metrics to `runs/<id>/`.
[`configs/RUNS.md`](configs/RUNS.md) lists every model with its exact settings and parameter count,
every recipe and every run.

## Distilling your own teacher

Any model can be a teacher. Save its logits and labels per split, in the row order of
`data/toptagging_full.npz`, then

```bash
python scripts/import_logits.py --name my_teacher --format logits --file my_logits_{split}.npz
python scripts/train.py --model lgatr_slim_200k --teacher my_teacher --teacher-seed 1001
```

The import checks that every row matches the dataset. [REPRODUCE.md](REPRODUCE.md#pretrained-part-and-omnilearned-s)
shows how we did this for pretrained ParT and OmniLearned-S.

## Reproducing the paper

[`REPRODUCE.md`](REPRODUCE.md) gives the commands behind every table and figure: retraining every
model, evaluating the released checkpoints and timing the models on your own hardware.

## Acknowledgements

This work builds on public code. The networks of [lloca](https://github.com/heidelberg-hepml/lloca)
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

## Citation

If you find this code useful in your research, please cite our paper

```bibtex
@article{agarwal2026distilling,
  title  = {How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?},
  author = {Agarwal, Jay and Khare, Siddharth and Kumar, Dhruv},
  year   = {2026},
}
```
