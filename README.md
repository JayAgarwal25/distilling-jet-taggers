# Distilling Foundation Models for High-Energy Physics into Fast Jet Taggers

Code, run definitions and results for the paper by Jay Agarwal, Siddharth Khare and Dhruv Kumar.
We distill large top taggers (L-GATr, L-GATr-slim, ParT, MIParT, pretrained ParT and
OmniLearned-S) into students of about 200k parameters (Transformer, ParT, MIParT,
L-GATr-slim and LorentzNet), and study how distillation depends on student width and depth.

The package is `jetkd`. It trains every model in the paper, from scratch or distilled from a
teacher's logits, and it can distill any teacher you provide.

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
[`docs/RUNS.md`](docs/RUNS.md) lists every model with its exact settings and parameter count,
every recipe and every run.

## Distilling your own teacher

Any model can be a teacher. Save its logits and labels per split, in the row order of
`data/toptagging_full.npz`, then

```bash
python scripts/import_logits.py --name my_teacher --format logits --file my_logits_{split}.npz
python scripts/train.py --model lgatr_slim_200k --teacher my_teacher --teacher-seed 1001
```

The import checks that every row matches the dataset. [`teachers/`](teachers/) shows how our
six teachers were trained, including the commands and patches for pretrained ParT and
OmniLearned-S.

## Reproducing the paper

[`REPRODUCE.md`](REPRODUCE.md) gives one command block per table: from the released
predictions (exact), from the released checkpoints, or by retraining.

## Credits

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
[OmniLearned](https://github.com/ViniciusMikuni/OmniLearned). We thank their authors.

## Citation

```bibtex
@article{agarwal2026distilling,
  title  = {Distilling Foundation Models for High-Energy Physics into Fast Jet Taggers},
  author = {Agarwal, Jay and Khare, Siddharth and Kumar, Dhruv},
  year   = {2026},
}
```

## License

MIT, see [LICENSE](LICENSE).
