"""Training recipes: one per model family, each the recipe the family was published with.

The Transformers and the width ladders share one recipe per size class, from the
Economical Jet Taggers (arXiv:2512.17011, Table 8). Every count is in iterations, as
the source configs state them. `source` names the config each recipe was copied from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from jetkd.ranger import Ranger


@dataclass(frozen=True)
class Recipe:
    name: str
    source: str
    optimizer: str                 # AdamW, Lion or Ranger
    lr: float
    weight_decay: float
    scheduler: str                 # "cosine" (per step) or "flat+decay" (per epoch)
    batch_size: int
    iterations: int
    validate_every: int
    clip_grad_norm: float = 1.0
    es_patience: int = 100         # validations without improvement before stopping
    # tagging-guide treats a parameter as a bias when it has one non-trivial dimension
    # (`p.squeeze().ndim <= 1`); lloca-experiments when `p.ndim <= 1`. Biases get no decay.
    squeeze_bias_rule: bool = False

    def as_record(self) -> dict:
        return asdict(self)


def _size(size: str, fname: str, batch_size: int, lr: float, weight_decay: float) -> Recipe:
    return Recipe(f"size_{size}", f"tagger-quantization/config/training/top_{fname}.yaml",
                  "AdamW", lr, weight_decay, "cosine", batch_size, 200_000, 5_000)


RECIPES = {r.name: r for r in [
    _size("200k", "100k", 256, 1e-3, 0.1),
    _size("20k", "10k", 512, 3e-3, 0.0),
    _size("2k", "1k", 4096, 1e-2, 0.0),
    Recipe("part", "lloca-experiments/config/training/top_ParT.yaml",
           "Ranger", 1e-3, 0.01, "flat+decay", 512, 48_000, 2_400),
    Recipe("mipart", "lloca-experiments/config/training/top_MIParT.yaml",
           "Ranger", 1e-3, 0.01, "flat+decay", 256, 72_000, 4_800),
    Recipe("lgatr", "tagging-guide/config/training/top_lgatr.yaml",
           "Lion", 3e-4, 0.2, "cosine", 128, 200_000, 5_000, squeeze_bias_rule=True),
    Recipe("lgatr_slim", "tagging-guide/config/training/top_slim.yaml",
           "Lion", 3e-5, 2.0, "cosine", 128, 200_000, 5_000, squeeze_bias_rule=True),
    Recipe("lorentznet", "lloca-experiments/config/training/top_lorentznet.yaml",
           "AdamW", 1e-3, 0.01, "cosine", 128, 336_000, 9_600),
]}


def recipe_for(model: str) -> Recipe:
    """The recipe of a model. Depth variants (`_L<n>`) keep the recipe of their base model."""
    base = model.split("_L")[0]
    if base.startswith(("transformer", "lgatr_slim_deep")):
        return RECIPES[f"size_{base.rsplit('_', 1)[1]}"]
    if base == "lgatr_slim_200k":
        return RECIPES["lgatr_slim"]
    return RECIPES[base.removesuffix("_200k")]


def build_optimizer(model: torch.nn.Module, recipe: Recipe) -> torch.optim.Optimizer:
    def is_bias(p):
        return (p.squeeze() if recipe.squeeze_bias_rule else p).ndim <= 1

    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if not is_bias(p)], "lr": recipe.lr,
         "weight_decay": recipe.weight_decay},
        {"params": [p for p in params if is_bias(p)], "lr": recipe.lr, "weight_decay": 0.0},
    ]
    if recipe.optimizer == "AdamW":
        return torch.optim.AdamW(groups)
    if recipe.optimizer == "Lion":
        import pytorch_optimizer

        return pytorch_optimizer.Lion(groups, betas=(0.9, 0.999))
    if recipe.optimizer == "Ranger":
        return Ranger(groups)
    raise ValueError(recipe.optimizer)


def build_scheduler(optimizer, recipe: Recipe, batches_per_epoch: int):
    """The scheduler and when to step it: "step" (every iteration) or "epoch"."""
    if recipe.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.iterations), "step"
    # weaver's flat+decay: flat, then down by a factor 100 over the last 30 percent of
    # epochs, one step per epoch
    epochs = recipe.iterations // batches_per_epoch
    decay = max(1, int(epochs * 0.3))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(epochs - decay, epochs)), gamma=0.01 ** (1.0 / decay))
    return scheduler, "epoch"
