"""Every model in the paper, by name. `build(name)` returns it untrained.

Each model maps a `JetBatch` to one logit per jet.
"""

from __future__ import annotations

from functools import partial

from torch import nn

from jetkd.models.lgatr import LGATrSlimTagger, LGATrTagger
from jetkd.models.part import ParTTagger
from jetkd.models.transformer import TransformerTagger


def _transformer(heads: int, blocks: int, mlp: int, reps: str = "4x0n+1x1n"):
    return partial(TransformerTagger, attn_reps=reps, num_heads=heads, num_blocks=blocks,
                   mlp_factor=mlp)


def _slim(v: int, s: int, heads: int, blocks: int, mlp: int = 4, float64: bool = False):
    return partial(LGATrSlimTagger, hidden_v_channels=v, hidden_s_channels=s, num_heads=heads,
                   num_blocks=blocks, mlp_ratio=mlp, float64=float64)


MODELS = {
    # teachers
    "lgatr": LGATrTagger,
    "lgatr_slim": _slim(32, 96, heads=8, blocks=12, float64=True),
    # students
    "transformer_200k": _transformer(6, blocks=10, mlp=2),
    "lgatr_slim_200k": _slim(12, 48, heads=4, blocks=5),
    "part_200k": ParTTagger,
    # width study: the deep ladder of the Economical Jet Taggers (arXiv:2512.17011, Table 7)
    "transformer_deep_200k": _transformer(4, blocks=10, mlp=2),
    "transformer_deep_20k": _transformer(2, blocks=10, mlp=1),
    "transformer_deep_2k": _transformer(1, blocks=10, mlp=1, reps="1x1n"),
    "lgatr_slim_deep_200k": _slim(8, 32, heads=4, blocks=10),
    "lgatr_slim_deep_20k": _slim(4, 16, heads=2, blocks=10, mlp=1),
    "lgatr_slim_deep_2k": _slim(2, 4, heads=1, blocks=10, mlp=1),
    # depth study: only the block count of the student changes
    **{f"transformer_200k_L{n}": _transformer(6, blocks=n, mlp=2) for n in (0, 1, 2, 3, 4, 6, 8)},
    **{f"lgatr_slim_200k_L{n}": _slim(12, 48, heads=4, blocks=n) for n in (1, 2, 3, 4, 6, 8, 10)},
}


def build(name: str) -> nn.Module:
    return MODELS[name]()
