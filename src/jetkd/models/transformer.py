"""The Transformer tagger: lloca's Transformer on the seven particle features, mean pooled."""

from __future__ import annotations

import torch
from lloca.backbone.transformer import Transformer
from lloca.framesnet.frames import Frames
from torch import nn
from torch_geometric.utils import scatter

from jetkd.data import JetBatch
from jetkd.models.inputs import attention_mask, embed, pack


class TransformerTagger(nn.Module):
    def __init__(self, attn_reps: str, num_heads: int, num_blocks: int, mlp_factor: int):
        super().__init__()
        self.net = Transformer(in_channels=7, attn_reps=attn_reps, out_channels=1,
                               num_blocks=num_blocks, num_heads=num_heads, mlp_factor=mlp_factor)

    # tagging-guide's TransformerWrapper for identity frames, written out; same initial
    # weights, logits and checkpoints (verified bit for bit)
    def forward(self, batch: JetBatch) -> torch.Tensor:
        _, feats, mask = embed(batch.p4, spurions=False, dtype=torch.float32)
        [x], jet, ptr = pack(mask, feats)
        # the network is not Lorentz-equivariant, so every token keeps the lab frame
        frames = Frames(is_identity=True, device=x.device, dtype=x.dtype, shape=x.shape[:-1])
        out = self.net(inputs=x.unsqueeze(0), frames=frames.reshape(1, *frames.shape),
                       **attention_mask(jet, x.dtype))
        return scatter(out.squeeze(0), jet, dim=0, dim_size=len(ptr) - 1, reduce="mean")[:, 0]
