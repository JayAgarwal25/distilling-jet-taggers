"""Distillation loss.

    L = alpha * T^2 * KL(p_T || p_S) + (1 - alpha) * BCE(z_S, y),   with p = sigmoid(z / T)

z_S and z_T are the student and teacher logits and y is the label. The KL term compares
the softened teacher and student probabilities, the binary case of Hinton et al.'s
distillation loss. The T^2 factor keeps its gradients on the scale of the BCE term as T
changes. Every run in the paper uses alpha = 0.7 and T = 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class KDConfig:
    alpha: float = 0.7
    temperature: float = 2.0


def binary_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float):
    """Mean KL divergence between the softened teacher and student, computed from logits."""
    s = student_logits / temperature
    t = teacher_logits.detach() / temperature
    pt = torch.sigmoid(t)
    kl = (pt * (F.logsigmoid(t) - F.logsigmoid(s))
          + (1.0 - pt) * (F.logsigmoid(-t) - F.logsigmoid(-s)))
    return kl.mean()


def kd_loss(student_logits, labels, teacher_logits=None, cfg: KDConfig | None = None):
    """Total loss and its parts: `hard` (BCE) and `soft` (T^2 times the KL).

    Without teacher logits, as for training from scratch, the loss is the BCE alone.
    """
    cfg = cfg or KDConfig()
    hard = F.binary_cross_entropy_with_logits(student_logits, labels)
    if teacher_logits is None:
        return hard, {"hard": hard.detach(), "soft": None}
    soft = cfg.temperature**2 * binary_kl(student_logits, teacher_logits, cfg.temperature)
    total = cfg.alpha * soft + (1.0 - cfg.alpha) * hard
    return total, {"hard": hard.detach(), "soft": soft.detach()}
