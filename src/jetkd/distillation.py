"""Distillation loss: alpha * T^2 * KL(teacher || student) + (1 - alpha) * BCE(student, label).

The KL term compares the two Bernoulli distributions sigma(z / T). The paper
uses alpha = 0.7 and T = 2.
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
    s = student_logits / temperature
    t = teacher_logits.detach() / temperature
    pt = torch.sigmoid(t)
    kl = (pt * (F.logsigmoid(t) - F.logsigmoid(s))
          + (1.0 - pt) * (F.logsigmoid(-t) - F.logsigmoid(-s)))
    return kl.mean()


def kd_loss(student_logits, labels, teacher_logits=None, cfg: KDConfig | None = None):
    """Total loss and its hard and soft parts. Without teacher logits it is the plain BCE."""
    cfg = cfg or KDConfig()
    hard = F.binary_cross_entropy_with_logits(student_logits, labels)
    if teacher_logits is None:
        return hard, {"hard": hard.detach(), "soft": None}
    soft = cfg.temperature**2 * binary_kl(student_logits, teacher_logits, cfg.temperature)
    total = cfg.alpha * soft + (1.0 - cfg.alpha) * hard
    return total, {"hard": hard.detach(), "soft": soft.detach()}
