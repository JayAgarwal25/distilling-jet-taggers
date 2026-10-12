"""Ranger: RAdam inside Lookahead, as in lloca-experiments (Spinner et al.).

This version applies weight decay inside RAdam's update, which differs from
torch.optim.RAdam and from pytorch-optimizer's Lookahead. The ParT and MIParT recipes
were trained with it.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch.optim import Optimizer


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                p32 = p.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p32)
                    state["exp_avg_sq"] = torch.zeros_like(p32)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                state["step"] += 1
                t = state["step"]

                # the original caches these step sizes; recomputing them gives identical
                # updates (verified in training)
                beta2_t = beta2**t
                n_sma_max = 2 / (1 - beta2) - 1
                n_sma = n_sma_max - 2 * t * beta2_t / (1 - beta2_t)
                if n_sma >= 5:
                    step_size = math.sqrt(
                        (1 - beta2_t) * (n_sma - 4) / (n_sma_max - 4) * (n_sma - 2) / n_sma
                        * n_sma_max / (n_sma_max - 2)) / (1 - beta1**t)
                else:
                    step_size = 1.0 / (1 - beta1**t)

                if group["weight_decay"] != 0:
                    p32.add_(p32, alpha=-group["weight_decay"] * group["lr"])
                if n_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p32.addcdiv_(exp_avg, denom, value=-step_size * group["lr"])
                else:
                    p32.add_(exp_avg, alpha=-step_size * group["lr"])
                p.data.copy_(p32)


class Lookahead(Optimizer):
    """Every k steps, move the weights a fraction alpha of the way from the slow weights."""

    def __init__(self, optimizer, alpha=0.5, k=6):
        self.optimizer = optimizer
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.alpha, self.k = alpha, k
        self.counter = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["cached_params"] = p.data.clone()

    def step(self, closure=None):
        self.optimizer.step()
        self.counter += 1
        if len(self.state) == 0:
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["cached_params"] = p.data.clone()
        if self.counter >= self.k:
            self.counter = 0
            for group in self.param_groups:
                for p in group["params"]:
                    cached = self.state[p]["cached_params"]
                    p.data.mul_(self.alpha).add_(cached, alpha=1.0 - self.alpha)
                    cached.copy_(p.data)


def Ranger(params, lr=1e-3, betas=(0.95, 0.999), eps=1e-5, alpha=0.5, k=6):
    return Lookahead(RAdam(params, lr=lr, betas=betas, eps=eps), alpha, k)
