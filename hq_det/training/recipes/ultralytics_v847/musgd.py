# Ported from Ultralytics 8.4.7, ultralytics/optim/muon.py.
# This file is distributed under the AGPL-3.0 license:
# https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import optim


def zeropower_via_newtonschulz5(
    gradient: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Approximate matrix orthogonalization with five Newton-Schulz steps."""
    assert len(gradient.shape) == 2
    value = gradient.bfloat16()
    value /= value.norm() + eps
    transposed = gradient.size(0) > gradient.size(1)
    if transposed:
        value = value.T

    for a, b, c in [(3.4445, -4.7750, 2.0315)] * 5:
        gram = value @ value.T
        polynomial = b * gram + c * gram @ gram
        value = a * value + polynomial @ value

    return value.T if transposed else value


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    """Apply the 8.4.7 Muon momentum and orthogonalization update."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class MuSGD(optim.Optimizer):
    """Ultralytics 8.4.7 hybrid Muon and SGD optimizer."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    lr = group["lr"]
                    grad = parameter.grad
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                        state["momentum_buffer_SGD"] = torch.zeros_like(parameter)

                    update = muon_update(
                        grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        nesterov=group["nesterov"],
                    )
                    parameter.add_(update.reshape(parameter.shape), alpha=-(lr * self.muon))

                    if group["weight_decay"] != 0:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    state["momentum_buffer_SGD"].mul_(group["momentum"]).add_(grad)
                    sgd_update = (
                        grad.add(
                            state["momentum_buffer_SGD"], alpha=group["momentum"]
                        )
                        if group["nesterov"]
                        else state["momentum_buffer_SGD"]
                    )
                    parameter.add_(sgd_update, alpha=-(lr * self.sgd))
            else:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    lr = group["lr"]
                    grad = parameter.grad
                    if group["weight_decay"] != 0:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
                    update = (
                        grad.add(state["momentum_buffer"], alpha=group["momentum"])
                        if group["nesterov"]
                        else state["momentum_buffer"]
                    )
                    parameter.add_(update, alpha=-lr)
        return loss


__all__ = ["MuSGD", "muon_update", "zeropower_via_newtonschulz5"]
