# Ported from Ultralytics 8.4.7 ModelEMA.
# This file is distributed under the AGPL-3.0 license:
# https://ultralytics.com/license

import math
from copy import deepcopy

import torch

from hq_det.training.interfaces import EMAStrategy


def _unwrap_model(model):
    while True:
        if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, torch.nn.Module):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
            model = model.module
        else:
            return model


def _copy_attr(target, source, include=(), exclude=()):
    for key, value in source.__dict__.items():
        if (include and key not in include) or key.startswith("_") or key in exclude:
            continue
        setattr(target, key, value)


class ModelEMA847:
    """Frozen 8.4.7 EMA behavior for parameters and floating-point buffers."""

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(_unwrap_model(model)).eval()
        self.updates = updates
        self.decay = lambda value: decay * (1 - math.exp(-value / tau))
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)
        self.enabled = True

    @torch.no_grad()
    def update(self, model):
        if not self.enabled:
            return
        self.updates += 1
        decay = self.decay(self.updates)
        model_state = _unwrap_model(model).state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value *= decay
                value += (1 - decay) * model_state[key].detach()

    def update_attr(
        self,
        model,
        include=(),
        exclude=("process_group", "reducer"),
    ):
        if self.enabled:
            _copy_attr(self.ema, model, include, exclude)


class UltralyticsV847EMA(EMAStrategy):
    def build(self, model, decay: float, tau: float):
        return ModelEMA847(model, decay=decay, tau=tau)


__all__ = ["ModelEMA847", "UltralyticsV847EMA"]
