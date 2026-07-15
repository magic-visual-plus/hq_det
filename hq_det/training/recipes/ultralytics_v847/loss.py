"""Loss lifecycle adapter for the 8.4.7 YOLO criterion."""

import torch

from hq_det.training.interfaces import LossStrategy


def _unwrap_model(model):
    while hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        model = model.module
    return model


class UltralyticsV847Loss(LossStrategy):
    """Invoke the pinned model criterion through a framework-owned interface."""

    def compute(self, model, batch_data, forward_result):
        return _unwrap_model(model).compute_loss(batch_data, forward_result)

    def on_epoch_end(self, model) -> None:
        model = _unwrap_model(model)
        if hasattr(model, "update_epoch"):
            model.update_epoch()


__all__ = ["UltralyticsV847Loss"]
