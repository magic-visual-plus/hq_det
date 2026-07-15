"""YOLO output adapter kept separate from the industrial metric strategy."""

import torch

from hq_det.training.interfaces import PostprocessStrategy


def _unwrap_model(model):
    while hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        model = model.module
    return model


class UltralyticsV847Postprocess(PostprocessStrategy):
    def process(
        self,
        model,
        batch_data,
        forward_result,
        confidence: float = 0.0,
    ):
        model = _unwrap_model(model)
        return model.postprocess(
            forward_result, batch_data, confidence=float(confidence)
        )


__all__ = ["UltralyticsV847Postprocess"]
