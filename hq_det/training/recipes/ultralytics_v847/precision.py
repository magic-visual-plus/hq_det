"""AMP resolution and GradScaler creation for the 8.4.7 recipe."""

import torch

from hq_det.training.interfaces import PrecisionStrategy


class UltralyticsV847Precision(PrecisionStrategy):
    def resolve_amp(self, model, device, requested: bool) -> bool:
        device_type = torch.device(device).type
        if not requested or device_type != "cuda":
            return False
        from ultralytics.utils.checks import check_amp

        return bool(check_amp(model))

    def build_scaler(self, enabled: bool):
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda", enabled=enabled)
        return torch.cuda.amp.GradScaler(enabled=enabled)


__all__ = ["UltralyticsV847Precision"]
