"""Composite Ultralytics 8.4.7 detection recipe selected by YOLO26."""

import math
import re

from ultralytics import __version__ as ultralytics_version

from hq_det.training.interfaces import DetectionTrainingRecipe

from .augmentation import UltralyticsV847Augmentation
from .config import build_config
from .dataloader import UltralyticsV847DataLoader
from .ema import UltralyticsV847EMA
from .loss import UltralyticsV847Loss
from .optimizer import UltralyticsV847Optimizer
from .precision import UltralyticsV847Precision
from .postprocess import UltralyticsV847Postprocess
from .runtime import UltralyticsV847Runtime
from .schedule import UltralyticsV847Scheduler


ULTRALYTICS_V847_VERSION = "8.4.7"


def _version_tuple(version):
    parts = [int(value) for value in re.findall(r"\d+", str(version))[:3]]
    return tuple((parts + [0, 0, 0])[:3])


def require_ultralytics_v847():
    """Fail closed instead of silently changing model/loss primitive behavior."""
    if _version_tuple(ultralytics_version) != (8, 4, 7):
        raise ImportError(
            "The local YOLO26 recipe requires ultralytics==8.4.7 for its model, "
            "criterion, and low-level image primitives. Installed version: "
            f"{ultralytics_version}."
        )


class UltralyticsV847Recipe(DetectionTrainingRecipe):
    version = ULTRALYTICS_V847_VERSION

    def __init__(self):
        require_ultralytics_v847()
        self.augmentation = UltralyticsV847Augmentation()
        self.dataloader = UltralyticsV847DataLoader()
        self.optimizer = UltralyticsV847Optimizer()
        self.scheduler = UltralyticsV847Scheduler()
        self.ema = UltralyticsV847EMA()
        self.precision = UltralyticsV847Precision()
        self.postprocess = UltralyticsV847Postprocess()
        self.runtime = UltralyticsV847Runtime()
        self.loss = UltralyticsV847Loss()

    def build_config(self, args):
        config = build_config(args)
        if float(config.multi_scale) != 0.0:
            raise ValueError(
                "multi_scale is not exposed by the local 8.4.7 detection recipe."
            )
        if bool(config.compile):
            raise ValueError(
                "compile is not exposed by the local 8.4.7 detection recipe."
            )
        probability_names = (
            "bgr",
            "copy_paste",
            "cutmix",
            "flipud",
            "fliplr",
            "fraction",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "iou",
            "lr0",
            "lrf",
            "mixup",
            "momentum",
            "mosaic",
            "perspective",
            "scale",
            "translate",
            "warmup_bias_lr",
            "warmup_momentum",
            "weight_decay",
        )
        invalid = [
            name
            for name in probability_names
            if not 0.0 <= float(getattr(config, name)) <= 1.0
        ]
        if invalid:
            raise ValueError(
                "Recipe probabilities must be in [0, 1]: " + ", ".join(invalid)
            )
        return config

    def validate_arguments(self, args) -> None:
        optimizer = str(getattr(args, "optimizer", "auto")).lower()
        if optimizer not in {"auto", "musgd"}:
            raise ValueError("YOLO26 8.4.7 accepts only optimizer=auto or MuSGD.")
        if int(args.gradient_update_interval) != 1:
            raise ValueError(
                "YOLO26 derives accumulation from nbs/global batch; "
                "gradient_update_interval must be 1."
            )
        if not bool(args.use_ema):
            raise ValueError("YOLO26 8.4.7 requires EMA to remain enabled.")
        if not math.isclose(float(args.ema_decay), 0.9999) or not math.isclose(
            float(args.ema_tau), 2000.0
        ):
            raise ValueError(
                "YOLO26 8.4.7 requires EMA decay=0.9999 and tau=2000.0."
            )
        if int(getattr(args, "nbs", 64)) <= 0:
            raise ValueError("nbs must be positive.")
        if int(args.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(args.num_epoches) <= 0:
            raise ValueError("num_epoches must be positive.")
        if getattr(args, "copy_paste_mode", "flip") not in {"flip", "mixup"}:
            raise ValueError("copy_paste_mode must be flip or mixup.")
        if not math.isclose(float(args.max_grad_norm), 10.0):
            raise ValueError("YOLO26 8.4.7 requires max_grad_norm=10.0.")


__all__ = [
    "ULTRALYTICS_V847_VERSION",
    "UltralyticsV847Recipe",
    "require_ultralytics_v847",
]
