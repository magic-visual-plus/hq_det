"""Ultralytics 8.4.7 behavior frozen behind HQ-DET interfaces."""

from .augmentation import UltralyticsV847Augmentation
from .config import V847_DEFAULTS
from .dataloader import InfiniteDataLoader, UltralyticsV847DataLoader
from .ema import ModelEMA847, UltralyticsV847EMA
from .loss import UltralyticsV847Loss
from .musgd import MuSGD
from .optimizer import UltralyticsV847Optimizer
from .precision import UltralyticsV847Precision
from .postprocess import UltralyticsV847Postprocess
from .recipe import (
    ULTRALYTICS_V847_VERSION,
    UltralyticsV847Recipe,
    require_ultralytics_v847,
)
from .runtime import UltralyticsV847Runtime
from .schedule import UltralyticsV847Scheduler

__all__ = [
    "InfiniteDataLoader",
    "ModelEMA847",
    "MuSGD",
    "ULTRALYTICS_V847_VERSION",
    "UltralyticsV847Augmentation",
    "UltralyticsV847DataLoader",
    "UltralyticsV847EMA",
    "UltralyticsV847Loss",
    "UltralyticsV847Optimizer",
    "UltralyticsV847Precision",
    "UltralyticsV847Postprocess",
    "UltralyticsV847Recipe",
    "UltralyticsV847Runtime",
    "UltralyticsV847Scheduler",
    "V847_DEFAULTS",
    "require_ultralytics_v847",
]
