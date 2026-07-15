"""Frozen Ultralytics 8.4.7 detection defaults used by the local recipe."""

from copy import deepcopy
from types import SimpleNamespace


V847_DEFAULTS = {
    "task": "detect",
    "mode": "train",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "amp": True,
    "seed": 0,
    "deterministic": True,
    "optimizer": "auto",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "nbs": 64,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": False,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "pose": 12.0,
    "kobj": 1.0,
    "rle": 1.0,
    "angle": 1.0,
    "close_mosaic": 10,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "augmentations": None,
    "overlap_mask": True,
    "mask_ratio": 4,
    "iou": 0.7,
    "max_det": 300,
    "classes": None,
    "agnostic_nms": False,
    "single_cls": False,
    "rect": False,
    "fraction": 1.0,
    "multi_scale": 0.0,
    "freeze": None,
    "compile": False,
}


_ROUTING_KEYS = {
    "data",
    "device",
    "exist_ok",
    "mode",
    "model",
    "name",
    "project",
    "task",
}


def build_config(args):
    """Build an isolated namespace without reading Ultralytics DEFAULT_CFG."""
    values = deepcopy(V847_DEFAULTS)
    values.update(
        {
            "epochs": int(args.num_epoches),
            "batch": int(args.batch_size),
            "imgsz": int(args.image_size),
            "amp": bool(args.enable_amp),
            "seed": int(getattr(args, "seed", 0)),
            "deterministic": bool(getattr(args, "deterministic", True)),
            "optimizer": getattr(args, "optimizer", "auto"),
            "lr0": float(args.lr0),
            "lrf": float(getattr(args, "lrf", 0.01)),
            "momentum": float(getattr(args, "momentum", 0.937)),
            "weight_decay": float(getattr(args, "weight_decay", 0.0005)),
            "nbs": int(getattr(args, "nbs", 64)),
            "warmup_epochs": float(args.warmup_epochs),
            "warmup_momentum": float(getattr(args, "warmup_momentum", 0.8)),
            "warmup_bias_lr": float(getattr(args, "warmup_bias_lr", 0.1)),
            "cos_lr": bool(getattr(args, "cos_lr", False)),
            "box": float(getattr(args, "box", 7.5)),
            "cls": float(getattr(args, "cls", 0.5)),
            "dfl": float(getattr(args, "dfl", 1.5)),
            "close_mosaic": int(getattr(args, "close_mosaic", 10)),
            "hsv_h": float(getattr(args, "hsv_h", 0.015)),
            "hsv_s": float(getattr(args, "hsv_s", 0.7)),
            "hsv_v": float(getattr(args, "hsv_v", 0.4)),
            "degrees": float(getattr(args, "degrees", 0.0)),
            "translate": float(getattr(args, "translate", 0.1)),
            "scale": float(getattr(args, "scale_gain", 0.5)),
            "shear": float(getattr(args, "shear", 0.0)),
            "perspective": float(getattr(args, "perspective", 0.0)),
            "flipud": float(getattr(args, "flipud", 0.0)),
            "fliplr": float(getattr(args, "fliplr", 0.5)),
            "bgr": float(getattr(args, "bgr", 0.0)),
            "mosaic": float(getattr(args, "mosaic", 1.0)),
            "mixup": float(getattr(args, "mixup", 0.0)),
            "cutmix": float(getattr(args, "cutmix", 0.0)),
            "copy_paste": float(getattr(args, "copy_paste", 0.0)),
            "copy_paste_mode": getattr(args, "copy_paste_mode", "flip"),
        }
    )

    overrides = dict(getattr(args, "yolo_overrides", {}) or {})
    conflicting = sorted(_ROUTING_KEYS.intersection(overrides))
    if conflicting:
        raise ValueError(
            "yolo_overrides cannot replace framework routing keys: "
            + ", ".join(conflicting)
        )
    unknown = sorted(set(overrides).difference(values))
    if unknown:
        raise ValueError(
            "Unknown frozen 8.4.7 recipe settings: " + ", ".join(unknown)
        )
    values.update(overrides)
    return SimpleNamespace(**values)


__all__ = ["V847_DEFAULTS", "build_config"]
