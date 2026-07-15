import re

from ultralytics import __version__ as ultralytics_version

from .yolo import HQYOLO, YOLO26_SCALES, build_yolo26_model_name, infer_yolo26_cfg


def _version_tuple(version):
    parts = [int(value) for value in re.findall(r"\d+", version)[:3]]
    return tuple((parts + [0, 0, 0])[:3])


if _version_tuple(ultralytics_version) != (8, 4, 7):
    raise ImportError(
        "YOLO26 requires exactly ultralytics==8.4.7 so its model, loss, "
        "and low-level augmentation primitives stay version-aligned. "
        f"Installed version: {ultralytics_version}. Install it with: "
        'python -m pip install --force-reinstall "ultralytics==8.4.7"'
    )


class HQYOLO26(HQYOLO):
    def __init__(self, class_id2names, **kwargs):
        scale = kwargs.pop("scale", "n")
        pretrained = kwargs.pop("pretrained", True)
        p2 = kwargs.pop("p2", False)
        model_source = kwargs.get("model") or kwargs.get("model_path") or kwargs.get("weights")

        if model_source is None:
            model_source = build_yolo26_model_name(
                scale=scale,
                pretrained=pretrained,
                p2=p2,
            )
            kwargs["model"] = model_source

        if str(model_source).lower().endswith(".pth") and "cfg" not in kwargs:
            kwargs["cfg"] = infer_yolo26_cfg(model_source, scale=scale, p2=p2)

        super().__init__(
            class_id2names=class_id2names,
            scale=scale,
            p2=p2,
            **kwargs,
        )
        self.yolo26_scale = scale


__all__ = ["HQYOLO26", "YOLO26_SCALES", "build_yolo26_model_name"]
