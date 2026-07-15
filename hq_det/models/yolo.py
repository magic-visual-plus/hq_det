import os
import re
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import ultralytics.utils
from ultralytics import __version__
from ultralytics.utils import DEFAULT_CFG

try:
    from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
except ImportError:
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint

    def attempt_load_one_weight(weight):
        return load_checkpoint(weight)

from ..common import PredictionResult
from .base import HQModel


YOLO26_SCALES = ("n", "s", "m", "l", "x")


def build_yolo26_model_name(scale="n", pretrained=True, p2=False):
    scale = str(scale).lower()
    if scale not in YOLO26_SCALES:
        raise ValueError(f"Unsupported YOLO26 scale: {scale}. Use one of {YOLO26_SCALES}.")
    suffix = "-p2" if p2 else ""
    # Ultralytics ships P2/P6 architectures as YAML only, without scale-specific weights.
    ext = "yaml" if p2 else "pt" if pretrained else "yaml"
    return f"yolo26{scale}{suffix}.{ext}"


def infer_yolo26_scale(source):
    if not source:
        return None
    match = re.search(r"yolo(?:v)?26([nslmx])", str(source).lower())
    return match.group(1) if match else None


def infer_yolo26_cfg(source, scale=None, p2=False):
    scale = scale or infer_yolo26_scale(source)
    if scale is None:
        return None
    return build_yolo26_model_name(scale=scale, pretrained=False, p2=p2)


class HQYOLO(HQModel):
    def __init__(self, class_id2names, **kwargs):
        super(HQYOLO, self).__init__()

        model_source = (
            kwargs.get("model")
            or kwargs.get("model_path")
            or kwargs.get("weights")
            or kwargs.get("cfg")
        )
        if model_source is None:
            raise ValueError("HQYOLO requires model/model_path/weights/cfg.")

        self.model_source = str(model_source)
        self.class_names = self._build_class_names(class_id2names)
        self.ckpt = {}
        self.criterion = None
        requested_args = kwargs.get("args")
        self.hyp = self._build_args(requested_args)
        self._apply_training_args(kwargs)

        cfg = kwargs.get("cfg")
        self.model = self._load_detection_model(
            self.model_source,
            nc=len(self.class_names),
            cfg=cfg,
            scale=kwargs.get("scale"),
            p2=kwargs.get("p2", False),
        )
        self._apply_args(requested_args)
        self._apply_training_args(kwargs)
        self.model.names = {i: name for i, name in enumerate(self.class_names)}
        self.model.args = self.hyp

    def _build_class_names(self, class_id2names):
        if class_id2names is None:
            return []
        return [class_id2names[class_id] for class_id in sorted(class_id2names)]

    def _build_args(self, args=None):
        hyp = copy(DEFAULT_CFG)
        self._update_args(hyp, args)
        return hyp

    @staticmethod
    def _update_args(hyp, args):
        if args is None:
            return
        if isinstance(args, dict):
            items = args.items()
        else:
            items = vars(args).items()
        for key, value in items:
            try:
                setattr(hyp, key, value)
            except Exception:
                pass

    def _apply_args(self, args):
        self._update_args(self.hyp, args)

    def _apply_training_args(self, kwargs):
        epochs = kwargs.get("epochs")
        if epochs is not None:
            setattr(self.hyp, "epochs", epochs)
        image_size = kwargs.get("image_size")
        if image_size is not None:
            setattr(self.hyp, "imgsz", image_size)

    def _load_detection_model(self, source, nc, cfg=None, scale=None, p2=False):
        suffix = Path(str(source)).suffix.lower()

        if suffix in {".yaml", ".yml"}:
            return DetectionModel(str(source), nc=nc, verbose=False)

        if suffix == ".pth":
            cfg = cfg or infer_yolo26_cfg(source, scale=scale, p2=p2)
            if cfg is None:
                raise ValueError(
                    "Loading a YOLO .pth state_dict requires cfg or an inferable YOLO26 scale."
                )
            model = DetectionModel(cfg, nc=nc, verbose=False)
            state_dict = self._load_state_dict(source)
            model.load_state_dict(state_dict, strict=False)
            return model

        loaded_model, ckpt = attempt_load_one_weight(str(source))
        cfg = cfg or getattr(loaded_model, "yaml", None)
        if cfg is None:
            raise ValueError(f"Unable to find model yaml in weights: {source}")

        self.ckpt = ckpt if isinstance(ckpt, dict) else {}
        model = DetectionModel(cfg, nc=nc, verbose=False)
        model.names = {i: name for i, name in enumerate(self.class_names)}
        model.load(loaded_model)
        loaded_args = getattr(loaded_model, "args", None)
        if loaded_args is not None:
            self.hyp = self._build_args(loaded_args)
        return model

    def _load_state_dict(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model_state = checkpoint["model"]
                if hasattr(model_state, "state_dict"):
                    return model_state.state_dict()
                return model_state
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
        return checkpoint

    def get_class_names(self):
        return self.class_names

    def forward(self, batch_data):
        return self.model(batch_data["img"])

    def preprocess(self, batch_data):
        return batch_data

    def _prediction_tensor(self, forward_result):
        prediction = forward_result
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        if isinstance(prediction, dict):
            for key in ("one2one", "preds", "prediction"):
                if key in prediction:
                    prediction = prediction[key]
                    break
        return prediction

    def _filter_end2end_predictions(self, prediction, confidence):
        preds = []
        max_det = getattr(self.hyp, "max_det", 300)
        for pred in prediction:
            pred = pred[pred[:, 4] >= confidence]
            if pred.shape[0] > max_det:
                pred = pred[:max_det]
            preds.append(pred)
        return preds

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        prediction = self._prediction_tensor(forward_result)
        end2end = bool(getattr(self.model, "end2end", False))

        if (
            end2end
            and isinstance(prediction, torch.Tensor)
            and prediction.ndim == 3
            and prediction.shape[-1] == 6
        ):
            preds = self._filter_end2end_predictions(prediction, confidence)
        else:
            nms_kwargs = dict(
                conf_thres=confidence,
                iou_thres=getattr(self.hyp, "iou", 0.7),
                classes=getattr(self.hyp, "classes", None),
                agnostic=getattr(self.hyp, "agnostic_nms", False),
                max_det=getattr(self.hyp, "max_det", 300),
                nc=len(self.class_names),
                end2end=end2end,
                rotated=False,
            )
            try:
                preds = ultralytics.utils.ops.non_max_suppression(prediction, **nms_kwargs)
            except TypeError:
                nms_kwargs.pop("end2end", None)
                nms_kwargs.pop("rotated", None)
                preds = ultralytics.utils.ops.non_max_suppression(prediction, **nms_kwargs)

        results = []
        for pred in preds:
            record = PredictionResult(
                bboxes=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                cls=np.zeros((0,), dtype=np.int32),
            )
            if pred.shape[0] > 0:
                pred = pred.detach().float().cpu()
                record.bboxes = pred[:, :4].numpy()
                record.scores = pred[:, 4].numpy()
                record.cls = pred[:, 5].numpy().astype(np.int32)
            results.append(record)
        return results

    def predict(self, imgs, bgr=True, confidence=0.0):
        tensors = []
        shapes = []
        stride = int(max(getattr(self.model, "stride", torch.tensor([32])).max().item(), 32))

        for img in imgs:
            if bgr:
                img = img[..., ::-1].copy()
            shapes.append(img.shape[:2])
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(tensor)

        max_h = int(np.ceil(max(t.shape[1] for t in tensors) / stride) * stride)
        max_w = int(np.ceil(max(t.shape[2] for t in tensors) / stride) * stride)
        batch = []
        for tensor in tensors:
            pad_h = max_h - tensor.shape[1]
            pad_w = max_w - tensor.shape[2]
            batch.append(F.pad(tensor, (0, pad_w, 0, pad_h), value=0.44))

        device = getattr(self, "device", next(self.model.parameters()).device)
        batch_data = {"img": torch.stack(batch, dim=0).to(device)}
        with torch.no_grad():
            forward_result = self.forward(batch_data)
            results = self.postprocess(forward_result, batch_data, confidence=confidence)

        for result, (height, width) in zip(results, shapes):
            if result.bboxes is not None and len(result.bboxes) > 0:
                result.bboxes[:, [0, 2]] = np.clip(result.bboxes[:, [0, 2]], 0, width)
                result.bboxes[:, [1, 3]] = np.clip(result.bboxes[:, [1, 3]], 0, height)
        return results

    def _init_criterion(self):
        if self.criterion is None:
            self.criterion = self.model.init_criterion()
            if hasattr(self.criterion, "hyp"):
                self.criterion.hyp = self.hyp
        return self.criterion

    def compute_loss(self, batch_data, forward_result):
        criterion = self._init_criterion()
        loss, loss_item = criterion(forward_result, batch_data)
        loss_item = loss_item.detach() if isinstance(loss_item, torch.Tensor) else loss_item
        box_loss = float(loss_item[0])
        cls_loss = float(loss_item[1]) if len(loss_item) > 1 else 0.0
        dfl_loss = float(loss_item[2]) if len(loss_item) > 2 else 0.0

        info = {
            "box": box_loss,
            "cls": cls_loss,
            "dfl": dfl_loss,
        }

        return loss.sum(), info

    def update_epoch(self):
        criterion = self._init_criterion()
        if hasattr(criterion, "update"):
            criterion.update()

    def _checkpoint_paths(self, path):
        root, ext = os.path.splitext(path)
        if ext in {".pt", ".pth"}:
            return root + ".pt", root + ".pth"
        return path + ".pt", path + ".pth"

    def save(self, path):
        pt_path, pth_path = self._checkpoint_paths(path)
        model_to_save = deepcopy(self.model).half()
        model_to_save.names = {i: name for i, name in enumerate(self.class_names)}

        updates = {
            "model": model_to_save,
            "ema": None,
            "optimizer": None,
            "train_args": vars(self.hyp),
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, pt_path)
        torch.save(self.model.state_dict(), pth_path)
