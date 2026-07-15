from typing import List

import numpy as np
import pydantic
from omegaconf import DictConfig

from . import box_utils


class PredictionResult(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True

    image_id: int = None
    bboxes: np.ndarray = None
    scores: np.ndarray = None
    cls: np.ndarray = None
    names: List[str] = None
    annotation_ids: List[int] = None

    def to_coco(self):
        coco_result = []
        if self.bboxes is None:
            return coco_result

        for i in range(len(self.bboxes)):
            bbox = box_utils.xyxy2xywh(self.bboxes[i])
            rec = {
                "image_id": self.image_id,
                "bbox": bbox,
                "score": self.scores[i],
                "category_id": int(self.cls[i]),
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
            if self.annotation_ids is not None:
                rec["id"] = int(self.annotation_ids[i])
            else:
                rec["id"] = -1
            coco_result.append(rec)
        return coco_result


class HQTrainerArguments(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        protected_namespaces=(), arbitrary_types_allowed=True
    )

    dataset_provider: str = "roboflow"
    data_path: str = ""
    data_path_list: List[str] = None
    data_path_weight_list: List[float] = None
    data_path_valid: str = None
    image_size: int = 640
    num_data_workers: int = 0

    num_epoches: int = 100
    warmup_epochs: int = 5
    batch_size: int = 4
    gradient_update_interval: int = 1
    enable_amp: bool = False
    max_grad_norm: float = 5.0
    early_stopping: bool = False
    early_stopping_patience: int = 10
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_tau: float = 2000.0

    lr0: float = 1e-4
    lr_min: float = 1e-6
    lr_backbone_mult: float = 1

    devices: List[int] = [0]

    checkpoint_path: str = "output"
    checkpoint_name: str = "ckpt.pth"
    output_path: str = "output"
    checkpoint_interval: int = 10

    model_argument: dict = {}

    class_id2names: dict = None
    eval_class_names: List[str] | None = None

    find_unused_parameters: bool = False
    sync_bn: bool = True

    augment_proba: float = 0.3
    augment_split_size: int = -1
    augment_split_proba: float = 0.5
    augment_foreground_proba: float = 0.8
    augment_foreground_path: str = ""
    augment_force_resize: bool = False

    # Kept for compatibility with config objects that are passed through.
    cfg: DictConfig = None
