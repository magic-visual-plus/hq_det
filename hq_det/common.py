import pydantic
import numpy as np
from typing import List
from . import box_utils


class PredictionResult(pydantic.BaseModel):
    # model_config  = pydantic.ConfigDict(arbitrary_types_allowed=True)
    class Config:
        arbitrary_types_allowed = True
        pass
    image_id: int = None
    bboxes: np.ndarray = None
    scores: np.ndarray = None
    cls: np.ndarray = None
    names: List[str] = None
    annotation_ids: List[int] = None

    def to_coco(self):
        # Convert to coco format
        
        coco_result = []
        for i in range(len(self.bboxes)):
            bbox = self.bboxes[i]
            bbox = box_utils.xyxy2xywh(bbox)
            score = self.scores[i]
            cls = int(self.cls[i])
            rec = {
                'image_id': self.image_id,
                'bbox': bbox,
                'score': score,
                'category_id': cls,
                'iscrowd': 0,
                'area': bbox[2] * bbox[3],
            }
            if self.annotation_ids is not None:
                rec['id'] = int(self.annotation_ids[i])
                pass
            else:
                rec['id'] = -1
                pass
            coco_result.append(rec)
            pass
        return coco_result

    pass


class HQTrainerArguments(pydantic.BaseModel):
    model_config  = pydantic.ConfigDict(protected_namespaces=())
    data_path: str
    num_epoches: int = 100
    warmup_epochs: int = 5
    num_data_workers: int = 0
    lr0: float = 1e-4
    lr_min: float = 1e-6
    lr_backbone_mult: float = 0.1
    batch_size: int = 4
    device: str = 'cuda:0'
    checkpoint_path: str = 'output'
    output_path: str = 'output'
    checkpoint_interval: int = 10
    model_argument: dict = {}
    image_size: int = 640
    enable_amp: bool = False
    gradient_update_interval: int = 1
    
    class_id2names: dict = None
    eval_class_names: list = None
    pass
