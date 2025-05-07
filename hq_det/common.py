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