from ultralytics import YOLO
import torch.nn
from ultralytics.utils import DEFAULT_CFG
import ultralytics.utils
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.nn.tasks import attempt_load_one_weight, DetectionModel
from ..common import PredictionResult
import numpy as np

from .base import HQModel

class HQYOLO(HQModel):
    def __init__(self, class_id2names, **kwargs):
        super(HQYOLO, self).__init__()
        model, weights = attempt_load_one_weight(kwargs['model'])
        cfg = model.yaml
        self.model = DetectionModel(cfg, nc=len(class_id2names), verbose=False)
        self.model.load(weights)
        self.model.args = model.args
        self.criterion = None
        self.hyp = DEFAULT_CFG
        self.class_names = ['' for _ in range(len(class_id2names))]
        self.ckpt = weights
        for i in range(len(class_id2names)):
            self.class_names[i] = class_id2names[i]
            pass

    def get_class_names(self):
        # Get the class names from the model
        return self.model.names

    def load_model(self):
        # Load the YOLO model using the specified path and device
        pass

    def forward(self, batch_data):
        return self.model(batch_data['img'])
        pass
    
    def preprocess(self, batch_data):
        # Preprocess the input data for the YOLO model
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        # Post-process the predictions
        preds = ultralytics.utils.ops.non_max_suppression(
            forward_result,
            confidence,
            self.hyp.iou,
            self.hyp.classes,
            self.hyp.agnostic_nms,
            max_det=self.hyp.max_det,
            nc=len(self.model.names),
            end2end=False,
            rotated=False,
        )
        # print(preds)
        
        results = []
        for pred in preds:
            record = PredictionResult()
            if pred.shape[0] == 0:
                pass
            else:
                # add pred
                pred_bboxes = pred[:, :4].cpu().numpy()
                pred_scores = pred[:, 4].cpu().numpy()
                pred_cls = pred[:, 5].cpu().numpy().astype(np.int32)
                record.bboxes = pred_bboxes
                record.scores = pred_scores
                record.cls = pred_cls
                pass
            results.append(record)
        return results

    
    def predict(self, imgs):
        pass

    def compute_loss(self, batch_data, forward_result):
        # Compute the loss using the YOLO model
        # This is a placeholder; actual implementation may vary
        if self.criterion is None:
            self.criterion = self.model.init_criterion()
            self.criterion.hyp = DEFAULT_CFG
            pass
        loss, loss_item = self.criterion(forward_result, batch_data)
        
        info = {
            'box': loss_item[0].item(),
            'cls': loss_item[1].item(),
            'giou': loss_item[2].item(),
        }
        
        return loss.sum(), info

    def save(self, path):
        # Save the YOLO model to the specified path
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, path + '.pt')
        torch.save(self.model.state_dict(), path + '.pth')
        pass

    pass