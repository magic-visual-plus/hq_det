import json
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple

import torch.nn

from hq_det.models.rfdetr.config import RFDETRBaseConfig, RFDETRLargeConfig, TrainConfig, ModelConfig
from hq_det.models.rfdetr.main import Model as RFDETR_Model
from hq_det.models.rfdetr.main import populate_args
from hq_det.models.rfdetr.models import build_criterion_and_postprocessors
from hq_det.models.rfdetr.util import misc as utils

from hq_det import torch_utils
from hq_det.common import PredictionResult
from hq_det.models.base import HQModel
from hq_det.models.rfdetr.util.misc import NestedTensor


class HQRFDETR(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQRFDETR, self).__init__(class_id2names, **kwargs)
        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            class_names = data['meta']['dataset_meta']['CLASSES']
            self.id2names = {i: name for i, name in enumerate(class_names)}
        else:
            self.id2names = class_id2names
        
        if 'model' in kwargs:
            kwargs['pretrain_weights'] = kwargs.pop('model')
        model_type = kwargs.pop('model_type', None)

        if model_type is None or model_type == "base":
            self.model_config = RFDETRBaseConfig(**kwargs)
        elif model_type == "large":
            self.model_config = RFDETRLargeConfig(**kwargs)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        self.model, self.criterion, self.postprocessors = self.load_model(self.model_config)

    def get_class_names(self):
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
        return names

    def load_model(self, model_config: ModelConfig):
        model_cls =  RFDETR_Model(**model_config.dict())
        return model_cls.model, model_cls.criterion, model_cls.postprocessors

    def forward(self, batch_data: Tuple):
        return self.model(*batch_data)
        

    def postprocess(self, forward_result: Tuple, batch_data: Dict, confidence: float = 0.0) -> List[PredictionResult]:
        """
        
        """
        pass
        

    def compute_loss(self, batch_data: Tuple, forward_result: Tuple) -> Tuple[torch.FloatTensor, Dict]:
        samples, targets = batch_data
        loss_dict = self.criterion(forward_result, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            (1.0 / len(targets)) * loss_dict[k] * weight_dict[k]
            for k in loss_dict.keys()
            if k in weight_dict
        )
        info = {
            'loss': losses.item(),
            'cls': loss_dict['loss_ce'].item(),
            'box': loss_dict['loss_bbox'].item(),
            'giou': loss_dict['loss_giou'].item(),
            'loss_ce_0': loss_dict['loss_ce_0'].item(),
            'loss_bbox_0': loss_dict['loss_bbox_0'].item(),
            'loss_giou_0': loss_dict['loss_giou_0'].item(),
            'loss_ce_1': loss_dict['loss_ce_1'].item(),
            'loss_bbox_1': loss_dict['loss_bbox_1'].item(),
            'loss_giou_1': loss_dict['loss_giou_1'].item(),
            'cardinality_error': loss_dict['cardinality_error'].item(),
            'cardinality_error_0': loss_dict['cardinality_error_0'].item(),
            'cardinality_error_1': loss_dict['cardinality_error_1'].item(),
            'loss_ce_enc': loss_dict['loss_ce_enc'].item(),
            'loss_bbox_enc': loss_dict['loss_bbox_enc'].item(),
            'loss_giou_enc': loss_dict['loss_giou_enc'].item(),
            'weight_dict': weight_dict,
        }
        return losses, info

    def save(self, path):
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'meta': {
                    'dataset_meta': {
                        'CLASSES': self.get_class_names(),
                    },
                }
            }, path + '.pth')

    def to(self, device):
        super(HQRFDETR, self).to(device)
        self.model.to(device)