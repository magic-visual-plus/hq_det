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
from hq_det.common import HQTrainerArguments, PredictionResult
from hq_det.models.base import HQModel
from hq_det.models.rfdetr.util.misc import NestedTensor
from hq_det.models.rfdetr.models.backbone import Joiner
from hq_det.models.rfdetr.util.misc import nested_tensor_from_tensor_list


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
        self.train_config = TrainConfig(**kwargs)
        self.args = populate_args({**self.model_config.dict(), **self.train_config.dict()})
        print(self.args)
        self.model, self.criterion, self.postprocessors = self.load_model(self.model_config)

    def get_class_names(self):
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
        return names

    def load_model(self, model_config: ModelConfig):
        model_cls =  RFDETR_Model(**model_config.dict())
        return model_cls.model, model_cls.criterion, model_cls.postprocessors

    def forward(self, batch_data: Dict):
        samples = nested_tensor_from_tensor_list(batch_data['img'])
        targets = batch_data['targets'] 
        return self.model(samples, targets)
        

    def postprocess(self, forward_result: Tuple, batch_data: Dict, confidence: float = 0.0) -> List[PredictionResult]:
        orig_target_sizes = torch.stack([t["orig_size"] for t in batch_data['targets']], dim=0)
        preds = self.postprocessors["bbox"](forward_result, orig_target_sizes)

        results = []

        for i, pred in enumerate(preds):
            record = PredictionResult()
            if len(pred['labels']) > 0:
                pred_bboxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_cls = pred['labels'].cpu().numpy().astype(np.int32)
                record.bboxes = pred_bboxes
                record.cls = pred_cls
                record.scores = pred_scores
                record.image_id = batch_data['targets'][i]['image_id']
            results.append(record)
        return results
        

    def compute_loss(self, batch_data: Dict, forward_result: Tuple) -> Tuple[torch.FloatTensor, Dict]:
        targets = batch_data['targets']
        
        loss_dict = self.criterion(forward_result, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k]
            for k in loss_dict.keys()
            if k in weight_dict
        )

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        info = {
            'loss': loss_value,
            'class_error': loss_dict['class_error'].item(),
            'cls': loss_dict['loss_ce'].item(),
            'box': loss_dict['loss_bbox'].item(),
            'giou': loss_dict['loss_giou'].item(),
            # 'loss_ce_0': loss_dict['loss_ce_0'].item(),
            # 'loss_bbox_0': loss_dict['loss_bbox_0'].item(),
            # 'loss_giou_0': loss_dict['loss_giou_0'].item(),
            # 'loss_ce_1': loss_dict['loss_ce_1'].item(),
            # 'loss_bbox_1': loss_dict['loss_bbox_1'].item(),
            # 'loss_giou_1': loss_dict['loss_giou_1'].item(),
            # 'cardinality_error': loss_dict['cardinality_error'].item(),
            # 'cardinality_error_0': loss_dict['cardinality_error_0'].item(),
            # 'cardinality_error_1': loss_dict['cardinality_error_1'].item(),
            # 'loss_ce_enc': loss_dict['loss_ce_enc'].item(),
            # 'loss_bbox_enc': loss_dict['loss_bbox_enc'].item(),
            # 'loss_giou_enc': loss_dict['loss_giou_enc'].item(),
            # 'weight_dict': weight_dict,
        }
        return losses, info
    
    def get_param_dict(self, args: HQTrainerArguments):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        backbone = model.backbone[0]
        backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(self.args, prefix="backbone.0")
        backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]
        
        decoder_key = 'transformer.decoder'
        decoder_params = [
            p
            for n, p in model.named_parameters() if decoder_key in n and p.requires_grad
        ]

        decoder_param_lr_pairs = [
            {"params": param, "lr": args.lr0 * self.args.lr_component_decay} 
            for param in decoder_params
        ]

        other_params = [
            p
            for n, p in model.named_parameters() if (
                n not in backbone_named_param_lr_pairs and decoder_key not in n and p.requires_grad)
        ]
        other_param_dicts = [
            {"params": param, "lr": args.lr0} 
            for param in other_params
        ]
        
        final_param_dicts = (
            other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs
        )

        return final_param_dicts

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