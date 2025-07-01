import json
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple

import torch.nn
import torchvision.transforms.functional as VF

from hq_det.models.rfdetr.config import RFDETRBaseConfig, RFDETRLargeConfig, TrainConfig, ModelConfig
from hq_det.models.rfdetr.main import Model as RFDETR_Model
from hq_det.models.rfdetr.main import populate_args
from hq_det.models.rfdetr.models import build_criterion_and_postprocessors
from hq_det.models.rfdetr.util import misc as utils

from hq_det.common import HQTrainerArguments, PredictionResult
from hq_det.models.base import HQModel
from hq_det.models.rfdetr.util.get_param_dicts import get_param_dict
from hq_det.models.rfdetr.util.misc import nested_tensor_from_tensor_list

from hq_det import torch_utils


class HQRFDETR(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQRFDETR, self).__init__(class_id2names, **kwargs)
        if class_id2names is None:
            pass
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
        self.model = self.load_model(self.model_config)
        self.train_config = TrainConfig(**kwargs)
        all_kwargs = self.train_from_config(self.train_config, **kwargs)
        self.args = populate_args(**all_kwargs)
        utils.init_distributed_mode(self.args)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(self.args)
        self.image_size = kwargs.get('image_size', 1024)


    def get_class_names(self):
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
        return names

    def train_from_config(self, config: TrainConfig, **kwargs):
        class_names = self.get_class_names()
        num_classes = len(class_names)
        train_config = config.dict()
        model_config = self.model_config.dict()
        model_config.pop("num_classes")
        if "class_names" in model_config:
            model_config.pop("class_names")
        
        if "class_names" in train_config and train_config["class_names"] is None:
            train_config["class_names"] = class_names
        for k, v in train_config.items():
            if k in model_config:
                model_config.pop(k)
            if k in kwargs:
                kwargs.pop(k)
        all_kwargs = {**model_config, **train_config, **kwargs, "num_classes": num_classes}

        return all_kwargs 

    def load_model(self, model_config: ModelConfig):
        model_cls =  RFDETR_Model(**model_config.dict())
        return model_cls.model

    def forward(self, batch_data: Dict):
        samples = batch_data['img']
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
                record.image_id = batch_data['image_id'][i]
            results.append(record)
        return results
        

    def compute_loss(self, batch_data: Dict, forward_result: Tuple) -> Tuple[torch.FloatTensor, Dict]:
        if self.training:
            self.criterion.train()
        else:
            self.criterion.eval()
        targets = batch_data['targets']
        
        loss_dict = self.criterion(forward_result, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k]
            for k in loss_dict.keys()
            if k in weight_dict
        )
        if torch.isnan(losses):
            raise ValueError("Loss is NaN, please check your model and data.")
        
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
            'cls': loss_dict['loss_ce'].item(),
            'box': loss_dict['loss_bbox'].item(),
            'giou': loss_dict['loss_giou'].item(),
            # 'class_error': loss_dict['class_error'].item(),
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


    def imgs_to_batch(self, imgs: List[np.ndarray]) -> Dict:
        max_h, max_w = self.image_size, self.image_size

        new_imgs = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = VF.to_tensor(img)
            img, _ = torch_utils.pad_image(img, torch.zeros((0, 4)), (max_h, max_w))
            new_imgs.append(img)
        
        new_imgs = torch.stack(new_imgs, 0)

        targets = [
            {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int32),
                "image_id": 0,
                'iscrowd': torch.zeros((0,), dtype=torch.int64),
                'area': torch.zeros((0,), dtype=torch.float32),
                'orig_size': torch.tensor(img.shape[1:3], dtype=torch.int64),
                'size': torch.tensor(img.shape[1:3], dtype=torch.int64),
            }
            for _ in range(len(imgs))
        ]

        return {
            "img": new_imgs,
            "targets": targets,
            "image_id": [0 for _ in range(len(imgs))],
        }


    def predict(self, imgs: List[np.ndarray], bgr: bool = False, confidence: float = 0.0, max_size: int = -1) -> List[PredictionResult]:
        if not bgr:
            for i in range(len(imgs)):
                imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        img_scales = np.ones((len(imgs),))
        if max_size > 0:
            for i in range(len(imgs)):
                max_hw = max(imgs[i].shape[0], imgs[i].shape[1])
                if max_hw > max_size:
                    rate = max_size / max_hw
                    imgs[i] = cv2.resize(imgs[i], (int(imgs[i].shape[1] * rate), int(imgs[i].shape[0] * rate)))
                    img_scales[i] = rate
        
        device = self.device
        with torch.no_grad():
            batch_data = self.imgs_to_batch(imgs)
            batch_data = torch_utils.batch_to_device(batch_data, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                forward_result = self.forward(batch_data)
                preds = self.postprocess(forward_result, batch_data, confidence)
            torch.cuda.empty_cache()
        
        for i in range(len(preds)):
            pred = preds[i]
            pred.bboxes = pred.bboxes / img_scales[i]

        return preds
    
    def get_param_dict(self, args: HQTrainerArguments):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.args.lr = args.lr0
        params =  get_param_dict(self.args, model)
        return params

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
        self.device = torch.device(device)