import cv2
import numpy as np
from typing import List, Dict, Tuple

import torch.nn
from mmengine import MODELS
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.configs.rtmdet import rtmdet_l_8xb32_300e_coco as rtmdet_config
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import DetDataSample

from .. import torch_utils
from ..common import PredictionResult
from .base import HQModel



class HQRTMDET(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQRTMDET, self).__init__(class_id2names, **kwargs)
        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            class_names = data['meta']['dataset_meta']['CLASSES']
            self.id2names = {i: name for i, name in enumerate(class_names)}
        else:
            self.id2names = class_id2names

        rtmdet_config.model['data_preprocessor']['pad_size_divisor'] = 32 
        rtmdet_config.model['bbox_head']['num_classes'] = len(self.id2names)
        self.model = MODELS.build(rtmdet_config.model)
        self.load_model(kwargs['model'])

    def get_class_names(self):
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
        return names

    def load_model(self, path: str):
        data = torch.load(path, map_location='cpu')
        new_state_dict = {}
        for k, v in data['state_dict'].items():
            if k in self.model.state_dict() and data['state_dict'][k].shape == self.model.state_dict()[k].shape:
                new_state_dict[k] = v
        print(f"Loaded {len(new_state_dict)}/{len(data['state_dict'])} parameters")
        self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, batch_data: Dict) -> Tuple:
        batch_data.update(self.model.data_preprocessor(batch_data, self.training))
        inputs = batch_data['inputs']
        img_feats = self.model.extract_feat(inputs)
        outputs = self.model.bbox_head.forward(img_feats)
        
        return outputs

    def postprocess(self, forward_result: Tuple, batch_data: Dict, confidence: float = 0.0) -> List[PredictionResult]:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data['data_samples']
        ]
        test_cfg = ConfigDict(**self.model.bbox_head.test_cfg)
        predictions = self.model.bbox_head.predict_by_feat(
            *forward_result, batch_img_metas=batch_img_metas, rescale=False, cfg=test_cfg)
        results = []
        for pred in predictions:
            record = PredictionResult()
            mask = pred.scores > confidence
            if not mask.any():
                record.bboxes = np.zeros((0, 4), dtype=np.float32)
                record.scores = np.zeros((0,), dtype=np.float32)
                record.cls = np.zeros((0,), dtype=np.int32)
            else:
                pred_bboxes = pred.bboxes[mask].cpu().numpy()
                pred_scores = pred.scores[mask].cpu().numpy()
                pred_cls = pred.labels[mask].cpu().numpy().astype(np.int32)
                record.bboxes = pred_bboxes
                record.scores = pred_scores
                record.cls = pred_cls
            results.append(record)
        return results
        

    def imgs_to_batch(self, imgs: List[np.ndarray]) -> Dict:
        batch_data = {
            "inputs": [],
            "data_samples": [],
        }

        for img in imgs:
            img = torch.permute(torch.from_numpy(img), (2, 0, 1)).contiguous()
            batch_data['inputs'].append(img)
            data_sample = DetDataSample(metainfo={
                'img_shape': (img.shape[1], img.shape[2]),
            })
            gt_instance = InstanceData()
            data_sample.gt_instances = gt_instance

            batch_data['data_samples'].append(data_sample)

    def predict(self, imgs: List[np.ndarray], bgr: bool = False, confidence: float = 0.0, max_size: int = -1, device: str = 'cpu') -> List[PredictionResult]:
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

        with torch.no_grad():
            batch_data = self.imgs_to_batch(imgs)
            batch_data = torch_utils.batch_to_device(batch_data, device)
            forward_result = self.forward(batch_data)
            preds = self.postprocess(forward_result, batch_data, confidence)
            torch.cuda.empty_cache()
        
        for i in range(len(preds)):
            pred = preds[i]
            pred.bboxes = pred.bboxes / img_scales[i]
            # pred.bboxes[:, 0] = np.clip(pred.bboxes[:, 0], 0, imgs[i].shape[1])
            # pred.bboxes[:, 1] = np.clip(pred.bboxes[:, 1], 0, imgs[i].shape[0])
            # pred.bboxes[:, 2] = np.clip(pred.bboxes[:, 2], 0, imgs[i].shape[1])
            # pred.bboxes[:, 3] = np.clip(pred.bboxes[:, 3], 0, imgs[i].shape[0])

        return preds

    def compute_loss(self, batch_data: Dict, forward_result: Tuple) -> Tuple[torch.FloatTensor, Dict]:
        outputs = unpack_gt_instances(batch_data['data_samples'])
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs

        loss_inputs = forward_result + (batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)
        losses = self.model.bbox_head.loss_by_feat(*loss_inputs)
        loss, info = self.model.parse_losses(losses)
        info = {
            'loss': loss.item(),
            'cls': info['loss_cls'].item(),
            'box': info['loss_bbox'].item(),
            'giou': 0.0,
        }
        return loss, info

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
        super(HQRTMDET, self).to(device)
        self.model.to(device)