import os
import sys
from typing import List

import cv2
import numpy as np
import torch
import torch.nn
import torchvision.transforms.functional as VF
from detectron2.config import LazyConfig, instantiate

from ... import torch_utils
from ...common import PredictionResult
from ..base import HQModel

detrex_path = os.path.join(os.path.dirname(__file__), 'detrex')
if os.path.exists(detrex_path) and detrex_path not in sys.path:
    sys.path.insert(0, detrex_path)


class HQDetrex(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQDetrex, self).__init__()

        if class_id2names is None:
            if 'model' in kwargs and kwargs['model']:
                data = torch.load(kwargs['model'], map_location='cpu')
                self.id2names = data.get('id2names', {})
                self.image_size = kwargs.get('image_size', data.get('image_size', 1024))
            else:
                raise ValueError("Either class_id2names or model checkpoint must be provided")
        else:
            self.id2names = class_id2names
            self.image_size = kwargs.get('image_size', 1024)

        current_module_path = __file__
        config_path = kwargs.get('config_path', None)
        
        if config_path is None:
            project = kwargs.get('project', 'dino')
            model_size = kwargs.get('model_size', 'r50')
            model_name = kwargs.get('model_name', None)
            
            if model_name is None:
                if project == 'dino':
                    if model_size.startswith('r'):
                        model_name = f'dino_{model_size}_4scale_12ep'
                        config_dir = 'dino-resnet'
                    elif 'swin' in model_size:
                        model_name = f'dino_{model_size}_4scale_12ep'
                        config_dir = 'dino-swin'
                    elif 'convnext' in model_size:
                        model_name = f'dino_{model_size}_4scale_12ep'
                        config_dir = 'dino-convnext'
                    else:
                        model_name = f'dino_{model_size}_4scale_12ep'
                        config_dir = 'dino-resnet'
                else:
                    config_dir = project
                    model_name = f'{project}_{model_size}'
            else:
                if 'dino' in model_name:
                    if 'swin' in model_name:
                        config_dir = 'dino-swin'
                    elif 'convnext' in model_name:
                        config_dir = 'dino-convnext'
                    elif 'eva' in model_name:
                        config_dir = 'dino-eva-01'
                    else:
                        config_dir = 'dino-resnet'
                else:
                    config_dir = project if 'project' in kwargs else 'dino-resnet'
            
            config_path = os.path.join(
                os.path.dirname(current_module_path), 'detrex', 
                'projects', project, 'configs', config_dir, f'{model_name}.py')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        print(f"Loading detrex config from: {config_path}")
        cfg = LazyConfig.load(config_path)
        
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'num_classes'):
            cfg.model.num_classes = len(self.id2names)
        elif 'model' in cfg and 'num_classes' in cfg.model:
            cfg.model.num_classes = len(self.id2names)
        
        if 'model' in kwargs and kwargs['model']:
            if hasattr(cfg, 'train') and hasattr(cfg.train, 'init_checkpoint'):
                cfg.train.init_checkpoint = ""
            elif 'train' in cfg and 'init_checkpoint' in cfg.train:
                cfg.train.init_checkpoint = ""
        
        self.model = instantiate(cfg.model)
        self.cfg = cfg
        self.device = torch.device('cpu')
        
        if 'model' in kwargs and kwargs['model']:
            self.load_model(kwargs['model'])

    def get_class_names(self):
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
        return names

    def load_model(self, path):
        data = torch.load(path, map_location='cpu')
        if 'model' in data:
            state_dict = data['model']
        elif 'state_dict' in data:
            state_dict = data['state_dict']
        elif 'model_state_dict' in data:
            state_dict = data['model_state_dict']
        else:
            state_dict = data
        
        model_state_dict = self.model.state_dict()
        new_state_dict = {k: v for k, v in state_dict.items() 
                         if k in model_state_dict and v.shape == model_state_dict[k].shape}
        
        print(f"Loaded {len(new_state_dict)}/{len(state_dict)} parameters from checkpoint")
        missing_keys = [k for k in model_state_dict if k not in new_state_dict]
        if missing_keys:
            msg = f"Missing keys: {missing_keys[:10]}..." if len(missing_keys) > 10 else f"Missing keys: {missing_keys}"
            print(msg)
        
        self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, batch_data):
        if isinstance(batch_data, dict):
            if 'img' in batch_data:
                images = batch_data['img']
                targets = batch_data.get('targets', None)
            else:
                return self.model(batch_data)
        else:
            return self.model(batch_data)
        
        from detectron2.structures import ImageList
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                image_list = ImageList.from_tensors([images[i] for i in range(images.shape[0])], 
                                                     self.model.backbone.size_divisibility)
            else:
                image_list = ImageList.from_tensors(images if isinstance(images, (list, tuple)) else [images],
                                                     self.model.backbone.size_divisibility)
        else:
            image_list = images
        
        batched_inputs = []
        if hasattr(image_list, 'tensor'):
            batch_size = image_list.tensor.shape[0]
        elif hasattr(image_list, 'image_sizes'):
            batch_size = len(image_list.image_sizes)
        else:
            batch_size = images.shape[0] if isinstance(images, torch.Tensor) else len(images)
        
        for i in range(batch_size):
            if hasattr(image_list, 'tensor'):
                img_tensor = image_list.tensor[i]
            else:
                img_tensor = image_list[i] if isinstance(image_list, (list, tuple)) else images[i]
            
            if hasattr(image_list, 'image_sizes') and i < len(image_list.image_sizes):
                img_h, img_w = image_list.image_sizes[i]
            else:
                img_h, img_w = self.image_size, self.image_size
            
            input_dict = {
                "image": img_tensor,
                "height": int(img_h),
                "width": int(img_w),
            }
            
            if targets is not None and i < len(targets):
                from detectron2.structures import Instances, Boxes
                target = targets[i]
                instances = Instances((img_h, img_w))
                
                if 'boxes' in target:
                    boxes = target['boxes']
                    if isinstance(boxes, torch.Tensor):
                        if boxes.numel() == 0:
                            instances.gt_boxes = Boxes(boxes)
                        else:
                            if boxes.shape[-1] == 4:
                                if boxes.numel() > 0 and boxes.max().item() <= 1.0:
                                    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
                                if boxes.numel() > 0:
                                    cx, cy, w, h = boxes.unbind(-1)
                                    boxes = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
                            instances.gt_boxes = Boxes(boxes)
                    else:
                        instances.gt_boxes = Boxes(boxes)
                
                if 'labels' in target:
                    instances.gt_classes = target['labels']
                
                input_dict["instances"] = instances
            
            batched_inputs.append(input_dict)
        
        return self.model(batched_inputs)

    def preprocess(self, batch_data):
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        from detectron2.modeling import detector_postprocess
        from detectron2.structures import Instances
        
        if 'origin_size' not in batch_data:
            h, w = batch_data['img'].shape[2], batch_data['img'].shape[3]
            origin_size = torch.tensor([[w, h]] * batch_data['img'].shape[0], 
                                       device=batch_data['img'].device)
        else:
            origin_size = batch_data['origin_size']
        
        results = []
        instances_list = []
        if isinstance(forward_result, (list, tuple)):
            if len(forward_result) > 0 and isinstance(forward_result[0], dict) and 'instances' in forward_result[0]:
                instances_list = [item['instances'] for item in forward_result]
            else:
                instances_list = forward_result
        elif isinstance(forward_result, dict):
            if 'instances' in forward_result:
                instances_list = [forward_result['instances']]
            else:
                return results
        else:
            instances_list = [forward_result] if not isinstance(forward_result, list) else forward_result
        
        for i, instances in enumerate(instances_list):
            if isinstance(instances, Instances):
                h, w = origin_size[i].tolist()
                instances = detector_postprocess(instances, int(h), int(w))
                
                record = PredictionResult()
                if hasattr(instances, 'scores'):
                    mask = instances.scores > confidence
                    if mask.any():
                        record.bboxes = instances.pred_boxes.tensor[mask].cpu().numpy()
                        record.scores = instances.scores[mask].cpu().numpy()
                        record.cls = instances.pred_classes[mask].cpu().numpy().astype(np.int32)
                    else:
                        record.bboxes = np.zeros((0, 4), dtype=np.float32)
                        record.scores = np.zeros((0,), dtype=np.float32)
                        record.cls = np.zeros((0,), dtype=np.int32)
                else:
                    record.bboxes = np.zeros((0, 4), dtype=np.float32)
                    record.scores = np.zeros((0,), dtype=np.float32)
                    record.cls = np.zeros((0,), dtype=np.int32)
                
                if 'targets' in batch_data and i < len(batch_data['targets']):
                    record.image_id = batch_data['targets'][i].get('image_id', i)
                else:
                    record.image_id = i
                results.append(record)
        
        return results

    def imgs_to_batch(self, imgs):
        max_size = (self.image_size, self.image_size)
        new_imgs = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = VF.to_tensor(img)
            img, _ = torch_utils.pad_image(img, torch.zeros((0, 4)), max_size)
            new_imgs.append(img)
        
        empty_target = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "labels": np.zeros((0,), dtype=np.int32),
            "image_id": 0,
        }
        return {
            "img": torch.stack(new_imgs, 0),
            "targets": [empty_target.copy() for _ in range(len(imgs))],
        }

    def predict(self, imgs: List[np.ndarray], bgr=False, confidence=0.0, max_size=-1) -> List[PredictionResult]:
        if not bgr:
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
        
        max_size = self.image_size if max_size <= 0 else max_size
        imgs_, img_scales = [], np.ones((len(imgs),))
        for i, img in enumerate(imgs):
            max_hw = max(img.shape[0], img.shape[1])
            if max_hw > max_size:
                rate = max_size / max_hw
                imgs_.append(cv2.resize(img, (int(img.shape[1] * rate), int(img.shape[0] * rate))))
                img_scales[i] = rate
            else:
                imgs_.append(img)

        with torch.no_grad():
            batch_data = torch_utils.batch_to_device(self.imgs_to_batch(imgs_), self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=False):
                forward_result = self.forward(batch_data)
                preds = self.postprocess(forward_result, batch_data, confidence)
        
        for i, pred in enumerate(preds):
            pred.bboxes = pred.bboxes / img_scales[i]
        return preds

    def compute_loss(self, batch_data, forward_result):
        if isinstance(forward_result, dict):
            loss_dict = forward_result
        else:
            loss_dict = forward_result if isinstance(forward_result, dict) else {}
        
        if not hasattr(self, '_loss_keys_printed'):
            print(f"Available loss keys: {list(loss_dict.keys())}")
            self._loss_keys_printed = True

        def get_value(key):
            v = loss_dict.get(key)
            return v.item() if isinstance(v, torch.Tensor) else (v if v is not None else 0.0)
        
        info = {
            'box': get_value('loss_bbox'),
            'giou': get_value('loss_giou'),
            'cls': get_value('loss_class'),
        }
        
        for key in loss_dict.keys():
            if key.startswith('loss_') and key not in ['loss_bbox', 'loss_giou', 'loss_class']:
                info[key.replace('loss_', '')] = get_value(key)
        
        total_loss = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0)
        return total_loss, info

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'id2names': self.id2names,
            'image_size': self.image_size,
        }, path)

    def to(self, device):
        super(HQDetrex, self).to(device)
        self.device = torch.device(device)
        if hasattr(self, 'model'):
            self.model = self.model.to(device)


