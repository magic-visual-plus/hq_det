import torch
import torch.nn
from ...common import PredictionResult
import numpy as np
import os
import sys
from typing import List
import cv2
from ... import torch_utils
import torchvision.transforms.functional as VF
from ..base import HQModel

deim_path = os.path.join(os.path.dirname(__file__), 'DEIM')
if os.path.exists(deim_path) and deim_path not in sys.path:
    sys.path.insert(0, deim_path)

from engine.core import YAMLConfig

class HQDEIM(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQDEIM, self).__init__()

        if class_id2names is None:
            # load from model file
            data = torch.load(kwargs['model'], map_location='cpu')
            self.id2names = data['id2names']
            
            self.image_size = kwargs.get('image_size', data.get('image_size', 1024))
        else:
            self.id2names = class_id2names
            self.image_size = kwargs.get('image_size', 1024)

        current_module_path = __file__
        config_path = kwargs.get('config_path', None)
        if config_path is None:
            # Support series parameter: 'dfine' or 'rtdetrv2' (default: 'dfine')
            series = kwargs.get('series', 'dfine')
            # Support model_size parameter: 
            #   For dfine: 'n', 's', 'm', 'l', 'x'
            #   For rtdetrv2: 'r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd'
            model_size = kwargs.get('model_size', 's' if series == 'dfine' else 'r50vd')
            # Support model_type parameter: 'deim' or 'dfine'/'rtdetrv2' (default: 'deim')
            model_type = kwargs.get('model_type', 'deim')
            # Support model_name parameter: full config name (overrides above)
            model_name = kwargs.get('model_name', None)
            
            if model_name is None:
                if series == 'dfine':
                    # Build config filename for dfine series
                    if model_type == 'deim':
                        model_name = f'deim_hgnetv2_{model_size}_coco'
                    else:  # dfine
                        model_name = f'dfine_hgnetv2_{model_size}_coco'
                    config_dir = 'deim_dfine'
                else:  # rtdetrv2
                    # Build config filename for rtdetrv2 series
                    if model_type == 'deim':
                        # Map model_size to rtdetrv2 naming
                        size_map = {
                            'r18vd': 'r18vd_120e',
                            'r34vd': 'r34vd_120e',
                            'r50vd': 'r50vd_60e',
                            'r50vd_m': 'r50vd_m_60e',
                            'r101vd': 'r101vd_60e',
                        }
                        suffix = size_map.get(model_size, 'r50vd_60e')
                        model_name = f'deim_{suffix}_coco'
                    else:  # rtdetrv2
                        size_map = {
                            'r18vd': 'r18vd_120e',
                            'r34vd': 'r34vd_120e',
                            'r50vd': 'r50vd_6x',
                            'r50vd_m': 'r50vd_m_7x',
                            'r101vd': 'r101vd_6x',
                        }
                        suffix = size_map.get(model_size, 'r50vd_6x')
                        model_name = f'rtdetrv2_{suffix}_coco'
                    config_dir = 'deim_rtdetrv2'
            else:
                # Determine config_dir from model_name
                if 'hgnetv2' in model_name:
                    config_dir = 'deim_dfine'
                elif 'r' in model_name and ('vd' in model_name or 'rtdetrv2' in model_name):
                    config_dir = 'deim_rtdetrv2'
                else:
                    # Default to dfine if cannot determine
                    config_dir = 'deim_dfine'
            
            config_path = os.path.join(
                os.path.dirname(current_module_path), 'DEIM', 
                'configs', config_dir, f'{model_name}.yml')

        print('=================================', kwargs)
        cfg = YAMLConfig(config_path)
        cfg.yaml_cfg['num_classes'] = len(self.id2names)
        cfg.yaml_cfg['remap_mscoco_category'] = False
        cfg.yaml_cfg['eval_spatial_size'] = [self.image_size, self.image_size]
        
        if 'model' in kwargs and kwargs['model']:
            if 'HGNetv2' in cfg.yaml_cfg:
                cfg.yaml_cfg['HGNetv2']['pretrained'] = False
            if 'PResNet' in cfg.yaml_cfg:
                cfg.yaml_cfg['PResNet']['pretrained'] = False
        
        self.model = cfg.model
        self.criterion = cfg.criterion
        self.postprocessor = cfg.postprocessor
        if 'model' in kwargs and kwargs['model']:
            self.load_model(kwargs['model'])
        self.device = 'cpu'

    def get_class_names(self):
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
        return names

    def load_model(self, path):
        data = torch.load(path, map_location='cpu')
        # Handle different checkpoint formats
        if 'ema' in data and 'module' in data['ema']:
            state_dict = data['ema']['module']
        elif 'model' in data:
            state_dict = data['model']
        elif 'state_dict' in data:
            state_dict = data['state_dict']
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
        

    def extract_target(self, batch_data):
        return batch_data['targets']
    
    def forward(self, batch_data):
        targets = batch_data.get('targets')
        if targets is not None and not self.training:
            was_training = self.model.training
            self.model.train()
            try:
                result = self.model(batch_data['img'], targets)
            finally:
                self.model.train(was_training)
            return result
        return self.model(batch_data['img'], targets)
    
    def preprocess(self, batch_data):
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        if 'origin_size' not in batch_data:
            h, w = batch_data['img'].shape[2], batch_data['img'].shape[3]
            origin_size = torch.tensor([[w, h]] * batch_data['img'].shape[0], 
                                       device=batch_data['img'].device)
        else:
            origin_size = batch_data['origin_size']

        preds = self.postprocessor(forward_result, origin_size)
        results = []
        for i, pred in enumerate(preds):
            record = PredictionResult()
            mask = pred["scores"] > confidence
            if mask.any():
                record.bboxes = pred['boxes'][mask].cpu().numpy()
                record.scores = pred['scores'][mask].cpu().numpy()
                record.cls = pred['labels'][mask].cpu().numpy().astype(np.int32)
            else:
                record.bboxes = np.zeros((0, 4), dtype=np.float32)
                record.scores = np.zeros((0,), dtype=np.float32)
                record.cls = np.zeros((0,), dtype=np.int32)
            record.image_id = batch_data['targets'][i]['image_id']
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
        
        max_size = self.image_size
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
        loss_dict = self.criterion(forward_result, batch_data['targets'])
        
        if not hasattr(self, '_loss_keys_printed'):
            print(f"Available loss keys: {list(loss_dict.keys())}")
            self._loss_keys_printed = True

        def get_value(key):
            v = loss_dict.get(key)
            return v.item() if isinstance(v, torch.Tensor) else (v if v is not None else 0.0)
        
        info = {
            'box': get_value('loss_bbox'),
            'giou': get_value('loss_giou'),
            'cls': get_value('loss_vfl'),
        }
        for key, name in [('loss_fgl', 'fgl'), ('loss_ddf', 'ddf'), ('loss_mal', 'mal')]:
            if key in loss_dict:
                info[name] = get_value(key)
        
        return sum(loss_dict.values()), info

    def save(self, path):
        torch.save({
            'ema': {'module': self.model.state_dict()},
            'id2names': self.id2names,
            'image_size': self.image_size,
        }, path)

    def to(self, device):
        super(HQDEIM, self).to(device)
        self.device = torch.device(device)
