
import torch.nn
from ...common import PredictionResult
import numpy as np
import os
from .core import YAMLConfig
from typing import List
import cv2
from ... import torch_utils
import torchvision.transforms.functional as VF
import time
from ..base import HQModel

class HQRTDETR(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQRTDETR, self).__init__()

        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            self.id2names = data['id2names']
        else:
            self.id2names = class_id2names
            pass

        current_module_path = __file__
        config_path = os.path.join(
            os.path.dirname(current_module_path), 'configs', 'rtdetrv2', 'rtdetrv2_r50vd_m_7x_coco.yml')

        self.image_size = kwargs.get('image_size', 1024)
        print('=================================', kwargs)
        cfg = YAMLConfig(config_path)
        cfg.yaml_cfg['num_classes'] = len(self.id2names)
        cfg.yaml_cfg['remap_mscoco_category'] = False
        cfg.yaml_cfg['eval_spatial_size'] = [self.image_size, self.image_size]
        self.model = cfg.model
        self.criterion = cfg.criterion
        self.postprocessor = cfg.postprocessor
        self.load_model(kwargs['model'])
        self.device = 'cpu'

    def get_class_names(self):
        # Get the class names from the model
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
            pass
        return names

    def load_model(self, path):
        # Load the YOLO model using the specified path and device
        data = torch.load(path, map_location='cpu')
        state_dict = data['ema']['module']
        new_state_dict={k: v for k, v in state_dict.items() if state_dict[k].shape == self.model.state_dict()[k].shape}
        print(len(new_state_dict), len(state_dict))
        print([k for k in self.model.state_dict().keys() if k not in new_state_dict.keys()])
        self.model.load_state_dict(new_state_dict, strict=False)
        

    def extract_target(self, batch_data):
        # Extract the target data from the batch
        return batch_data['targets']
    
    def forward(self, batch_data):
        samples = batch_data['img']
        targets = self.extract_target(batch_data)
        forward_result = self.model(samples, targets)
        
        return forward_result
        
    
    def preprocess(self, batch_data):
        # Preprocess the input data for the YOLO model
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        # Post-process the predictions
        if 'origin_size' not in batch_data:
            origin_size = torch.stack([torch.tensor([batch_data['img'].shape[3], batch_data['img'].shape[2]]) for _ in range(batch_data['img'].shape[0])], 0)
            origin_size = origin_size.to(batch_data['img'].device)
            pass
        else:
            origin_size = batch_data['origin_size']

        preds = self.postprocessor(forward_result, origin_size)

        results = []

        for i, pred in enumerate(preds):
            record = PredictionResult()
            mask = pred["scores"] > confidence
            if not mask.any():
                # no pred
                record.bboxes = np.zeros((0, 4), dtype=np.float32)
                record.scores = np.zeros((0,), dtype=np.float32)
                record.cls = np.zeros((0,), dtype=np.int32)
                pass
            else:
                # add pred
                pred_bboxes = pred['boxes'][mask].cpu().numpy()
                pred_scores = pred['scores'][mask].cpu().numpy()
                pred_cls = pred['labels'][mask].cpu().numpy().astype(np.int32)
                record.bboxes = pred_bboxes
                record.cls = pred_cls
                record.scores = pred_scores
                record.image_id = batch_data['targets'][i]['image_id']
            results.append(record)
    
        return results


    def imgs_to_batch(self, imgs):
        # Convert a list of images to a batch

        # find max size of imgs
        max_h, max_w = self.image_size, self.image_size

        new_imgs = []
        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = VF.to_tensor(img)
            img, _ = torch_utils.pad_image(img, torch.zeros((0, 4)), (max_h, max_w))
            new_imgs.append(img)
            pass

        new_imgs = torch.stack(new_imgs, 0)

        targets = [
            {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int32),
                "image_id": 0,
            }
            for _ in range(len(imgs))
        ]

        return {
            "img": new_imgs,
            "targets": targets,
        }
        pass
    
    def predict(self, imgs: List[np.ndarray], bgr=False, confidence=0.0, max_size=-1) -> List[PredictionResult]:
        if not bgr:
            for i in range(len(imgs)):
                imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
                pass
            pass
        
        max_size = self.image_size
        imgs_ = []
        img_scales = np.ones((len(imgs),))
        if max_size > 0:
            for i in range(len(imgs)):
                max_hw = max(imgs[i].shape[0], imgs[i].shape[1])
                if max_hw > max_size:
                    rate = max_size / max_hw
                    imgs_.append(cv2.resize(imgs[i], (int(imgs[i].shape[1] * rate), int(imgs[i].shape[0] * rate))))
                    img_scales[i] = rate
                    pass
                else:
                    imgs_.append(imgs[i])
                    pass
                pass
            pass

        original_shapes = []
        for img in imgs_:
            original_shapes.append(img.shape)
            pass
        device = self.device
        start = time.time()
        with torch.no_grad():
            batch_data = self.imgs_to_batch(imgs_)
            batch_data = torch_utils.batch_to_device(batch_data, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                forward_result = self.forward(batch_data)
                preds = self.postprocess(forward_result, batch_data, confidence)
        

        for i in range(len(preds)):
            pred = preds[i]
            # pred.bboxes[:, 0] = pred.bboxes[:, 0] / self.image_size * original_shapes[i][1]
            # pred.bboxes[:, 1] = pred.bboxes[:, 1] / self.image_size * original_shapes[i][0]
            # pred.bboxes[:, 2] = pred.bboxes[:, 2] / self.image_size * original_shapes[i][1]
            # pred.bboxes[:, 3] = pred.bboxes[:, 3] / self.image_size * original_shapes[i][0]
            pred.bboxes = pred.bboxes / img_scales[i]
            pass

        # print(f"predict time: {time.time() - start}")

        return preds

        pass

    def compute_loss(self, batch_data, forward_result):
        # Compute the loss using the YOLO model
        # This is a placeholder; actual implementation may vary
        
        if True:
        #if self.training:
            target = self.extract_target(batch_data)
            loss_dict = self.criterion(forward_result, target)

            loss = sum(loss_dict.values())
            info = {
                'box': loss_dict['loss_bbox'].item(),
                'giou': loss_dict['loss_giou'].item(),
                'cls': loss_dict['loss_vfl'].item(),
            }
        else:
            loss = torch.tensor(0.0, device=forward_result.device)
            info = {
                'vfl': 0.0,
                'bbox': 0.0,
                'giou': 0.0,
            }
        return loss, info

    def save(self, path):
        # Save the YOLO model to the specified path
        
        torch.save(
            {
                'ema': {
                    'module': self.model.state_dict()
                },
                'id2names': self.id2names,
            }, path + '.pth')
        pass


    def to(self, device):
        super(HQRTDETR, self).to(device)
        self.device = torch.device(device)

    pass
