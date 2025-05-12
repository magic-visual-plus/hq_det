from mmdet.configs.dino import dino_4scale_r50_8xb2_12e_coco as dino_config
from mmengine import MODELS, Config
import torch.nn
from ...common import PredictionResult
from ..base import HQModel
import numpy as np
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
import cv2
from typing import List
from ... import torch_utils
import os


class HQDINO(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQDINO, self).__init__(class_id2names, **kwargs)
        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            class_names = data['meta']['dataset_meta']['CLASSES']
            self.id2names = {i: name for i, name in enumerate(class_names)}
        else:
            self.id2names = class_id2names
            pass

        current_dir = os.path.dirname(os.path.abspath(__file__))
        dino_config_path = os.path.join(current_dir, 'configs', 'dino_4scale_r50_8xb2_12e_coco.py')
        dino_config = Config.fromfile(dino_config_path)
        dino_config.model['bbox_head']['num_classes'] = len(self.id2names)
        self.model = MODELS.build(dino_config.model)
        self.load_model(kwargs['model'])
        self.device = 'cpu'

    def get_class_names(self):
        # Get the class names from the model
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v

        return names

    def load_model(self, path):
        # Load the YOLO model using the specified path and device
        data = torch.load(path, map_location='cpu')
        new_state_dict={k: v for k, v in data['state_dict'].items() if data['state_dict'][k].shape == self.model.state_dict()[k].shape}
        print(len(new_state_dict), len(data['state_dict']))
        self.model.load_state_dict(new_state_dict, strict=False)


    def forward(self, batch_data):
        batch_data.update(self.model.data_preprocessor(batch_data, self.training))
        inputs = batch_data['inputs']
        data_samples = batch_data['data_samples']
        img_feats = self.model.extract_feat(inputs)
        head_inputs_dict = self.model.forward_transformer(
            img_feats, data_samples)
        
        return {
            'img_feats': img_feats,
            'head_inputs_dict': head_inputs_dict,
        }
        pass
    
    def preprocess(self, batch_data):
        # Preprocess the input data for the YOLO model
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        # Post-process the predictions
        head_inputs_dict = forward_result['head_inputs_dict']

        preds = self.model.bbox_head.predict(
            head_inputs_dict['hidden_states'],
            head_inputs_dict['references'],
            batch_data_samples = batch_data['data_samples'],
            rescale=False)

        results = []
        for pred in preds:
            record = PredictionResult()
            mask = pred.scores > confidence
            if not mask.any():
                # no pred
                record.bboxes = np.zeros((0, 4), dtype=np.float32)
                record.scores = np.zeros((0,), dtype=np.float32)
                record.cls = np.zeros((0,), dtype=np.int32)
                pass
            else:
                # add pred
                pred_bboxes = pred.bboxes[mask].cpu().numpy()
                pred_scores = pred.scores[mask].cpu().numpy()
                pred_cls = pred.labels[mask].cpu().numpy().astype(np.int32)
                record.bboxes = pred_bboxes
                record.scores = pred_scores
                record.cls = pred_cls
                pass
            results.append(record)
        return results
        

    def imgs_to_batch(self, imgs):
        # Convert a list of images to a batch

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
            pass

        return batch_data
        pass

    def predict(self, imgs, bgr=False, confidence=0.0, max_size=-1) -> List[PredictionResult]:
        if not bgr:
            # Convert RGB to BGR
            for i in range(len(imgs)):
                imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
                pass
            pass

        img_scales = np.ones((len(imgs),))
        if max_size > 0:
            for i in range(len(imgs)):
                max_hw = max(imgs[i].shape[0], imgs[i].shape[1])
                if max_hw > max_size:
                    rate = max_size / max_hw
                    imgs[i] = cv2.resize(imgs[i], (int(imgs[i].shape[1] * rate), int(imgs[i].shape[0] * rate)))
                    img_scales[i] = rate
                    pass
                pass
            pass
        device = self.device
        with torch.no_grad():
            batch_data = self.imgs_to_batch(imgs)
            batch_data = torch_utils.batch_to_device(batch_data, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                forward_result = self.forward(batch_data)
                preds = self.postprocess(forward_result, batch_data, confidence)
                pass
            pass
        
        for i in range(len(preds)):
            pred = preds[i]
            pred.bboxes = pred.bboxes / img_scales[i]
            # pred.bboxes[:, 0] = np.clip(pred.bboxes[:, 0], 0, imgs[i].shape[1])
            # pred.bboxes[:, 1] = np.clip(pred.bboxes[:, 1], 0, imgs[i].shape[0])
            # pred.bboxes[:, 2] = np.clip(pred.bboxes[:, 2], 0, imgs[i].shape[1])
            # pred.bboxes[:, 3] = np.clip(pred.bboxes[:, 3], 0, imgs[i].shape[0])
            pass

        return preds

    def compute_loss(self, batch_data, forward_result):
        # Compute the loss using the YOLO model
        # This is a placeholder; actual implementation may vary
        head_inputs_dict = forward_result['head_inputs_dict']
        data_samples = batch_data['data_samples']
        if self.model.training:

            losses = self.model.bbox_head.loss(
                **head_inputs_dict, batch_data_samples=data_samples)
            
            loss, info = self.model.parse_losses(losses)

            info = {
                'loss': loss.item(),
                'cls': info['loss_cls'].item(),
                'box': info['loss_bbox'].item(),
                'giou': info['loss_iou'].item(),
            }
        else:
            loss = torch.tensor(0.0, device=head_inputs_dict['hidden_states'].device)
            info = {
                'loss': 0.0,
                'cls': 0.0,
                'box': 0.0,
                'giou': 0.0,
            }
            pass
        return loss, info

    def save(self, path):
        # Save the YOLO model to the specified path
        
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'meta': {
                    'dataset_meta': {
                        'CLASSES': self.get_class_names(),
                    },
                }
            }, path + '.pth')
        pass

    def to(self, device):
        super(HQDINO, self).to(device)
        self.model.to(device)
        self.device = torch.device(device)
    pass