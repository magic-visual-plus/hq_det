from mmengine import MODELS, Config
import torch.nn
import torch.nn.functional as F
from detectron2.config import LazyConfig, instantiate
from ...common import PredictionResult
from ..base import HQModel
import numpy as np
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
import cv2
from typing import List
from ... import torch_utils, box_utils
import detectron2.structures
import os


class HQDINO_EVA(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQDINO_EVA, self).__init__(class_id2names, **kwargs)
        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            class_names = data['meta']['dataset_meta']['CLASSES']
            self.id2names = {i: name for i, name in enumerate(class_names)}
        else:
            self.id2names = class_id2names

        current_dir = os.path.dirname(os.path.abspath(__file__))
        dino_eva_config_path = os.path.join(
            current_dir, 'configs', 'dino-eva-02',
            'dino_eva_02_vitdet_b_4attn_1024_lrd0p7_4scale_12ep.py')
        dino_eva_config = Config.fromfile(dino_eva_config_path)
        dino_eva_config = LazyConfig.load(dino_eva_config_path)
        # print(LazyConfig.to_py(dino_eva_config)) #查看配置结构
        # print(dino_eva_config)
        dino_eva_config.model.num_classes = len(self.id2names)
        # dino_eva_config.model['roi_heads']['box_predictor']['num_classes'] = len(self.id2names)
        # dino_eva_config.model['roi_heads']['mask_head']['num_classes'] = len(self.id2names)
        # dino_eva_config.model['roi_heads']['num_classes'] = len(self.id2names)
        # self.model = MODELS.build(dino_eva_config.model)
        self.model = instantiate(dino_eva_config.model)
        self.load_model(kwargs['model'])
        self.device = torch.device('cpu')

    def get_class_names(self):
        # Get the class names from the model
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v

        return names

    def load_model(self, path):
        # Load the YOLO model using the specified path and device
        data = torch.load(path, map_location='cpu')
        # print("权重文件的所有键：", list(data['model'].keys()))
        state_dict = data['model']
        new_state_dict={k: v for k, v in state_dict.items() if state_dict[k].shape == self.model.state_dict()[k].shape}
        print(len(new_state_dict), len(state_dict))
        print([k for k in self.model.state_dict().keys() if k not in new_state_dict.keys()])
        self.model.load_state_dict(new_state_dict, strict=False)


    def forward(self, batch_data):
        batch_data = batch_data['inputs']
        if self.model.training:
            loss_dict = self.model.forward(batch_data)
            return loss_dict
        else:
            return_dict = self.model.forward(batch_data)
            return return_dict
        pass
    
    def preprocess(self, batch_data):
        # Preprocess the input data for the YOLO model
        pass

    def postprocess(self, forward_result, batch_data, confidence=0.0):
        # Post-process the predictions

        batch_size = len(forward_result)
        results = []
        for i in range(batch_size):
            instances = forward_result[i]['instances']
            pred_boxes = instances.pred_boxes.tensor
            pred_scores = instances.scores
            pred_classes = instances.pred_classes

            record = PredictionResult()

            mask = pred_scores > confidence
            if not mask.any():
                # no pred
                record.bboxes = np.zeros((0, 4), dtype=np.float32)
                record.scores = np.zeros((0,), dtype=np.float32)
                record.cls = np.zeros((0,), dtype=np.int32)
                pass
            else:
                # add pred
                bboxes = pred_boxes[mask].cpu().numpy()
                scores = pred_scores[mask].cpu().numpy()
                cls = pred_classes[mask].cpu().numpy()
                record.bboxes = bboxes
                record.scores = scores
                record.cls = cls
                pass
            results.append(record)
        return results
        
    @classmethod
    def imgs_to_batch(cls, imgs, boxes=None, labels=None):
        '''
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instances"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.
        '''
        batch_size = len(imgs)
        batch_data = []
        for i in range(batch_size):
            data = {}
            img = imgs[i]
            data['image'] = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
            data['height'] = img.shape[0]
            data['width'] = img.shape[1]

            instance = detectron2.structures.Instances(
                image_size=(img.shape[0], img.shape[1])
            )
            if boxes is not None and labels is not None:
                boxes_i = boxes[i]
                labels_i = labels[i]
                if len(boxes_i) == 0:
                    boxes_i = torch.zeros((0, 4), dtype=torch.float32)
                    labels_i = torch.zeros((0,), dtype=torch.int64)
                    pass
                pass
            else:
                boxes_i = torch.zeros((0, 4), dtype=torch.float32)
                labels_i = torch.zeros((0,), dtype=torch.int64)
                pass
            instance.gt_boxes = detectron2.structures.Boxes(boxes_i)
            instance.gt_classes = labels_i
            data['instances'] = instance

            batch_data.append(data)
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
            batch_data = HQDINO_EVA.imgs_to_batch(imgs)
            batch_data = torch_utils.batch_to_device(batch_data, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=False):
                forward_result = self.forward({"inputs": batch_data})
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
        if self.model.training:
            loss_dict = forward_result
            loss = sum(loss_dict.values())
            info = {
                "class": loss_dict["loss_class"].item(),
                "bbox": loss_dict["loss_bbox"].item(),
                "giou": loss_dict["loss_giou"].item(),
            }
            return loss, info
        else:
            return torch.tensor(0, device=self.device), dict()


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
            }, path)
        pass

    def to(self, device):
        super(HQDINO_EVA, self).to(device)
        self.model.to(device)
        self.device = torch.device(device)
    pass