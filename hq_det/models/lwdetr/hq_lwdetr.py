import yaml

import torch.nn
from ...common import PredictionResult
import numpy as np
import os
from typing import List
import cv2
from ... import torch_utils
import torchvision.transforms.functional as VF
import time

from .configs.config_loader import ConfigLoader
from .util.utils import ModelEma, BestMetricHolder, clean_state_dict
import hq_det.models.lwdetr.util.misc as utils
from .util.get_param_dicts import get_param_dict

# from engine import evaluate, train_one_epoch
try:
    from hq_det.models.lwdetr.models import build_model
except:
    import os
    import subprocess
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ops_dir = os.path.join(current_dir, 'models', 'ops')

    # Change to ops directory and run setup.py
    try:
        subprocess.run(['python', 'setup.py', 'build', 'install'], 
                      cwd=ops_dir, 
                      check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error compiling ops: {e}")
        raise
    from hq_det.models.lwdetr.models import build_model

from ..base import HQModel

class HQLWDETR(HQModel):
    def __init__(self, class_id2names=None, **kwargs):
        super(HQLWDETR, self).__init__()

        if class_id2names is None:
            data = torch.load(kwargs['model'], map_location='cpu')
            self.id2names = data['id2names']
        else:
            self.id2names = class_id2names
        model_name = kwargs['model_name']
        model_dir = os.path.join(os.path.dirname(__file__), 'configs')
        config_loader = ConfigLoader(config_dir=model_dir)
        self.model_args = config_loader.get_args(model_name)
        for k, v in kwargs.items():
            setattr(self.model_args, k, v)
        
        utils.init_distributed_mode(self.model_args)
        print(self.model_args)

        self.model, self.criterion, self.postprocessors = build_model(self.model_args)
        self.load_model()
        # print(self.model)
        # print(self.criterion)
        # print(self.postprocessors)
        # exit()
        self.image_size = kwargs.get('image_size', 1024)
        # self.device = 'cpu'

    def get_class_names(self):
        # Get the class names from the model
        names = ['' for _ in range(len(self.id2names))]
        for k, v in self.id2names.items():
            names[k] = v
            pass
        return names

    def load_model(self):

        self.model.to(self.model_args.device)
        if self.model_args.use_ema:
            self.ema_m = ModelEma(self.model, decay=self.model_args.ema_decay)
        else:
            self.ema_m = None
        self.model_without_ddp = self.model
        if self.model_args.distributed:
            if self.model_args.sync_bn:
                # 转换为同步BatchNorm
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            # 使用DistributedDataParallel进行分布式训练
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.model_args.gpu])
            self.model_without_ddp = self.model.module
        # 计算模型参数量
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        if self.model_args.pretrain_weights is not None:
            checkpoint = torch.load(self.model_args.pretrain_weights, map_location='cpu')
            # 支持排除某些键
            # 例如,加载object365预训练时,不加载`class_embed.[weight, bias]`
            if self.model_args.pretrain_exclude_keys is not None:
                assert isinstance(self.model_args.pretrain_exclude_keys, list)
                for exclude_key in self.model_args.pretrain_exclude_keys:
                    checkpoint['model'].pop(exclude_key)
            if self.model_args.pretrain_keys_modify_to_load is not None:
                from hq_det.models.lwdetr.util.obj365_to_coco_model import get_coco_pretrain_from_obj365
                assert isinstance(self.model_args.pretrain_keys_modify_to_load, list)
                for modify_key_to_load in self.model_args.pretrain_keys_modify_to_load:
                    checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                        self.model_without_ddp.state_dict()[modify_key_to_load],
                        checkpoint['model'][modify_key_to_load]
                    )
            self.model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            if self.model_args.use_ema:
                del self.ema_m
            ema_m = ModelEma(self.model_without_ddp)
        # 模型恢复
        if self.model_args.resume:
            checkpoint = torch.load(self.model_args.resume, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'], strict=True)
            if self.model_args.use_ema:
                if 'ema_model' in checkpoint:
                    self.ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
                else:
                    del ema_m
                    ema_m = ModelEma(self.model)     


    def extract_target(self, batch_data):
        # Extract the target data from the batch
        return batch_data['targets']
    
    def forward(self, batch_data):
        samples = batch_data['img']
        targets = self.extract_target(batch_data)
        print(samples.shape)
        print(targets)
        forward_result = self.model(samples, targets)
        print(forward_result)
        exit()
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
            if len(pred['labels']) == 0:
                pass
            else:
                # add pred
                pred_bboxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_cls = pred['labels'].cpu().numpy().astype(np.int32)
                record.bboxes = pred_bboxes
                record.cls = pred_cls
                record.scores = pred_scores
                record.image_id = batch_data['targets'][i]['image_id']
                pass
            results.append(record)
            pass
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
        img_scales = np.ones((len(imgs),))
        if max_size > 0:
            for i in range(len(imgs)):
                max_hw = max(imgs[i].shape[0], imgs[i].shape[1])
                if max_hw > max_size:
                    rate = max_size / max_hw
                    imgs[i] = cv2.resize(imgs[i], (int(imgs[i].shape[1] * rate), int(imgs[i].shape[0] * rate)))
                    img_scales[i] = rate

        original_shapes = []
        for img in imgs:
            original_shapes.append(img.shape)
            pass

        start = time.time()
        with torch.no_grad():
            batch_data = self.imgs_to_batch(imgs)
            batch_data = torch_utils.batch_to_device(batch_data, self.device)
            forward_result = self.forward(batch_data)
            preds = self.postprocess(forward_result, batch_data, confidence)
            pass
        

        for i in range(len(preds)):
            pred = preds[i]
            pred.bboxes[:, 0] = pred.bboxes[:, 0] / self.image_size * original_shapes[i][1]
            pred.bboxes[:, 1] = pred.bboxes[:, 1] / self.image_size * original_shapes[i][0]
            pred.bboxes[:, 2] = pred.bboxes[:, 2] / self.image_size * original_shapes[i][1]
            pred.bboxes[:, 3] = pred.bboxes[:, 3] / self.image_size * original_shapes[i][0]
            pred.bboxes = pred.bboxes / img_scales[i]
            pass

        # print(f"predict time: {time.time() - start}")

        return preds

        pass

    def get_param_dict(self, args):
        model_without_ddp = self.model_without_ddp
        self.model_args.lr = args.lr0
        param_dicts = get_param_dict(self.model_args, model_without_ddp)
        return param_dicts
    
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
        super(HQLWDETR, self).to(device)
        self.device = device if torch.cuda.is_available() else 'cpu'

    pass