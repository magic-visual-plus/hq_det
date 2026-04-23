import sys
from hq_det.models import dino_eva
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det import augment
import os
import torch
import torch.optim
from hq_det import torch_utils
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from ..models.base import HQModel
from typing import Tuple
from torch import distributed
import numpy as np


class MyTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)
        pass

    def build_model(self):
        id2names = self.args.class_id2names
        model = dino_eva.HQDINO_EVA(class_id2names=id2names, **self.args.model_argument)
        return model
    
    def collate_fn(self, batch):
        images = [b['img'] for b in batch]
        boxes_xyxy = [b['bboxes_xyxy'] for b in batch]
        labels = [b['cls'] for b in batch]
        
        batch_data = dino_eva.HQDINO_EVA.imgs_to_batch(images, boxes=boxes_xyxy, labels=labels)
        
        batch_data_dict = {
            'inputs': batch_data,
            'image_id': [b['image_id'] for b in batch],
            'bboxes_xyxy': torch.cat([b['bboxes_xyxy'] for b in batch], 0),
            'cls': torch.cat([b['cls'] for b in batch], 0),
            'batch_idx': torch.cat([b['batch_idx']+i for i, b in enumerate(batch)], 0),
        }

        return batch_data_dict
        
    def build_dataset(self, train_transforms=None, val_transforms=None):
        # Load the dataset using the specified path and device
        path_train = os.path.join(self.args.data_path, "train")
        path_val = os.path.join(self.args.data_path, "valid")
        image_path_train = path_train
        image_path_val = path_val
        annotation_file_train = os.path.join(path_train, "_annotations.coco.json")
        annotation_file_val = os.path.join(path_val, "_annotations.coco.json")

        train_transforms.extend([augment.Pad(min_size=256)])
        val_transforms.extend([augment.Pad(min_size=256)])

        dataset_train = CocoDetection(
            image_path_train, annotation_file_train, transforms=train_transforms
        )
        dataset_val = CocoDetection(
            image_path_val, annotation_file_val, transforms=val_transforms
        )
        return dataset_train, dataset_val



def run(
        data_path, output_path, num_epoches, lr0, load_checkpoint, eval_class_names=None, batch_size=4, image_size=1024,
        gradient_update_interval=1, devices=[0], lr_backbone_mult=0.1, num_data_workers=12, checkpoint_name='ckpt.pth'
    ):
    trainer = MyTrainer(
        HQTrainerArguments(
            data_path=data_path,
            num_epoches=num_epoches,
            warmup_epochs=0,
            num_data_workers=num_data_workers,
            lr0=lr0,
            lr_min=1e-6,
            lr_backbone_mult=lr_backbone_mult,
            batch_size=batch_size,
            device='cuda:0',
            checkpoint_path=output_path,
            output_path=output_path,
            checkpoint_interval=-1,
            image_size=image_size,
            model_argument={
                "model": load_checkpoint,
            },
            eval_class_names=eval_class_names,
            gradient_update_interval=gradient_update_interval,
            devices=devices,
            checkpoint_name=checkpoint_name,
        )
    )
    trainer.run()
    pass