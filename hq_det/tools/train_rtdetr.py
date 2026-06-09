import sys
from hq_det.models import rtdetr
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det import augment
import os
import torch
import torch.optim
from hq_det import torch_utils
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


class RtDetrTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)
        pass

    def build_model(self):
        # Load the YOLO model using the specified path and device
        id2names = self.args.class_id2names
        model = rtdetr.HQRTDETR(class_id2names=id2names, **self.args.model_argument)
        return model
    
    def collate_fn(self, batch):

        max_h, max_w = self.args.image_size, self.args.image_size
        
        for b in batch:
            b['img'], b['bboxes_cxcywh_norm'] = torch_utils.pad_image(b['img'], b['bboxes_cxcywh_norm'], (max_h, max_w))
            pass
    
        new_batch = {}
        new_batch['image_id'] = [b['image_id'] for b in batch]
        new_batch['bboxes_xyxy'] = torch.cat([b['bboxes_xyxy'] for b in batch], 0)
        new_batch['cls'] = torch.cat([b['cls'] for b in batch], 0)
        new_batch['batch_idx'] = torch.cat([b['batch_idx']+i for i, b in enumerate(batch)], 0)
        new_batch['img'] = torch.stack([b['img'] for b in batch], 0)
        new_batch['targets'] = [
            {
                "boxes": b["bboxes_cxcywh_norm"],
                "labels": b["cls"],
                "image_id": b["image_id"],
            }
            for b in batch
        ]
        
        return new_batch
    
    def build_train_transforms(self, image_size, proba):
        transforms = super().build_train_transforms(image_size, proba)
        transforms.extend([
            augment.BGR2RGB(),
            augment.ToTensor(),
        ])
        return transforms

    def build_valid_transforms(self, image_size):
        transforms = super().build_valid_transforms(image_size)
        transforms.extend([
            augment.BGR2RGB(),
            augment.ToTensor(),
        ])
        return transforms


def run(data_path, output_path, num_epoches, lr0, load_checkpoint, eval_class_names=None, batch_size=6, image_size=1024,
        gradient_update_interval=1, lr_backbone_mult=1.0, num_data_workers=16, checkpoint_name='ckpt.pth',
        devices=[0],
        augment_foreground_path="", augment_foreground_proba=0.0,
    ):
    trainer = RtDetrTrainer(
        HQTrainerArguments(
            data_path=data_path,
            num_epoches=num_epoches,
            warmup_epochs=0,
            num_data_workers=num_data_workers,
            lr0=lr0,
            lr_min=1e-6,
            lr_backbone_mult=lr_backbone_mult,
            batch_size=batch_size,
            devices=devices,
            output_path=output_path,
            checkpoint_path=output_path,
            checkpoint_interval=-1,
            image_size=image_size,
            model_argument={
                "model": load_checkpoint,
                "image_size": image_size,
            },
            eval_class_names=eval_class_names,
            gradient_update_interval=gradient_update_interval,
            checkpoint_name=checkpoint_name,
            find_unused_parameters=True,
            augment_foreground_path=augment_foreground_path,
            augment_foreground_proba=augment_foreground_proba,
        )
    )
    trainer.run()
    pass
