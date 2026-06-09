import sys
from hq_det.models import dino
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det import augment
import os
import torch
import torch.optim
from hq_det import torch_utils
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from . import tools_mmdet


class DinoTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)
        pass

    def build_model(self):
        # Load the YOLO model using the specified path and device
        id2names = self.args.class_id2names
        model = dino.HQDINO(class_id2names=id2names, **self.args.model_argument)
        return model
    
    def collate_fn(self, batch):

        return tools_mmdet.collate_fn(batch)
    
    def build_train_transforms(self, image_size, p=0.3):
        transforms = super().build_train_transforms(image_size, p)
        return transforms.extend([augment.Pad(min_size=256)])
    
    def build_valid_transforms(self, image_size):
        transforms = super().build_valid_transforms(image_size)
        return transforms.extend([augment.Pad(min_size=256)])



def run(
        data_path, output_path, num_epoches, lr0, load_checkpoint, eval_class_names=None, batch_size=4, image_size=1024,
        gradient_update_interval=1, devices=[0], lr_backbone_mult=0.1, num_data_workers=12, checkpoint_name='ckpt.pth',
        augment_split_size=-1, augment_split_proba=0.5, augment_foreground_path="",
        augment_foreground_proba=0.8,
        dino_config_name=None,
        focal_loss_alpha=None, focal_loss_gamma=None,
        focal_loss_gamma_pos=None, focal_loss_gamma_neg=None,
    ):
    model_argument = {"model": load_checkpoint, "image_size": image_size}
    if dino_config_name:
        model_argument["config_name"] = dino_config_name
    for key in ('focal_loss_alpha', 'focal_loss_gamma', 'focal_loss_gamma_pos', 'focal_loss_gamma_neg'):
        if locals()[key] is not None:
            model_argument[key] = locals()[key]
    trainer = DinoTrainer(
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
            model_argument=model_argument,
            eval_class_names=eval_class_names,
            gradient_update_interval=gradient_update_interval,
            devices=devices,
            checkpoint_name=checkpoint_name,
            augment_split_size=augment_split_size,
            augment_split_proba=augment_split_proba,
            augment_foreground_path=augment_foreground_path,
            augment_foreground_proba=augment_foreground_proba
        )
    )
    trainer.run()
    pass