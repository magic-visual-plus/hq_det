import sys
import os
from hq_det.models import deim
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det import augment
import torch
import torch.optim
from hq_det import torch_utils


class MyTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)

    def build_model(self):
        id2names = self.args.class_id2names
        model = deim.HQDEIM(class_id2names=id2names, **self.args.model_argument)
        return model

    def collate_fn(self, batch):
        max_h, max_w = self.args.image_size, self.args.image_size

        for b in batch:
            b['img'], b['bboxes_cxcywh_norm'] = torch_utils.pad_image(
                b['img'], b['bboxes_cxcywh_norm'], (max_h, max_w))

        new_batch = {}
        new_batch['image_id'] = [b['image_id'] for b in batch]
        new_batch['bboxes_xyxy'] = torch.cat([b['bboxes_xyxy'] for b in batch], 0)
        new_batch['cls'] = torch.cat([b['cls'] for b in batch], 0)
        new_batch['batch_idx'] = torch.cat([b['batch_idx'] + i for i, b in enumerate(batch)], 0)
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

    def build_dataset(self, train_transforms=None, val_transforms=None):
        path_train = os.path.join(self.args.data_path, "train")
        path_val = os.path.join(self.args.data_path, "valid")
        image_path_train = path_train
        image_path_val = path_val
        annotation_file_train = os.path.join(path_train, "_annotations.coco.json")
        annotation_file_val = os.path.join(path_val, "_annotations.coco.json")

        train_transforms.extend([
            augment.BGR2RGB(),
            augment.ToTensor(),
        ])
        val_transforms.extend([
            augment.BGR2RGB(),
            augment.ToTensor(),
        ])

        dataset_train = CocoDetection(
            image_path_train, annotation_file_train, transforms=train_transforms
        )
        dataset_val = CocoDetection(
            image_path_val, annotation_file_val, transforms=val_transforms
        )
        return dataset_train, dataset_val

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, total_iters=self.args.num_epoches,
            end_factor=self.args.lr_min / self.args.lr0
        )


def run(data_path, output_path, num_epoches, lr0, load_checkpoint, eval_class_names=None, 
        batch_size=6, image_size=1024, gradient_update_interval=1, lr_backbone_mult=1.0, 
        num_data_workers=16, checkpoint_name='ckpt.pth', devices=[0], config_path=None,
        warmup_epochs=0, model_size='l', model_name=None, series='dfine', model_type='deim'):
    """
    Run DEIM training
    
    Args:
        series: Model series, 'dfine' or 'rtdetrv2' (default: 'dfine')
        model_type: Model type, 'deim' or 'dfine'/'rtdetrv2' (default: 'deim')
        model_size: Model size:
            - For dfine: 'n', 's', 'm', 'l', 'x' (default: 's')
            - For rtdetrv2: 'r18vd', 'r34vd', 'r50vd', 'r50vd_m', 'r101vd' (default: 'r50vd')
        model_name: Full model config name, e.g., 'deim_hgnetv2_s_coco' (overrides above)
        config_path: Full path to config file (overrides all above)
    """
    model_argument = {
        "image_size": image_size,
    }
    
    if load_checkpoint:
        model_argument["model"] = load_checkpoint
    
    if config_path:
        model_argument["config_path"] = config_path
    elif model_name:
        model_argument["model_name"] = model_name
    else:
        model_argument["series"] = series
        model_argument["model_type"] = model_type
        model_argument["model_size"] = model_size
    
    trainer = MyTrainer(
        HQTrainerArguments(
            data_path=data_path,
            num_epoches=num_epoches,
            warmup_epochs=warmup_epochs,
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
            model_argument=model_argument,
            eval_class_names=eval_class_names,
            gradient_update_interval=gradient_update_interval,
            checkpoint_name=checkpoint_name,
            find_unused_parameters=True,
            sync_bn=True,
        )
    )
    trainer.run()

