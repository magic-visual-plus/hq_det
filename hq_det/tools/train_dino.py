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


class MyTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)
        pass

    def build_model(self):
        # Load the YOLO model using the specified path and device
        id2names = self.args.class_id2names
        model = dino.HQDINO(class_id2names=id2names, **self.args.model_argument)
        return model
    
    def collate_fn(self, batch):

        new_batch = {}
        new_batch['inputs'] = [torch.permute(torch.from_numpy(b['img']), (2, 0, 1)).contiguous() for b in batch]
        new_batch['image_id'] = [b['image_id'] for b in batch]
        new_batch['bboxes_xyxy'] = torch.cat([b['bboxes_xyxy'] for b in batch], 0)
        new_batch['cls'] = torch.cat([b['cls'] for b in batch], 0)
        new_batch['batch_idx'] = torch.cat([b['batch_idx']+i for i, b in enumerate(batch)], 0)

        data_samples = []

        for i, b in enumerate(batch):
            data_sample = DetDataSample(metainfo={
                'img_shape': (b['img'].shape[0], b['img'].shape[1]),
            })
            gt_instance = InstanceData()
            gt_instance.bboxes = b['bboxes_xyxy']
            gt_instance.labels = b['cls']
            data_sample.gt_instances = gt_instance
            data_samples.append(data_sample)
            pass

        new_batch['data_samples'] = data_samples
        return new_batch
    

    def build_dataset(self, train_transforms=None, val_transforms=None):
        # Load the dataset using the specified path and device
        path_train = os.path.join(self.args.data_path, "train")
        path_val = os.path.join(self.args.data_path, "valid")
        image_path_train = path_train
        image_path_val = path_val
        annotation_file_train = os.path.join(path_train, "_annotations.coco.json")
        annotation_file_val = os.path.join(path_val, "_annotations.coco.json")

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



def run(
        data_path, output_path, num_epoches, lr0, load_checkpoint, eval_class_names=None, batch_size=4, image_size=1024,
        gradient_update_interval=1, devices=[0]):
    trainer = MyTrainer(
        HQTrainerArguments(
            data_path=data_path,
            num_epoches=num_epoches,
            warmup_epochs=0,
            num_data_workers=12,
            lr0=lr0,
            lr_min=1e-6,
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
        )
    )
    trainer.run()
    pass