import sys
from hq_det.models import codetr
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
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
        model = codetr.HQCoDetr(class_id2names=id2names, **self.args.model_argument)
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


