import sys
from loguru import logger
from hq_det.models import rfdetr
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.models.rfdetr import datasets as rfdetr_datasets
import os
import torch
import torch.optim
import torchvision
from hq_det import torch_utils
from hq_det.dataset import CocoDetection as HQCocoDetection

class MyTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)

    def build_model(self):
        id2names = self.args.class_id2names
        self.args.model_argument.update({
            'dataset_dir': self.args.data_path,
            'num_classes': len(id2names)
        })
        model = rfdetr.HQRFDETR(class_id2names=id2names, **self.args.model_argument)
        return model
    
    def collate_fn(self, batch):
        def safe_to_tensor(data, dtype=None):  
            if isinstance(data, torch.Tensor):  
                result = data.clone().detach()  
                if dtype is not None:  
                    result = result.to(dtype=dtype)
            else:  
                result = torch.as_tensor(data, dtype=dtype)  
            
            return result
        max_h, max_w = 0, 0
        for b in batch:
            h, w = b["img"].shape[:2]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            
        max_h = round(max_h / 56) * 56
        max_w = round(max_w / 56) * 56

        for b in batch:
            if not isinstance(b['img'], torch.Tensor):
                b['img'] = torch.tensor(b['img'], dtype=torch.float32).permute(2, 0, 1)
            b['img'], b['bboxes_cxcywh_norm'] = torch_utils.pad_image(b['img'], b['bboxes_cxcywh_norm'], (max_h, max_w))

        targets = [{
            'boxes': safe_to_tensor(b['bboxes_cxcywh_norm'], dtype=torch.float32),
            'labels': safe_to_tensor(b['cls'], dtype=torch.int64),
            'iscrowd': safe_to_tensor(b['iscrowd'], dtype=torch.int64),
            'area': safe_to_tensor(b['area'], dtype=torch.float32),
            'image_id': b['image_id'],
            'orig_size': safe_to_tensor(b['original_shape'], dtype=torch.int64),
            'size': safe_to_tensor(b['size'], dtype=torch.int64),
        } for b in batch]

        return {
            'targets': tuple(targets),
            'image_id': [b['image_id'] for b in batch],
            'bboxes_xyxy': torch.cat([b['bboxes_xyxy'] for b in batch], 0),
            'cls': torch.cat([b['cls'] for b in batch], 0),
            'batch_idx': torch.cat([b['batch_idx']+i for i, b in enumerate(batch)], 0),
            'img': torch.stack([b['img'] for b in batch], 0)
        }

    def build_dataset(self, train_transforms=None, val_transforms=None):
        # Load the dataset using the specified path and device
        path_train = os.path.join(self.args.data_path, "train")
        path_val = os.path.join(self.args.data_path, "valid")
        image_path_train = path_train
        image_path_val = path_val
        annotation_file_train = os.path.join(path_train, "_annotations.coco.json")
        annotation_file_val = os.path.join(path_val, "_annotations.coco.json")

        dataset_train = HQCocoDetection(
            image_path_train, annotation_file_train, transforms=train_transforms
        )
        dataset_val = HQCocoDetection(
            image_path_val, annotation_file_val, transforms=val_transforms
        )
        return dataset_train, dataset_val
    
    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, total_iters=self.args.num_epoches,
            end_factor=self.args.lr_min / self.args.lr0
        )


