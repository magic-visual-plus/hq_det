import sys
from hq_det.models import rfdetr
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
import os
import torch
import torch.optim
from hq_det.models.rfdetr.util.misc import nested_tensor_from_tensor_list
from hq_det import torch_utils

class MyTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)
        pass

    def build_model(self):
        # Load the YOLO model using the specified path and device
        id2names = self.args.class_id2names
        self.args.model_argument.update({
            'dataset_dir': self.args.data_path,
            'num_classes': len(id2names)
        })
        model = rfdetr.HQRFDETR(class_id2names=id2names, **self.args.model_argument)
        return model
    
    def collate_fn(self, batch):

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

        samples = []
        targets = []
        for b in batch:  # Change from (H,W,C) to (C,H,W)
            samples.append(b['img'].clone().detach())
            targets.append({
                'boxes': b['bboxes'].clone().detach(),
                'labels': b['labels'].clone().detach(),
                'iscrowd': b['iscrowd'].clone().detach(),
                'orig_size': b['orig_size'].clone().detach(),
                'cls': b['cls'].clone().detach(),
                'image_id': b['image_id'],
                'batch_idx': b['batch_idx'].clone().detach(),
                'size': b['size'].clone().detach(),
                'area': b['area'].clone().detach(),
                'bboxes_xyxy': b['bboxes_xyxy'].clone().detach(),
                'bboxes_cxcywh_norm': b['bboxes_cxcywh_norm'].clone().detach(),
            })
        samples = nested_tensor_from_tensor_list(samples)
        return samples, targets

    def data_preprocessor(self, batch_data, training=True):
        batch_data.update(self.model.data_preprocessor(batch_data, training))
        return batch_data
    

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


