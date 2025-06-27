from hq_det.models import rfdetr
from hq_det.trainer import HQTrainer, HQTrainerArguments
import os
import torch
from hq_det import augment
import torch.optim
from hq_det.dataset import CocoDetection as HQCocoDetection
from hq_det.models.rfdetr.util.misc import nested_tensor_from_tensor_list
from hq_det.models.rfdetr.engine import evaluate
from hq_det.models.rfdetr.datasets import get_coco_api_from_dataset
from hq_det.models.rfdetr.datasets import build_dataset
import hq_det.models.rfdetr.util.misc as utils



class MyTrainer(HQTrainer):

    def build_model(self) -> rfdetr.HQRFDETR:
        id2names = self.args.class_id2names
        self.args.model_argument.update({
            'dataset_dir': self.args.data_path,
            'num_classes': len(id2names)
        })
        model = rfdetr.HQRFDETR(class_id2names=id2names, **self.args.model_argument)
        print(model.args)
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

        targets = []
        imgs = []
        for b in batch:
            img = b['img']
            target = {
                'boxes': b['bboxes_cxcywh_norm'],
                'labels': safe_to_tensor(b['cls'], dtype=torch.int64),
                'iscrowd': safe_to_tensor(b['iscrowd'], dtype=torch.int64),
                'area': safe_to_tensor(b['area'], dtype=torch.float32),
                # 'orig_size': safe_to_tensor(b['original_shape'], dtype=torch.int64),
                'orig_size': safe_to_tensor(img.shape[1:3], dtype=torch.int64),
                'size': safe_to_tensor(img.shape[1:3], dtype=torch.int64),
                'image_id': safe_to_tensor(b['image_id'], dtype=torch.int64),
            }

            targets.append(target)
            imgs.append(img)

        new_batch = {
            'targets': tuple(targets),
            'image_id': torch.tensor([b['image_id'] for b in batch], dtype=torch.int64),
            'bboxes_xyxy': torch.cat([b['bboxes_xyxy'] for b in batch], 0),
            'cls': torch.cat([b['labels'] for b in targets], 0),
            'batch_idx': torch.cat([b['batch_idx']+i for i, b in enumerate(batch)], 0),
            'img': nested_tensor_from_tensor_list(imgs)
        }
        
        return new_batch
    
    def build_transforms(self, aug=True):
        transforms = super().build_transforms(aug)
        transforms.extend([
            augment.BGR2RGB(),
            augment.ToTensor(),
            augment.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return transforms

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