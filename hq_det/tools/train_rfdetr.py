import sys
from loguru import logger
from hq_det.models import rfdetr
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.models.rfdetr import datasets as rfdetr_datasets
import os
import torch
import torch.optim
import torchvision
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

        samples = []
        targets = []
        for b in batch:  # Change from (H,W,C) to (C,H,W)
            img = b.pop('img')
            samples.append(img.clone().detach())
            targets.append(
                {
                    'boxes': safe_to_tensor(b['bboxes'], dtype=torch.float32),
                    'labels': safe_to_tensor(b['cls'], dtype=torch.int),
                    'iscrowd': safe_to_tensor(b['iscrowd'], dtype=torch.int),
                    'area': safe_to_tensor(b['area'], dtype=torch.float32),
                    'image_id': b['image_id'],
                    # 'batch_idx': b['batch_idx'],
                    'orig_size': safe_to_tensor(b['original_shape'], dtype=torch.int64),
                    'size': safe_to_tensor(b['size'], dtype=torch.int),        
                }
            )
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



class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = rfdetr_datasets.coco.ConvertCoco()
        self.id2names = {}
        for item in self.coco.cats.values():
            self.id2names[item['id']] = item['name']
        logger.info("id 2 names {}", self.id2names)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        target['img'] = img
        target['cls'] = target['labels'].numpy()
        target['bboxes'] = target['boxes'].numpy()
        target.pop('labels')
        target.pop('boxes')
        target['original_shape'] = img.size
        
        
        if self._transforms is not None:
            target = self._transforms(target)

        return target
    
    @property
    def class_id2names(self):
        return self.id2names

