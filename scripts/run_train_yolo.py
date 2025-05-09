import sys
from hq_det.models import yolo
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.dataset import CocoDetection
from hq_det import augment
import os
import torch
import torch.optim
from hq_det import torch_utils
from ultralytics.utils import DEFAULT_CFG


class MyTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)
        pass

    def build_model(self):
        # Load the YOLO model using the specified path and device
        id2names = self.args.class_id2names
        model = yolo.HQYOLO(model=self.args.model_argument["model_path"], class_id2names=id2names)
        return model
    
    def collate_fn(self, batch):

        max_h, max_w = 0, 0
        for b in batch:
            h, w = b["img"].shape[1:]
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            
        max_h = round(max_h / 32) * 32
        max_w = round(max_w / 32) * 32

        for b in batch:
            b['img'], b['bboxes_cxcywh_norm'] = torch_utils.pad_image(b['img'], b['bboxes_cxcywh_norm'], (max_h, max_w))
            pass
    
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"} or k.startswith("bboxes_"):
                value = torch.cat(value, 0)
            new_batch[k] = value
            pass
        
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        new_batch['bboxes'] = new_batch['bboxes_cxcywh_norm']
        return new_batch
    

    def build_dataset(self, train_transforms=None, val_transforms=None):
        # Load the dataset using the specified path and device
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



if __name__ == '__main__':
    trainer = MyTrainer(
        HQTrainerArguments(
            data_path=sys.argv[1],
            num_epoches=50,
            warmup_epochs=0,
            num_data_workers=8,
            lr0=1e-4,
            lr_min=1e-6,
            batch_size=4,
            device='cuda:0',
            checkpoint_interval=-1,
            model_argument={
                "model_path": sys.argv[2]
            },
            image_size=1024,
        )
    )
    trainer.run()
    pass