from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.models import dino
from hq_det.dataset import CocoDetection, CombinedDataset
from hq_det.tools.train_dino import MyTrainer as DinoTrainer
import torch
import sys
import os
from hq_det import augment


class CombinedTrainer(HQTrainer):
    def __init__(self, trainer: HQTrainer, input_paths, input_weights, valid_path):
        super().__init__(trainer.args)

        self.input_paths = input_paths
        self.input_weights = input_weights
        self.valid_path = valid_path
        self.trainer = trainer
        pass

    def build_model(self):
        return self.trainer.build_model()
    
    def collate_fn(self, batch):
        return self.trainer.collate_fn(batch)
    
    def build_dataset(self, train_transforms=None, val_transforms=None):
        datasets = []
        dataset_valid = None
        train_transforms.extend([augment.Pad(min_size=256)])
        val_transforms.extend([augment.Pad(min_size=256)])

        for i, path in enumerate(self.input_paths):
            annotation_file = os.path.join(path, "_annotations.coco.json")

            dataset_train = CocoDetection(
                path, annotation_file, transforms=train_transforms
            )
            datasets.append(dataset_train)
            pass

        annotation_file_valid = os.path.join(self.valid_path, "_annotations.coco.json")
        dataset_valid = CocoDetection(
            self.valid_path, annotation_file_valid, transforms=val_transforms
        )

        combined_dataset = CombinedDataset(datasets, self.input_weights)

        return combined_dataset, dataset_valid
    pass

if __name__ == '__main__':
    pretrained_path = sys.argv[1]
    base_trainer = DinoTrainer(
        HQTrainerArguments(
            data_path="",
            num_epoches=100,
            warmup_epochs=0,
            num_data_workers=12,
            lr0=1e-4,
            lr_min=1e-6,
            lr_backbone_mult=0.1,
            batch_size=1,
            device='cuda:0',
            checkpoint_path="output",
            output_path="output",
            checkpoint_interval=-1,
            image_size=1536,
            model_argument={
                "model": pretrained_path,
            },
            eval_class_names=[],
            gradient_update_interval=2,
            devices=[0],
            checkpoint_name='ckpt.pth',
        ),
        )
    trainer = CombinedTrainer(
        base_trainer,
        input_paths = [sys.argv[2], sys.argv[3]],
        input_weights = [1.0, 0.1],
        valid_path = sys.argv[4],
    )
    trainer.run()
    pass