import sys
from hq_det.models.lwdetr import hq_lwdetr
from hq_det.trainer import HQTrainer, HQTrainerArguments
from hq_det.models.lwdetr.datasets.coco import CocoDetection
from hq_det import augment
import os
import torch
import torch.optim
from hq_det import torch_utils
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from hq_det.models.lwdetr.util import misc as utils
from hq_det.models.lwdetr.datasets.coco import make_coco_transforms_square_div_64, make_coco_transforms
from hq_det.models.lwdetr.configs.config_loader import ConfigLoader


class MyTrainer(HQTrainer):
    def __init__(self, args: HQTrainerArguments):
        super().__init__(args)
        model_name = args.model_argument['model_name']
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'lwdetr', 'configs')
        config_loader = ConfigLoader(config_dir=model_dir)
        self.model_args = config_loader.get_args(model_name)

    def build_model(self):
        # Load the YOLO model using the specified path and device
        id2names = self.args.class_id2names
        model = hq_lwdetr.HQLWDETR(class_id2names=id2names, **self.args.model_argument)
        return model
    
    def collate_fn(self, batch):
        return utils.collate_fn(batch)
    

    def build_dataset(self, train_transforms=None, val_transforms=None):
        def build_coco_transforms(image_set, img_folder, ann_file):
            try:
                square_resize_div_64 = self.model_args.square_resize_div_64
            except:
                square_resize_div_64 = False

            
            if square_resize_div_64:
                dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms_square_div_64(image_set))
            else:
                dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set))

            return dataset
        # Load the dataset using the specified path and device
        path_train = os.path.join(self.args.data_path, "train")
        path_val = os.path.join(self.args.data_path, "valid")
        image_path_train = path_train
        image_path_val = path_val
        annotation_file_train = os.path.join(path_train, "_annotations.coco.json")
        annotation_file_val = os.path.join(path_val, "_annotations.coco.json")

        dataset_train = build_coco_transforms('train', image_path_train, annotation_file_train)
        dataset_val = build_coco_transforms('val', image_path_val, annotation_file_val)
        return dataset_train, dataset_val
    

    # def build_optimizer(self, model):
    #     print(type(model))
    #     param_dict = model.get_param_dict(self.args)
    #     print('optimizer param dict:')
    #     print(param_dict)
    #     return torch.optim.AdamW(param_dict, lr=self.args.lr0)
    
    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, total_iters=self.args.num_epoches,
            end_factor=self.args.lr_min / self.args.lr0
        )